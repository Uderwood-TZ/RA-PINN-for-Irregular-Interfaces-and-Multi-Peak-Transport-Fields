
import os
import time
import math
import random
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

SEED = 20260308
USE_FLOAT64 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CASE_NAME = "case1_lstm_pinn"
OUTPUT_ROOT = f"outputs_{CASE_NAME}"

FIG_DIRNAME = "figures"
DATA_DIRNAME = "data"
LOG_DIRNAME = "logs"
MODEL_DIRNAME = "models"

DOMAIN_X_MIN = -1.0
DOMAIN_X_MAX = 1.0
DOMAIN_Y_MIN = -1.0
DOMAIN_Y_MAX = 1.0

TRAIN_STEPS = 50000
LEARNING_RATE = 8.0e-4
MIN_LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-10
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 500
VAL_INTERVAL = 200

HIDDEN_DIM = 128
NUM_BLOCKS = 6
MLP_DEPTH = 6

TRAIN_RATIO = 0.7
SUPERVISED_GRID_NX = 91
SUPERVISED_GRID_NY = 91
EVAL_GRID_NX = 181
EVAL_GRID_NY = 181
N_COLL_POINTS = 16000
N_BC_PER_EDGE = 360

DATA_BATCH_SIZE = 1024
COLL_BATCH_SIZE = 2048
BC_BATCH_SIZE = 1024

LOSS_WEIGHT_PDE = 1.0
LOSS_WEIGHT_DATA = 18.0
LOSS_WEIGHT_BC = 25.0
PDE_WEIGHT_WARMUP = 0.25
WARMUP_STEPS = 5000

RES_WEIGHT_CONT = 2.0
RES_WEIGHT_MOMX = 1.0
RES_WEIGHT_MOMY = 1.0
RES_WEIGHT_ENERGY = 1.0
RES_WEIGHT_PHI = 1.0

VISCOSITY_NU = 0.035
THERMAL_DIFFUSIVITY = 0.020
ELECTRIC_FORCE_COEFF = 0.45
THERMAL_FORCE_COEFF = 0.22
JOULE_HEATING_COEFF = 0.08
PHI_REACTION_COEFF = 1.15

JET_CMAP = "jet"

PSI_A = 0.30
PSI_B = 0.12
PSI_BETA_X = 3.5
PSI_BETA_Y = 3.0

PRESSURE_A = 0.80
PRESSURE_B = 0.25
PRESSURE_C = 0.15
PRESSURE_BETA_1 = 3.2
PRESSURE_BETA_2 = 2.7

TEMP_BASE = 0.50
TEMP_A = 0.70
TEMP_B = 0.18
TEMP_BETA_X = 3.8
TEMP_BETA_Y = 3.4

PHI_A = 0.65
PHI_B = 0.22
PHI_C = 0.12
PHI_BETA_1 = 3.6
PHI_BETA_2 = 2.9

if USE_FLOAT64:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(root: str) -> Dict[str, str]:
    dirs = {
        "root": root,
        "fig": os.path.join(root, FIG_DIRNAME),
        "data": os.path.join(root, DATA_DIRNAME),
        "log": os.path.join(root, LOG_DIRNAME),
        "model": os.path.join(root, MODEL_DIRNAME),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def to_device_tensor(array: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(array, device=DEVICE, dtype=torch.get_default_dtype())
    if requires_grad:
        t.requires_grad_(True)
    return t


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def build_stream_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    a = x - 0.30 * torch.sin(math.pi * y)
    b = y + 0.25 * torch.cos(math.pi * x)
    psi = PSI_A * torch.tanh(PSI_BETA_X * a) * torch.tanh(PSI_BETA_Y * b)
    psi = psi + PSI_B * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return psi


def exact_solution_from_xy(xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    psi = build_stream_function(x, y)
    grad_psi = grad(psi, xy)
    u = grad_psi[:, 1:2]
    v = -grad_psi[:, 0:1]

    p = PRESSURE_A * torch.tanh(PRESSURE_BETA_1 * (x + y - 0.15))
    p = p + PRESSURE_B * torch.sin(2.0 * math.pi * x) * torch.cos(math.pi * y)
    p = p + PRESSURE_C * torch.tanh(PRESSURE_BETA_2 * (y - x + 0.10))

    T = TEMP_BASE + TEMP_A * torch.tanh(TEMP_BETA_X * (x - 0.25)) * torch.tanh(TEMP_BETA_Y * (y + 0.20))
    T = T + TEMP_B * torch.sin(math.pi * x) * torch.sin(2.0 * math.pi * y)

    phi = PHI_A * torch.tanh(PHI_BETA_1 * (x + y + 0.10))
    phi = phi + PHI_B * torch.cos(math.pi * x) * torch.sin(math.pi * y)
    phi = phi + PHI_C * torch.tanh(PHI_BETA_2 * (x - y - 0.05))

    return u, v, p, T, phi


def split_fields(fields: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.chunk(fields, chunks=5, dim=1)


def compute_field_derivatives(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    T: torch.Tensor,
    phi: torch.Tensor,
    xy: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    gu = grad(u, xy)
    gv = grad(v, xy)
    gp = grad(p, xy)
    gT = grad(T, xy)
    gphi = grad(phi, xy)

    u_x = gu[:, 0:1]
    u_y = gu[:, 1:2]
    v_x = gv[:, 0:1]
    v_y = gv[:, 1:2]
    p_x = gp[:, 0:1]
    p_y = gp[:, 1:2]
    T_x = gT[:, 0:1]
    T_y = gT[:, 1:2]
    phi_x = gphi[:, 0:1]
    phi_y = gphi[:, 1:2]

    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    v_xx = grad(v_x, xy)[:, 0:1]
    v_yy = grad(v_y, xy)[:, 1:2]
    T_xx = grad(T_x, xy)[:, 0:1]
    T_yy = grad(T_y, xy)[:, 1:2]
    phi_xx = grad(phi_x, xy)[:, 0:1]
    phi_yy = grad(phi_y, xy)[:, 1:2]

    return {
        "u_x": u_x, "u_y": u_y, "v_x": v_x, "v_y": v_y, "p_x": p_x, "p_y": p_y,
        "T_x": T_x, "T_y": T_y, "phi_x": phi_x, "phi_y": phi_y,
        "u_xx": u_xx, "u_yy": u_yy, "v_xx": v_xx, "v_yy": v_yy,
        "T_xx": T_xx, "T_yy": T_yy, "phi_xx": phi_xx, "phi_yy": phi_yy,
    }


def compute_mms_sources(xy: torch.Tensor) -> torch.Tensor:
    u, v, p, T, phi = exact_solution_from_xy(xy)
    d = compute_field_derivatives(u, v, p, T, phi, xy)

    f_u = u * d["u_x"] + v * d["u_y"] + d["p_x"]
    f_u = f_u - VISCOSITY_NU * (d["u_xx"] + d["u_yy"])
    f_u = f_u - ELECTRIC_FORCE_COEFF * d["phi_x"] - THERMAL_FORCE_COEFF * d["T_x"]

    f_v = u * d["v_x"] + v * d["v_y"] + d["p_y"]
    f_v = f_v - VISCOSITY_NU * (d["v_xx"] + d["v_yy"])
    f_v = f_v - ELECTRIC_FORCE_COEFF * d["phi_y"] - THERMAL_FORCE_COEFF * d["T_y"]

    q_T = u * d["T_x"] + v * d["T_y"]
    q_T = q_T - THERMAL_DIFFUSIVITY * (d["T_xx"] + d["T_yy"])
    q_T = q_T - JOULE_HEATING_COEFF * (d["phi_x"] ** 2 + d["phi_y"] ** 2)

    s_phi = d["phi_xx"] + d["phi_yy"] - PHI_REACTION_COEFF * phi
    return torch.cat([f_u, f_v, q_T, s_phi], dim=1)


def exact_fields_no_grad(xy_np: np.ndarray) -> np.ndarray:
    xy = torch.tensor(xy_np, dtype=torch.get_default_dtype(), device=DEVICE, requires_grad=True)
    u, v, p, T, phi = exact_solution_from_xy(xy)
    return torch.cat([u, v, p, T, phi], dim=1).detach().cpu().numpy()


def batched_exact_fields(points: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        outputs.append(exact_fields_no_grad(points[start:start + batch_size]))
    return np.vstack(outputs)


def batched_exact_sources(points: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        batch_np = points[start:start + batch_size]
        batch = torch.tensor(batch_np, dtype=torch.get_default_dtype(), device=DEVICE, requires_grad=True)
        outputs.append(compute_mms_sources(batch).detach().cpu().numpy())
    return np.vstack(outputs)


def sample_interior_points(n_points: int) -> np.ndarray:
    x = np.random.uniform(DOMAIN_X_MIN, DOMAIN_X_MAX, size=(n_points, 1))
    y = np.random.uniform(DOMAIN_Y_MIN, DOMAIN_Y_MAX, size=(n_points, 1))
    return np.hstack([x, y])


def sample_boundary_points(n_per_edge: int) -> np.ndarray:
    xs = np.linspace(DOMAIN_X_MIN, DOMAIN_X_MAX, n_per_edge).reshape(-1, 1)
    ys = np.linspace(DOMAIN_Y_MIN, DOMAIN_Y_MAX, n_per_edge).reshape(-1, 1)
    left = np.hstack([np.full_like(ys, DOMAIN_X_MIN), ys])
    right = np.hstack([np.full_like(ys, DOMAIN_X_MAX), ys])
    bottom = np.hstack([xs, np.full_like(xs, DOMAIN_Y_MIN)])
    top = np.hstack([xs, np.full_like(xs, DOMAIN_Y_MAX)])
    all_boundary = np.vstack([left, right, bottom, top])
    return np.unique(np.round(all_boundary, decimals=12), axis=0)


def build_supervised_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(DOMAIN_X_MIN, DOMAIN_X_MAX, SUPERVISED_GRID_NX)
    y = np.linspace(DOMAIN_Y_MIN, DOMAIN_Y_MAX, SUPERVISED_GRID_NY)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    vals = batched_exact_fields(pts)
    idx = np.arange(len(pts))
    np.random.shuffle(idx)
    n_train = int(TRAIN_RATIO * len(idx))
    return pts[idx[:n_train]], vals[idx[:n_train]], pts[idx[n_train:]], vals[idx[n_train:]]


def build_eval_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(DOMAIN_X_MIN, DOMAIN_X_MAX, EVAL_GRID_NX)
    y = np.linspace(DOMAIN_Y_MIN, DOMAIN_Y_MAX, EVAL_GRID_NY)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    return X, Y, pts


def random_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x[idx], y[idx]


def mse_loss(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x ** 2)


class LSTMPINN(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()
        self.lstm1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_layer(xy))
        h = h.unsqueeze(1)
        with torch.backends.cudnn.flags(enabled=False):
            h, _ = self.lstm1(h)
            h, _ = self.lstm2(h)
        h = h.squeeze(1)
        return self.output_head(h)
    

def build_model() -> nn.Module:
    return LSTMPINN()



def pde_residuals(model: nn.Module, xy: torch.Tensor, src: torch.Tensor) -> Dict[str, torch.Tensor]:
    xy = xy.clone().detach().requires_grad_(True)
    pred = model(xy)
    u, v, p, T, phi = split_fields(pred)
    d = compute_field_derivatives(u, v, p, T, phi, xy)

    src_u = src[:, 0:1]
    src_v = src[:, 1:2]
    src_T = src[:, 2:3]
    src_phi = src[:, 3:4]

    r_cont = d["u_x"] + d["v_y"]
    r_momx = u * d["u_x"] + v * d["u_y"] + d["p_x"]
    r_momx = r_momx - VISCOSITY_NU * (d["u_xx"] + d["u_yy"])
    r_momx = r_momx - ELECTRIC_FORCE_COEFF * d["phi_x"] - THERMAL_FORCE_COEFF * d["T_x"] - src_u

    r_momy = u * d["v_x"] + v * d["v_y"] + d["p_y"]
    r_momy = r_momy - VISCOSITY_NU * (d["v_xx"] + d["v_yy"])
    r_momy = r_momy - ELECTRIC_FORCE_COEFF * d["phi_y"] - THERMAL_FORCE_COEFF * d["T_y"] - src_v

    r_energy = u * d["T_x"] + v * d["T_y"]
    r_energy = r_energy - THERMAL_DIFFUSIVITY * (d["T_xx"] + d["T_yy"])
    r_energy = r_energy - JOULE_HEATING_COEFF * (d["phi_x"] ** 2 + d["phi_y"] ** 2) - src_T

    r_phi = d["phi_xx"] + d["phi_yy"] - PHI_REACTION_COEFF * phi - src_phi

    return {
        "cont": r_cont,
        "momx": r_momx,
        "momy": r_momy,
        "energy": r_energy,
        "phi": r_phi,
    }


def compute_losses(
    model: nn.Module,
    batch_data_xy: torch.Tensor,
    batch_data_u: torch.Tensor,
    batch_bc_xy: torch.Tensor,
    batch_bc_u: torch.Tensor,
    batch_coll_xy: torch.Tensor,
    batch_coll_src: torch.Tensor,
    step: int,
) -> Dict[str, torch.Tensor]:
    pred_data = model(batch_data_xy)
    loss_data = mse_loss(pred_data - batch_data_u)

    pred_bc = model(batch_bc_xy)
    loss_bc = mse_loss(pred_bc - batch_bc_u)

    residual_dict = pde_residuals(model, batch_coll_xy, batch_coll_src)
    loss_cont = mse_loss(residual_dict["cont"])
    loss_momx = mse_loss(residual_dict["momx"])
    loss_momy = mse_loss(residual_dict["momy"])
    loss_energy = mse_loss(residual_dict["energy"])
    loss_phi = mse_loss(residual_dict["phi"])

    loss_pde = (
        RES_WEIGHT_CONT * loss_cont
        + RES_WEIGHT_MOMX * loss_momx
        + RES_WEIGHT_MOMY * loss_momy
        + RES_WEIGHT_ENERGY * loss_energy
        + RES_WEIGHT_PHI * loss_phi
    )

    current_pde_weight = PDE_WEIGHT_WARMUP if step <= WARMUP_STEPS else LOSS_WEIGHT_PDE
    loss_total = current_pde_weight * loss_pde + LOSS_WEIGHT_DATA * loss_data + LOSS_WEIGHT_BC * loss_bc

    return {
        "total": loss_total,
        "pde": loss_pde,
        "data": loss_data,
        "bc": loss_bc,
        "cont": loss_cont,
        "momx": loss_momx,
        "momy": loss_momy,
        "energy": loss_energy,
        "phi": loss_phi,
    }


@torch.no_grad()
def compute_validation_loss(model: nn.Module, val_xy: torch.Tensor, val_u: torch.Tensor) -> float:
    pred = model(val_xy)
    return torch.mean((pred - val_u) ** 2).item()


@torch.no_grad()
def predict_numpy(model: nn.Module, points: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    preds: List[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        batch = torch.tensor(points[start:start + batch_size], dtype=torch.get_default_dtype(), device=DEVICE)
        preds.append(model(batch).cpu().numpy())
    return np.vstack(preds)


def save_xyz_txt(filename: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    xyz = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    np.savetxt(filename, xyz, fmt="%.10e", header="x y value", comments="")


def save_field_figure(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, fig_path: str, data_path: str) -> None:
    save_xyz_txt(data_path, X, Y, Z)
    plt.figure(figsize=(7.5, 6.0))
    mesh = plt.pcolormesh(X, Y, Z, shading="auto", cmap=JET_CMAP)
    plt.colorbar(mesh)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def save_max_error_location_map(
    X: np.ndarray,
    Y: np.ndarray,
    abs_err: np.ndarray,
    title: str,
    fig_path: str,
    data_path: str,
) -> None:
    max_map = np.zeros_like(abs_err)
    max_val = np.max(abs_err)
    max_map[np.isclose(abs_err, max_val)] = max_val
    save_field_figure(X, Y, max_map, title, fig_path, data_path)


def compute_metrics(pred: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
    err = pred - exact
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    l2 = float(np.linalg.norm(err.reshape(-1)) / (np.linalg.norm(exact.reshape(-1)) + 1.0e-12))
    max_abs = float(np.max(np.abs(err)))
    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "L2": l2, "MaxAbs": max_abs}


def save_loss_curves(history_array: np.ndarray, out_dir: str) -> None:
    names = ["total", "pde", "data", "bc", "cont", "momx", "momy", "energy", "phi", "val"]
    step = history_array[:, 0]
    cols = {
        "total": history_array[:, 1],
        "pde": history_array[:, 2],
        "data": history_array[:, 3],
        "bc": history_array[:, 4],
        "cont": history_array[:, 5],
        "momx": history_array[:, 6],
        "momy": history_array[:, 7],
        "energy": history_array[:, 8],
        "phi": history_array[:, 9],
        "val": history_array[:, 10],
    }

    for is_log in [False, True]:
        plt.figure(figsize=(9.0, 6.5))
        for name in names:
            values = cols[name]
            mask = np.isfinite(values)
            plt.plot(step[mask], values[mask], label=name)
        if is_log:
            plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"{CASE_NAME} loss curves" + (" (log scale)" if is_log else ""))
        plt.legend()
        plt.tight_layout()
        name = "loss_curves_log.png" if is_log else "loss_curves_linear.png"
        plt.savefig(os.path.join(out_dir, name), dpi=300)
        plt.close()


def save_all_outputs(
    model: nn.Module,
    dirs: Dict[str, str],
    history_array: np.ndarray,
    training_time: float,
    total_start_time: float,
) -> None:
    save_loss_curves(history_array, dirs["fig"])
    X, Y, eval_pts = build_eval_grid()
    eval_start = time.time()
    pred = predict_numpy(model, eval_pts)
    exact = batched_exact_fields(eval_pts)
    eval_time = time.time() - eval_start
    total_time = time.time() - total_start_time

    fields = ["u", "v", "p", "T", "phi"]
    metrics_lines = ["field RMSE MSE MAE L2 MaxAbs"]
    for i, name in enumerate(fields):
        pred_i = pred[:, i].reshape(X.shape)
        exact_i = exact[:, i].reshape(X.shape)
        abs_err_i = np.abs(pred_i - exact_i)
        metrics = compute_metrics(pred_i, exact_i)
        metrics_lines.append(
            f"{name} {metrics['RMSE']:.10e} {metrics['MSE']:.10e} {metrics['MAE']:.10e} {metrics['L2']:.10e} {metrics['MaxAbs']:.10e}"
        )
        save_field_figure(X, Y, exact_i, f"{CASE_NAME} - exact {name}", os.path.join(dirs["fig"], f"{name}_exact.png"), os.path.join(dirs["data"], f"{name}_exact.txt"))
        save_field_figure(X, Y, pred_i, f"{CASE_NAME} - prediction {name}", os.path.join(dirs["fig"], f"{name}_prediction.png"), os.path.join(dirs["data"], f"{name}_prediction.txt"))
        save_field_figure(X, Y, abs_err_i, f"{CASE_NAME} - abs error {name}", os.path.join(dirs["fig"], f"{name}_abs_error.png"), os.path.join(dirs["data"], f"{name}_abs_error.txt"))
        save_max_error_location_map(X, Y, abs_err_i, f"{CASE_NAME} - max error map {name}", os.path.join(dirs["fig"], f"{name}_max_error_map.png"), os.path.join(dirs["data"], f"{name}_max_error_map.txt"))

    with open(os.path.join(dirs["log"], "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines) + "\n")

    with open(os.path.join(dirs["log"], "timing.txt"), "w", encoding="utf-8") as f:
        f.write(f"training_time_seconds {training_time:.10f}\n")
        f.write(f"evaluation_time_seconds {eval_time:.10f}\n")
        f.write(f"total_time_seconds {total_time:.10f}\n")

    np.savetxt(
        os.path.join(dirs["log"], "loss_history_every_step.txt"),
        history_array,
        fmt="%.10e",
        header="step total_loss pde_loss data_loss bc_loss cont_loss momx_loss momy_loss energy_loss phi_loss val_loss lr",
        comments="",
    )


def write_case_description(filepath: str) -> None:
    text = f"""
Case: {CASE_NAME}
PDEs on [{DOMAIN_X_MIN}, {DOMAIN_X_MAX}] x [{DOMAIN_Y_MIN}, {DOMAIN_Y_MAX}]

1) u_x + v_y = 0
2) u u_x + v u_y + p_x - nu (u_xx + u_yy) - c_e phi_x - c_T T_x = f_u
3) u v_x + v v_y + p_y - nu (v_xx + v_yy) - c_e phi_y - c_T T_y = f_v
4) u T_x + v T_y - alpha (T_xx + T_yy) - gamma_J (phi_x^2 + phi_y^2) = q_T
5) phi_xx + phi_yy - lambda_phi phi = s_phi

Exact MMS:
psi(x, y) = {PSI_A} * tanh({PSI_BETA_X} * (x - 0.30 sin(pi y))) * tanh({PSI_BETA_Y} * (y + 0.25 cos(pi x))) + {PSI_B} * sin(pi x) sin(pi y)
u = d psi / d y, v = - d psi / d x
p(x, y), T(x, y), phi(x, y) are defined in the script and used to construct source terms.

This script uses data loss, boundary loss and PDE residual loss with 7:3 train/validation split for supervised MMS data.
""".strip()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    set_seed(SEED)
    dirs = ensure_dirs(OUTPUT_ROOT)
    total_start = time.time()

    print(f"Using device: {DEVICE}")
    print("Building MMS datasets ...")
    train_xy_np, train_u_np, val_xy_np, val_u_np = build_supervised_dataset()
    coll_xy_np = sample_interior_points(N_COLL_POINTS)
    coll_src_np = batched_exact_sources(coll_xy_np)
    bc_xy_np = sample_boundary_points(N_BC_PER_EDGE)
    bc_u_np = batched_exact_fields(bc_xy_np)

    train_xy = to_device_tensor(train_xy_np)
    train_u = to_device_tensor(train_u_np)
    val_xy = to_device_tensor(val_xy_np)
    val_u = to_device_tensor(val_u_np)
    coll_xy = to_device_tensor(coll_xy_np)
    coll_src = to_device_tensor(coll_src_np)
    bc_xy = to_device_tensor(bc_xy_np)
    bc_u = to_device_tensor(bc_u_np)

    model = build_model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS, eta_min=MIN_LEARNING_RATE)

    history: List[List[float]] = []
    train_start = time.time()
    print("Start training ...")
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        batch_data_xy, batch_data_u = random_batch(train_xy, train_u, DATA_BATCH_SIZE)
        batch_bc_xy, batch_bc_u = random_batch(bc_xy, bc_u, BC_BATCH_SIZE)
        batch_coll_xy, batch_coll_src = random_batch(coll_xy, coll_src, COLL_BATCH_SIZE)

        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(
            model,
            batch_data_xy,
            batch_data_u,
            batch_bc_xy,
            batch_bc_u,
            batch_coll_xy,
            batch_coll_src,
            step,
        )
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        if step % VAL_INTERVAL == 0 or step == 1 or step == TRAIN_STEPS:
            model.eval()
            val_loss = compute_validation_loss(model, val_xy, val_u)
        else:
            val_loss = float("nan")

        row = [
            float(step),
            losses["total"].detach().item(),
            losses["pde"].detach().item(),
            losses["data"].detach().item(),
            losses["bc"].detach().item(),
            losses["cont"].detach().item(),
            losses["momx"].detach().item(),
            losses["momy"].detach().item(),
            losses["energy"].detach().item(),
            losses["phi"].detach().item(),
            float(val_loss),
            float(optimizer.param_groups[0]["lr"]),
        ]
        history.append(row)

        if step % LOG_INTERVAL == 0 or step == 1 or step == TRAIN_STEPS:
            val_text = f"{val_loss:.6e}" if math.isfinite(val_loss) else "nan"
            print(
                f"Step {step:>7d}/{TRAIN_STEPS} | "
                f"total={row[1]:.6e} | pde={row[2]:.6e} | data={row[3]:.6e} | bc={row[4]:.6e} | val={val_text}"
            )

    training_time = time.time() - train_start
    torch.save(model.state_dict(), os.path.join(dirs["model"], f"{CASE_NAME}_state_dict.pth"))
    history_array = np.array(history, dtype=np.float64)
    save_all_outputs(model, dirs, history_array, training_time, total_start)
    write_case_description(os.path.join(dirs["log"], "case_description.txt"))
    print(f"Finished {CASE_NAME}. Outputs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
