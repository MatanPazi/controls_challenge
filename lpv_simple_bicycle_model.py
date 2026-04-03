"""
lpv_simple_bicycle_model .py

Fits LPV state-space model:

vy[k+1] = a11(v[k]) * vy[k] + a12(v[k]) * r[k] + b1(v[k]) * delta[k]

r[k+1] = a21(v[k]) * vy[k] + a22(v[k]) * r[k] + b2(v[k]) * delta[k]

State definitions:

vy[k] = lateral velocity
r[k] = yaw rate
delta[k] = steering input
v[k] = longitudinal velocity

a(v), b(v) use configurable LPV basis

Didn't work well.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy.optimize import lsq_linear
from tinyphysics import CONTROL_START_IDX

# ============================
# Config
# ============================

DATA_DIR = Path("data_excitation")
MAX_ROUTES = 1000
MIN_SPEED = 3.0
DT = 0.1

LAMBDA_RIDGE = 1e-3

# 🔴 CHANGE THIS ONLY
BASIS_DIM = 2   # 1 = const, 2 = [1,v], 3 = [1,v,v²]

# ============================
# Basis
# ============================

def lpv_basis(v):
    if BASIS_DIM == 1:
        return np.ones((len(v), 1))
    elif BASIS_DIM == 2:
        return np.stack([np.ones_like(v), v], axis=1)
    elif BASIS_DIM == 3:
        return np.stack([np.ones_like(v), v, v**2], axis=1)
    else:
        raise ValueError("Unsupported BASIS_DIM")

# ============================
# Load data
# ============================

def load_files():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    return files[:MAX_ROUTES]

# ============================
# Build regression
# ============================

def build_regression(files):
    X_blocks = []
    y_blocks = []

    for i, f in enumerate(files):
        df = pd.read_csv(f)

        if not all(col in df.columns for col in ["current_lataccel", "steerCommand", "vEgo"]):
            continue

        ay = df["current_lataccel"].values
        if "roll" in df.columns:
            ay = ay - 9.81 * df["roll"].values

        delta = df["steerCommand"].values
        v = df["vEgo"].values

        mask = (
            (np.arange(len(df)) >= CONTROL_START_IDX) &
            (v > MIN_SPEED) &
            np.isfinite(ay) &
            np.isfinite(delta) &
            np.isfinite(v)
        )

        ay = ay[mask]
        delta = delta[mask]
        v = v[mask]

        if len(ay) < 10:
            continue

        # ----------------------------
        # State estimation
        # ----------------------------

        # yaw rate approximation
        r = ay / np.maximum(v, 1e-3)

        # lateral velocity approximation
        vy = ay / np.maximum(v, 1e-3)

        # next-step signals
        vy_k   = vy[:-1]
        vy_k1  = vy[1:]
        r_k    = r[:-1]
        r_k1   = r[1:]
        delta_k = delta[:-1]
        v_k     = v[:-1]

        phi = lpv_basis(v_k)  # (N, BASIS_DIM)

        # ----------------------------
        # Build regression
        # ----------------------------

        # vy equation
        X_vy = np.hstack([
            phi * vy_k[:, None],
            phi * r_k[:, None],
            phi * delta_k[:, None],
        ])  # (N, 3*BASIS_DIM)

        y_vy = vy_k1

        # r equation
        X_r = np.hstack([
            phi * vy_k[:, None],
            phi * r_k[:, None],
            phi * delta_k[:, None],
        ])

        y_r = r_k1

        # stack both equations
        # number of features per equation
        n_feat = 3 * BASIS_DIM

        # build block-diagonal regression
        N = X_vy.shape[0]

        X_top = np.hstack([X_vy, np.zeros((N, n_feat))])
        X_bottom = np.hstack([np.zeros((N, n_feat)), X_r])

        X = np.vstack([X_top, X_bottom])
        y = np.concatenate([y_vy, y_r])

        X_blocks.append(X)
        y_blocks.append(y)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(files)}] processed")

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)

    return X, y

# ============================
# Fit
# ============================

def fit_model(X, y, lam=1e-3):
    n = X.shape[1]

    sqrt_lam = np.sqrt(lam)
    X_aug = np.vstack([X, sqrt_lam * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])

    res = lsq_linear(X_aug, y_aug, method='trf')

    if not res.success:
        print("WARNING:", res.message)

    return res.x

# ============================
# Pretty print
# ============================

def print_model(theta):
    n = BASIS_DIM

    a11 = theta[0*n:1*n]
    a12 = theta[1*n:2*n]
    b1  = theta[2*n:3*n]

    a21 = theta[3*n:4*n]
    a22 = theta[4*n:5*n]
    b2  = theta[5*n:6*n]

    basis_names = ["1", "v", "v²"][:BASIS_DIM]

    def print_block(name, coeffs):
        print(f"\n{name}(v) = ", end="")
        print(" + ".join([f"{name}{i}*{basis_names[i]}" for i in range(BASIS_DIM)]))
        for i, c in enumerate(coeffs):
            print(f"  {name}{i} = {c:.6f}")

    print("\n=== LPV Bicycle Model ===")

    print_block("a11", a11)
    print_block("a12", a12)
    print_block("b1", b1)

    print_block("a21", a21)
    print_block("a22", a22)
    print_block("b2", b2)

# ============================
# Simulation
# ============================

def simulate_model(df, theta, dt=0.1):
    n = BASIS_DIM

    a11 = theta[0*n:1*n]
    a12 = theta[1*n:2*n]
    b1  = theta[2*n:3*n]

    a21 = theta[3*n:4*n]
    a22 = theta[4*n:5*n]
    b2  = theta[5*n:6*n]

    ay = df["current_lataccel"].values
    delta = df["steerCommand"].values
    v = df["vEgo"].values

    # initial states
    vy = ay[0] / max(v[0], 1e-3)
    r  = ay[0] / max(v[0], 1e-3)

    ay_pred = []

    for k in range(len(ay)):
        v_k = v[k]
        phi = lpv_basis(np.array([v_k]))[0]

        a11_v = np.dot(phi, a11)
        a12_v = np.dot(phi, a12)
        b1_v  = np.dot(phi, b1)

        a21_v = np.dot(phi, a21)
        a22_v = np.dot(phi, a22)
        b2_v  = np.dot(phi, b2)

        vy_next = a11_v * vy + a12_v * r + b1_v * delta[k]
        r_next  = a21_v * vy + a22_v * r + b2_v * delta[k]

        vy, r = vy_next, r_next

        roll_k = df["roll"].values[k] if "roll" in df.columns else 0.0

        # gravity compensation
        ay_pred.append(v_k * r + 1.0 * roll_k)

    return np.array(ay_pred)

# ============================
# Compare
# ============================

def compare_model(theta, file_path):
    df = pd.read_csv(file_path)

    ay_true = df["current_lataccel"].values
    ay_pred = simulate_model(df, theta)

    plt.figure(figsize=(12,6))
    plt.plot(ay_true, label="TinyPhysics")
    plt.plot(ay_pred, '--', label="State-space model")
    plt.legend()
    plt.title("Model vs Ground Truth")
    plt.grid(True)
    plt.show()

# ============================
# Main
# ============================

if __name__ == "__main__":
    files = load_files()
    print(f"Using {len(files)} files")

    X, y = build_regression(files)
    print("Samples:", X.shape[0])

    theta = fit_model(X, y, LAMBDA_RIDGE)

    print_model(theta)

    np.save("lpv_ss_theta.npy", theta)

    compare_model(theta, str(DATA_DIR / "00000_excitation_step.csv"))
