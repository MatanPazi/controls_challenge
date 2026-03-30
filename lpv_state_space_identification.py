"""
lpv_state_space_identification.py

Fits LPV state-space model:

kappa_{k+1} = kappa_k + dt * kappa_dot_k
kappa_dot_{k+1} = a(v)*kappa_dot_k + b(v)*delta_k

a(v), b(v) are quadratic in v: [1, v, v^2]
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
DT = 0.1   # <-- adjust to your simulator timestep

LAMBDA_RIDGE = 1e-3

# ============================
# Basis
# ============================

def lpv_basis(v):
    return np.stack([np.ones_like(v), v, v**2], axis=1)  # (N,3)

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

    for f in files:
        df = pd.read_csv(f)

        if not all(col in df.columns for col in ["current_lataccel", "steerCommand", "vEgo"]):
            continue

        ay = df["current_lataccel"].values
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

        # Compute curvature
        kappa = ay / np.maximum(v**2, 1e-3)

        # Compute kappa_dot (finite difference)
        kappa_dot = np.zeros_like(kappa)
        kappa_dot[1:] = (kappa[1:] - kappa[:-1]) / DT

        # Shift for regression
        kdot_k   = kappa_dot[:-1]
        kdot_k1  = kappa_dot[1:]
        delta_k  = delta[:-1]
        v_k      = v[:-1]

        phi = lpv_basis(v_k)  # (N,3)

        # Build regression:
        # kdot[k+1] = a(v)*kdot[k] + b(v)*delta[k]

        X = np.hstack([
            phi * kdot_k[:, None],   # a(v)*kdot
            phi * delta_k[:, None],  # b(v)*delta
        ])  # shape (N, 6)

        y = kdot_k1

        X_blocks.append(X)
        y_blocks.append(y)

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
    a_coeffs = theta[:3]
    b_coeffs = theta[3:]

    print("\n=== LPV State-Space Model ===")

    print("\na(v) = a0 + a1*v + a2*v^2")
    print(f"  a0 = {a_coeffs[0]:.6f}")
    print(f"  a1 = {a_coeffs[1]:.6f}")
    print(f"  a2 = {a_coeffs[2]:.6f}")

    print("\nb(v) = b0 + b1*v + b2*v^2")
    print(f"  b0 = {b_coeffs[0]:.6f}")
    print(f"  b1 = {b_coeffs[1]:.6f}")
    print(f"  b2 = {b_coeffs[2]:.6f}")


def simulate_model(df, theta, dt=0.1):
    a_coeffs = theta[:3]
    b_coeffs = theta[3:]

    ay = df["current_lataccel"].values
    delta = df["steerCommand"].values
    v = df["vEgo"].values

    # init from real data
    kappa = ay[0] / max(v[0]**2, 1e-3)
    kappa_dot = 0.0

    ay_pred = []

    for k in range(len(ay)):
        v_k = v[k]

        phi = np.array([1.0, v_k, v_k**2])
        a_v = np.dot(phi, a_coeffs)
        b_v = np.dot(phi, b_coeffs)

        # state update
        kappa_dot = a_v * kappa_dot + b_v * delta[k]
        kappa = kappa + dt * kappa_dot

        ay_pred.append(v_k**2 * kappa)

    return np.array(ay_pred)    

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

    compare_model(theta, str(DATA_DIR / "00000_excitation_step_pos.csv"))
