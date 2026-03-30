"""
Improved LPV State-Space Identification
- Better regression
- Constraints (stability + positive gain)
- Cleaner simulation
- Diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import lsq_linear
from tinyphysics import CONTROL_START_IDX

# ============================
# Config
# ============================

DATA_DIR = Path("data_excitation")
MAX_ROUTES = 1000
MIN_SPEED = 3.0
DT = 0.1          # <<<=== CRITICAL: Try 0.1, 0.05, or 0.02 and compare fits
LAMBDA_RIDGE = 1e-3

# ============================
# LPV Basis
# ============================

def lpv_basis(v):
    return np.stack([np.ones_like(v), v, v**2], axis=1)  # (N,3)

# ============================
# Build Regression (cleaner)
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
            (np.arange(len(df)) >= CONTROL_START_IDX + 2) &  # need room for differences
            (v > MIN_SPEED) &
            np.isfinite(ay) & np.isfinite(delta) & np.isfinite(v)
        )

        ay = ay[mask]
        delta = delta[mask]
        v = v[mask]

        if len(ay) < 15:
            continue

        kappa = ay / np.maximum(v**2, 1e-3)

        # Finite difference for kappa_dot (centered where possible)
        kdot = np.zeros_like(kappa)
        kdot[1:-1] = (kappa[2:] - kappa[:-2]) / (2 * DT)   # smoother
        kdot[0] = (kappa[1] - kappa[0]) / DT
        kdot[-1] = (kappa[-1] - kappa[-2]) / DT

        # Regression: kdot[k+1] = a(v[k]) * kdot[k] + b(v[k]) * delta[k]
        X = np.hstack([
            (lpv_basis(v[:-1]) * kdot[:-1][:, None]),   # a(v) * kdot
            (lpv_basis(v[:-1]) * delta[:-1][:, None])   # b(v) * delta
        ])

        y = kdot[1:]

        X_blocks.append(X)
        y_blocks.append(y)

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    return X, y


# ============================
# Constrained Fit
# ============================

def fit_model(X, y, lam=1e-3):
    n = X.shape[1]  # 6

    lb = np.array([-np.inf]*3 + [0.0]*3)      # a free, b >= 0
    ub = np.full(n, np.inf)

    sqrt_lam = np.sqrt(lam)
    X_aug = np.vstack([X, sqrt_lam * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])

    res = lsq_linear(X_aug, y_aug, bounds=(lb, ub), method='trf', verbose=0)
    return res.x


# ============================
# Simulation (fixed order)
# ============================

def simulate_model(df, theta, dt=DT):
    a_coeffs = theta[:3]
    b_coeffs = theta[3:]

    ay = df["current_lataccel"].values
    delta = df["steerCommand"].values
    v = df["vEgo"].values

    kappa = np.zeros(len(ay))
    kdot = np.zeros(len(ay))

    # Warm-up from real data
    kappa[0] = ay[0] / max(v[0]**2, 1e-3)
    kdot[0] = 0.0

    for k in range(1, len(ay)):
        phi = np.array([1.0, v[k-1], v[k-1]**2])
        a_v = np.dot(phi, a_coeffs)
        b_v = np.dot(phi, b_coeffs)

        kdot[k] = a_v * kdot[k-1] + b_v * delta[k-1]
        kappa[k] = kappa[k-1] + dt * kdot[k-1]   # ← use OLD kdot (standard forward Euler)

    ay_pred = v**2 * kappa
    return ay_pred


# ============================
# Plot + Print
# ============================

def print_model(theta):
    a = theta[:3]
    b = theta[3:]
    print("\n=== LPV State-Space Model ===")
    print("a(v) =", f"{a[0]:.5f} + {a[1]:.5f}*v + {a[2]:.5f}*v²")
    print("b(v) =", f"{b[0]:.5f} + {b[1]:.5f}*v + {b[2]:.5f}*v²")
    print(f"  Pole at v=20m/s: {a[0] + a[1]*20 + a[2]*400:.3f}  (should be <1)")

def compare_model(theta, file_path):
    df = pd.read_csv(file_path)
    ay_true = df["current_lataccel"].values
    ay_pred = simulate_model(df, theta)

    plt.figure(figsize=(13, 6))
    plt.plot(ay_true, label="TinyPhysics Ground Truth", lw=1.8)
    plt.plot(ay_pred, '--', label="LPV State-Space Model", lw=1.6, alpha=0.9)
    plt.legend()
    plt.title(f"Model Fit - {Path(file_path).name}")
    plt.xlabel("Step")
    plt.ylabel("Lateral Accel [m/s²]")
    plt.grid(True, alpha=0.3)
    plt.show()

    rmse = np.sqrt(np.mean((ay_true - ay_pred)**2))
    print(f"RMSE on this file: {rmse:.4f} m/s²")


# ============================
# Main
# ============================

if __name__ == "__main__":
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))[:MAX_ROUTES]
    print(f"Using {len(files)} files")

    X, y = build_regression(files)
    print(f"Total regression samples: {len(y):,}")

    theta = fit_model(X, y, LAMBDA_RIDGE)
    print_model(theta)

    np.save("lpv_ss_theta.npy", theta)

    # Test on a few files
    test_files = ["00000_excitation_step_pos.csv", "00019_excitation_step.csv"]
    for tf in test_files:
        full_path = DATA_DIR / tf
        if full_path.exists():
            compare_model(theta, str(full_path))