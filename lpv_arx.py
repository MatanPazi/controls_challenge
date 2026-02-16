import numpy as np
import pandas as pd
import glob
from pathlib import Path
import time

# ============================
# Configuration
# ============================

DATA_DIR = Path("data")
MAX_ROUTES = 1000        # <<< key speed knob
LAMBDA_RIDGE = 1e-4

NA = 2
NDELTA = 3
BASIS_DIM = 3  # number of basis functions per regressor (const + v + v²)

CONTROL_START_IDX = 100
MIN_SPEED = 3

# ============================
# LPV basis
# ============================

def lpv(v):
    return np.stack([np.ones_like(v), v, v**2], axis=1)  # (N,3)


FEATURE_DIM = 3 * (NA + NDELTA + 3)

# ============================
# Data Loading
# ============================

def load_routes():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    files = files[:MAX_ROUTES]
    return files


# ============================
# Fast Regressor Construction
# ============================

def build_regression(files):
    X_blocks = []
    y_blocks = []

    t0 = time.time()
    total_samples = 0

    for i, f in enumerate(files):
        df = pd.read_csv(f)

        # Required columns
        ay_col    = "targetLateralAcceleration"
        steer_col = "steerCommand"
        v_col     = "vEgo"
        a_col     = "aEgo"
        roll_col  = "roll"

        if not all(col in df.columns for col in [ay_col, steer_col, v_col, a_col, roll_col]):
            print(f"Skipping {Path(f).name} — missing columns")
            continue

        # Extract and handle NaNs early
        ay    = df[ay_col].values
        steer = df[steer_col].values
        v     = df[v_col].values
        a     = df[a_col].values
        roll  = df[roll_col].values

        # Create a mask for rows where ALL relevant columns are finite
        finite_mask = (
            np.isfinite(ay) &
            np.isfinite(steer) &
            np.isfinite(v) &
            np.isfinite(a) &
            np.isfinite(roll)
        )

        if not np.any(finite_mask):
            print(f"No finite rows in {Path(f).name}")
            continue

        # Apply mask
        ay    = ay[finite_mask]
        steer = steer[finite_mask]
        v     = v[finite_mask]
        a     = a[finite_mask]
        roll  = roll[finite_mask]

        N = len(ay)
        k0 = max(NA, NDELTA)
        Ns = N - k0

        if Ns <= 0:
            continue

        phi = np.zeros((Ns, FEATURE_DIM))

        v_lpv = lpv(v[k0:])   # (Ns,3)

        col = 0

        # ---- AR terms ----
        for i_lag in range(1, NA + 1):
            ay_lag = ay[k0 - i_lag : N - i_lag]
            phi[:, col:col+3] = v_lpv * ay_lag[:, None]
            col += 3

        # ---- Steering terms ----
        for d_lag in range(1, NDELTA + 1):
            steer_lag = steer[k0 - d_lag : N - d_lag]
            phi[:, col:col+3] = v_lpv * steer_lag[:, None]
            col += 3

        # ---- Exogenous inputs ----
        for z in (v[k0:], a[k0:], roll[k0:]):
            phi[:, col:col+3] = v_lpv * z[:, None]
            col += 3

        X_blocks.append(phi)
        y_blocks.append(ay[k0:])

        total_samples += Ns

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"[{i+1}/{len(files)} routes] "
                f"samples: {total_samples:,} "
                f"elapsed: {elapsed:.1f}s"
            )

    if not X_blocks:
        raise ValueError("No valid data after NaN filtering")

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    return X, y


# ============================
# Ridge Regression
# ============================

def ridge_regression(X, y, lam):
    n = X.shape[1]
    return np.linalg.solve(
        X.T @ X + lam * np.eye(n),
        X.T @ y
    )


# ============================
# Sim vs Meas plot
# ============================
def plot_simulation_on_file(theta, file_path, na=NA, ndelta=NDELTA):
    """Plot actual vs simulated ay for one specific file"""
    df = pd.read_csv(file_path)
    
    ay    = df["targetLateralAcceleration"].values
    steer = np.nan_to_num(df["steerCommand"].values, nan=0.0)
    v     = df["vEgo"].values
    a     = df["aEgo"].values
    roll  = df["roll"].values

    # Use the same filtering as in training
    valid = (
        (np.arange(len(df)) >= CONTROL_START_IDX) &
        (v >= MIN_SPEED) &
        np.isfinite(ay) &
        np.isfinite(v)
    )

    ay    = ay[valid]
    steer = steer[valid]
    v     = v[valid]
    a     = a[valid]
    roll  = roll[valid]

    if len(ay) < max(na, ndelta) + 50:
        print(f"File too short after filtering: {len(ay)} steps")
        return

    # Simulate
    y_sim = np.zeros(len(ay))
    y_sim[:na] = ay[:na]  # warm-up with real values

    for k in range(max(na, ndelta), len(ay)):
        pred = 0.0
        col = 0

        v_now = v[k]
        v_lpv = lpv(np.array([v_now]))[0]  # (3,)

        # past ay
        for i in range(1, na + 1):
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * ay[k - i]
            col += BASIS_DIM

        # past steer
        for d in range(1, ndelta + 1):
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * steer[k - d]
            col += BASIS_DIM

        # exogenous
        for z in [v[k], a[k], roll[k]]:
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * z
            col += BASIS_DIM

        y_sim[k] = pred

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(ay, label='Actual ay', linewidth=1.4, alpha=0.9)
    plt.plot(y_sim, label='Simulated ay', linewidth=2.0, linestyle='--', color='red')
    plt.title(f"Actual vs Simulated Lateral Acceleration\n{Path(file_path).name}")
    plt.xlabel("Time step")
    plt.ylabel("Lateral acceleration [m/s²]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================
# Main
# ============================

if __name__ == "__main__":
    files = load_routes()
    print(f"Using {len(files)} routes")

    print("Building regression matrices...")
    X, y = build_regression(files)

    print(f"Total samples: {X.shape[0]:,}")
    print("Estimating parameters...")

    theta = ridge_regression(X, y, LAMBDA_RIDGE)

    print("\n=== Learned Parameters (theta) ===")
    print(f"Total number of parameters: {len(theta)}")

    # Group them nicely
    basis_names = ["const", "v", "v²"]

    print("\nCoefficients grouped by regressor type:")
    col = 0

    # AR terms (past ay)
    print("Past ay lags:")
    for lag in range(1, NA + 1):
        coeffs = theta[col:col + BASIS_DIM]
        print(f"  ay_{lag}:   {basis_names[0]:<6} {coeffs[0]:12.6f}   "
            f"{basis_names[1]:<6} {coeffs[1]:12.6f}   "
            f"{basis_names[2]:<6} {coeffs[2]:12.6f}")
        col += BASIS_DIM

    # Steering terms
    print("\nPast steer lags:")
    for lag in range(1, NDELTA + 1):
        coeffs = theta[col:col + BASIS_DIM]
        print(f"  delta_{lag}: {basis_names[0]:<6} {coeffs[0]:12.6f}   "
            f"{basis_names[1]:<6} {coeffs[1]:12.6f}   "
            f"{basis_names[2]:<6} {coeffs[2]:12.6f}")
        col += BASIS_DIM

    # Exogenous (v, a, roll)
    print("\nExogenous inputs:")
    for name in ["current vEgo", "aEgo", "roll"]:
        coeffs = theta[col:col + BASIS_DIM]
        print(f"  {name:12}: {basis_names[0]:<6} {coeffs[0]:12.6f}   "
            f"{basis_names[1]:<6} {coeffs[1]:12.6f}   "
            f"{basis_names[2]:<6} {coeffs[2]:12.6f}")
        col += BASIS_DIM

    print("\nConstant term (intercept):", theta[0])    

    np.save("lpv_arx_theta.npy", theta)

    rmse = np.sqrt(np.mean((X @ theta - y) ** 2))
    print(f"One-step RMSE: {rmse:.4f} m/s²")


    # Pick one or more files you want to visualize
    example_file = "data/00100.csv"   # change to any valid route
    plot_simulation_on_file(theta, example_file)