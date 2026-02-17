"""
LPV-ARX Model Identification for TinyPhysics Lateral Dynamics

This script fits a Linear Parameter-Varying AutoRegressive with eXogenous inputs (LPV-ARX) model 
to predict the next lateral acceleration (targetLateralAcceleration) from:
- Past lateral accelerations (NA lags)
- Past steering commands (NDELTA lags)
- Current exogenous signals: vEgo, aEgo, roll

The model uses a quadratic basis in vEgo (constant + linear + v²) to make coefficients speed-dependent, 
capturing the strong v² dependence in vehicle lateral dynamics.

Key steps:
1. Loads CSV files from data folder (limited to MAX_ROUTES)
2. For each file: filters pre-control-start data (before CONTROL_START_IDX, for original openpilot controller), 
   removes rows with NaN/inf in key columns
3. Builds regression matrix X by multiplying lagged values with LPV basis in vEgo (phi matrix)
4. Fits parameters theta using ridge regression (regularized least squares)
5. Saves theta to lpv_arx_theta.npy
6. Reports one-step RMSE on the full dataset
7. Plots a comparison between targetLateralAcceleration and simulated acceleration for a chosen route.

Usage: Run the script -> get theta -> use in simulation/control design.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
import time
from tinyphysics import CONTROL_START_IDX


# ============================
# Configuration
# ============================

DATA_DIR = Path("data")
MAX_ROUTES = 1000
LAMBDA_RIDGE = 1e-4     # Small penalty to prevent overfitting (higher = simpler model).

NA = 2                  # Use 2 past ay values
NDELTA = 3              # Use 3 past steering commands (Observed sample delay)
BASIS_DIM = 3           # number of basis functions per regressor (const + v + v²)
NUM_EXO_VAR = 3         # number of exogenous inputs (vEgo, aEgo, roll)

FEATURE_DIM = BASIS_DIM * (NA + NDELTA + NUM_EXO_VAR)   # Total columns in the feature matrix.
                                                        # Each regressor (past ay, past steer, exogenous) gets 3 basis terms (1, v, v²)
                                                        # So 3 * (2 + 3 + 3) = 24 total parameters.

MIN_SPEED = 3           # [m/s] — Excluding low speeds to avoid highly non-linear behavior

# ============================
# LPV basis
# ============================

def lpv(v):
    """
    Compute the LPV basis functions for speed vEgo.
    
    Args:
        v (np.ndarray): Speed values (shape: (N,)).
    
    Returns:
        np.ndarray: Basis matrix (N, 3) with columns [1, v, v²] for quadratic dependence.
    
    Logic: Each regressor (past ay, steer, exogenous) is multiplied by this basis
    to create speed-varying coefficients. Quadratic term captures v² physics in turning.
    """
    return np.stack([np.ones_like(v), v, v**2], axis=1)  # (N,3)


# ============================
# Data Loading
# ============================

def load_routes():
    """
    Load a list of CSV file paths from the data directory.
    
    Returns:
        list: Sorted list of file paths, limited to MAX_ROUTES.
    
    Logic: Prepares the file list for batch processing without loading data yet.
    """    
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    files = files[:MAX_ROUTES]
    return files


# ============================
# Fast Regressor Construction
# ============================

def build_regression(files):
    """
    Build the feature matrix X and target y from multiple CSV routes.
    
    Args:
        files (list): List of CSV file paths.
    
    Returns:
        tuple: (X: np.ndarray (total_samples, FEATURE_DIM), y: np.ndarray (total_samples,))
    
    Logic: For each file, extract signals, filter finite values, and create shifted regressors
    (past ay/steer * LPV basis + current exogenous * LPV basis). Stack all files into one big
    dataset for batch fitting. Vectorized for speed; skips short/invalid files.
    """    
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
        valid_mask = (
            (np.arange(len(df)) < CONTROL_START_IDX) &
            (v >= MIN_SPEED) &
            np.isfinite(ay) &
            np.isfinite(steer) &
            np.isfinite(v) &
            np.isfinite(a) &
            np.isfinite(roll)
        )

        if not np.any(valid_mask):
            print(f"No finite rows in {Path(f).name}")
            continue

        # Apply mask
        ay    = ay[valid_mask]
        steer = steer[valid_mask]
        v     = v[valid_mask]
        a     = a[valid_mask]
        roll  = roll[valid_mask]

        N = len(ay)
        k0 = max(NA, NDELTA)        # Minimum number of time steps we need to skip at the beginning of each route so that we have enough past history for every prediction.
        Ns = N - k0                 # Number of usable prediction samples we can extract

        if Ns <= 0:
            continue

        phi = np.zeros((Ns, FEATURE_DIM))   # The feature matrix.
                                            # Each row = one usable time step
                                            # Each column = one "feature" (a lagged value multiplied by one part of the LPV basis)
                                            # Partial one row example: ay[k-1] × [1, v[k], v[k]²], ay[k-2] × [1, v[k], v[k]²], delta[k-1] × [1, v[k], v[k]²], ...

        v_lpv = lpv(v[k0:])                 # Matrix that holds the LPV basis values for each time step. In our case: (1, v, v**2). (Ns,3)

        col = 0

        # ---- AR terms ----
        for i_lag in range(1, NA + 1):
            ay_lag = ay[k0 - i_lag : N - i_lag]
            phi[:, col:col+BASIS_DIM] = v_lpv * ay_lag[:, None]
            col += BASIS_DIM

        # ---- Steering terms ----
        for d_lag in range(1, NDELTA + 1):
            steer_lag = steer[k0 - d_lag : N - d_lag]
            phi[:, col:col+BASIS_DIM] = v_lpv * steer_lag[:, None]
            col += BASIS_DIM

        # ---- Exogenous inputs ----
        for z in (v[k0:], a[k0:], roll[k0:]):
            phi[:, col:col+BASIS_DIM] = v_lpv * z[:, None]
            col += BASIS_DIM

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
    """
    Fit LPV-ARX parameters using ridge regression (regularized least squares).
    
    Args:
        X (np.ndarray): Feature matrix (samples, features).
        y (np.ndarray): Target vector (samples,).
        lam (float): Ridge penalty strength.
    
    Returns:
        np.ndarray: Learned parameters theta (features,).
    
    Logic: Solves (X^T X + λ I) θ = X^T y for stable, shrunk coefficients.
    Prevents overfitting by penalizing large θ values.
    """    
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
    valid_mask = (
        (np.arange(len(df)) < CONTROL_START_IDX) &
        (v >= MIN_SPEED) &
        np.isfinite(ay) &
        np.isfinite(steer) &
        np.isfinite(v) &
        np.isfinite(a) &
        np.isfinite(roll)
    )      

    ay    = ay[valid_mask]
    steer = steer[valid_mask]
    v     = v[valid_mask]
    a     = a[valid_mask]
    roll  = roll[valid_mask]

    if len(ay) < max(na, ndelta) + 50:
        print(f"File too short after filtering: {len(ay)} steps")
        return

    # Simulate
    y_sim = np.zeros(len(ay))
    y_sim[:max(na, ndelta)] = ay[:max(na, ndelta)]  # warm-up with real values    

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
    plt.plot(y_sim[1:], label='Simulated (shifted forward 1)', linestyle='--', color='red')
    plt.plot(ay[:-1], label='Actual ay (shifted back 1)', alpha=0.7)
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
