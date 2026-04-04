"""
LPV-ARX Model Identification for TinyPhysics Lateral Dynamics

This script fits a Linear Parameter-Varying AutoRegressive with eXogenous inputs (LPV-ARX) model 
to predict the next lateral acceleration (current_lataccel) from:
- Past lateral accelerations (NA lags)
- Current steer command (NUM_STEER_TERMS)
- A choice of current exogenous signals: vEgo, aEgo, roll

The model allows the use of a quadratic basis in vEgo (constant + linear + v²) to make coefficients speed-dependent, 
capturing the potentially strong v² dependence in vehicle lateral dynamics.

Key steps:
1. Loads CSV files from data folder (limited to MAX_ROUTES)
2. For each file: filters pre-control-start data (before CONTROL_START_IDX, for original openpilot controller), 
   removes rows with NaN/inf in key columns
3. Builds regression matrix X by multiplying lagged values with LPV basis in vEgo (phi matrix)
4. Fits parameters theta using ridge regression (regularized least squares)
5. Saves theta to lpv_arx_theta.npy
6. Reports one-step RMSE on the full dataset
7. Plots a comparison between current_lataccel and simulated acceleration for a chosen route.

Usage: Run the script -> get theta -> use in simulation/control design.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import time
from tinyphysics import CONTROL_START_IDX
from scipy.optimize import lsq_linear



# ============================
# Configuration
# ============================

DATA_DIR = Path("data_excitation")
MAX_ROUTES = 1000
LAMBDA_RIDGE = 1e-4         # Small penalty to prevent overfitting (higher = simpler model).

NA = 2                      # Use 1 past ay values
NUM_STEER_TERMS = 2         # Only current steer (Assumes lag = 0)
BASIS_DIM = 1               # Number of basis functions per regressor (const + v + v²). BASIS_DIM = 1 disregards v and v².

# === NEW: Dynamic exogenous variables ===
EXO_VARS = ['roll']             # Change as needed, examples:
                                        # ["vEgo"] 
                                        # ["vEgo", "roll"]
                                        # ["vEgo", "aEgo"]
                                        # ["vEgo", "roll", "aEgo"]

FEATURE_DIM = BASIS_DIM * (NA + NUM_STEER_TERMS + len(EXO_VARS))    # Total columns in the feature matrix.
                                                                    # Each regressor (past ay, steer, exogenous) gets BASIS_DIM basis terms (1, v, v²)
                                                                    # So 1 * (1 + 1 + 2) = 4 total parameters.

MIN_SPEED = 1.0

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
    if BASIS_DIM == 1:
        return np.ones((len(v), 1))
    elif BASIS_DIM == 2:
        return np.stack([np.ones_like(v), v], axis=1)
    elif BASIS_DIM == 3:
        return np.stack([np.ones_like(v), v, v**2], axis=1)
    else:
        raise ValueError(f"Unsupported BASIS_DIM: {BASIS_DIM}")


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
        ay_col    = "current_lataccel"
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
            (np.arange(len(df)) > CONTROL_START_IDX) &
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
        k0 = max(NA, NUM_STEER_TERMS)   # Minimum number of time steps we need to skip at the beginning of each route so that we have enough past history for every prediction.
        Ns = N - k0                     # Number of usable prediction samples we can extract
        if Ns <= 0:
            continue

        phi = np.zeros((Ns, FEATURE_DIM))   # The feature matrix.
                                            # Each row = one usable time step
                                            # Each column = one "feature" (a lagged value multiplied by one part of the LPV basis)
                                            # Partial one row example: ay[k-1] × [1, v[k], v[k]²], ay[k-2] × [1, v[k], v[k]²], delta[k-1] × [1, v[k], v[k]²], ...
        
        v_lpv = lpv(v[k0:])                 # Matrix that holds the LPV basis values for each time step. For example, with BASIS_DIM = 3: (1, v, v**2). (Ns,3)

        col = 0

        # ---- AR terms (past ay) ----
        for i_lag in range(1, NA + 1):
            ay_lag = ay[k0 - i_lag : N - i_lag]
            phi[:, col:col+BASIS_DIM] = v_lpv * ay_lag[:, None]
            col += BASIS_DIM

        # ---- Steering terms (current + past) ----
        for d_lag in range(NUM_STEER_TERMS):
            steer_lag = steer[k0 - d_lag : N - d_lag]
            phi[:, col:col+BASIS_DIM] = v_lpv * steer_lag[:, None]
            col += BASIS_DIM

        # ---- Dynamic exogenous inputs ----
        for exo_col in EXO_VARS:
            z = df[exo_col].values[valid_mask][k0:]
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

def constrained_ridge_regression(X, y, lam=1e-4):
    """
    Fit with ridge + non-negativity on steering coefficients only.    
    """
    n = X.shape[1]
    
    # Calculate indices dynamically
    ar_end      = NA * BASIS_DIM                    # end of AR (past ay) part
    steer_start = ar_end
    steer_end   = steer_start + NUM_STEER_TERMS * BASIS_DIM   # end of steering part
    
    lb = np.full(n, -np.inf)
    ub = np.full(n,  np.inf)
    
    # Force steering coefficients >= 0 (physically meaningful gain). Avoiding this constraint for now.
    # if NUM_STEER_TERMS > 0:
    #     lb[steer_start:steer_end] = 0.0

    # Augmented system for ridge regularization
    sqrt_lam = np.sqrt(lam)
    X_aug = np.vstack([X, sqrt_lam * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    
    res = lsq_linear(
        X_aug, y_aug,
        bounds=(lb, ub),
        method='trf',
        verbose=1
    )
    
    if not res.success:
        print("Warning:", res.message)
    
    return res.x

# ============================
# Sim vs Meas plot
# ============================
def plot_simulation_on_file(theta, file_path, na=NA):
    """Plot actual vs simulated ay for one specific file"""    
    
    df = pd.read_csv(file_path)
    
    ay    = df["current_lataccel"].values
    steer = np.nan_to_num(df["steerCommand"].values, nan=0.0)
    v     = df["vEgo"].values
    a     = df["aEgo"].values
    roll  = df["roll"].values

    # Use the same filtering as in training
    valid_mask = (
        (np.arange(len(df)) > CONTROL_START_IDX) &
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

    k0 = max(na, NUM_STEER_TERMS - 1)
    if len(ay) < k0 + 50:    
        print(f"File too short after filtering: {len(ay)} steps")
        return

    # Simulate
    y_sim = np.zeros(len(ay))
    y_sim[:k0] = ay[:k0]

    for k in range(k0, len(ay)):
        pred = 0.0
        col = 0
        v_lpv = lpv(np.array([v[k]]))[0]

        # Past ay
        for i in range(1, na + 1):
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * ay[k - i]
            col += BASIS_DIM

        # Steering terms (current + past)
        for d in range(NUM_STEER_TERMS):
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * steer[max(0, k - d)]
            col += BASIS_DIM

        # Exogenous
        for exo_name in EXO_VARS:
            if exo_name == "vEgo":
                z_val = v[k]
            elif exo_name == "aEgo":
                z_val = a[k]
            elif exo_name == "roll":
                z_val = roll[k]
            else:
                z_val = 0.0
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * z_val
            col += BASIS_DIM

        y_sim[k] = pred

    # Plot
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

    theta = constrained_ridge_regression(X, y, LAMBDA_RIDGE)

    print("\n=== Learned Parameters (theta) ===")
    print(f"NA={NA} | Steer terms={NUM_STEER_TERMS} | Exo={EXO_VARS} | BASIS_DIM={BASIS_DIM}")
    print(f"Total parameters: {len(theta)}")

    if BASIS_DIM == 1:
        basis_names = ["1"]
    elif BASIS_DIM == 2:
        basis_names = ["1", "v"]
    elif BASIS_DIM == 3:
        basis_names = ["1", "v", "v^2"]
    else:
        raise ValueError("Unsupported BASIS_DIM")
    col = 0

    print("\nPast ay lags:")
    for lag in range(1, NA + 1):
        coeffs = theta[col:col + BASIS_DIM]
        terms = " + ".join(
            [f"{coeffs[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)]
        )
        print(f"  ay_{lag}: {terms}")
        col += BASIS_DIM

    print("\nSteering terms:")
    for d in range(NUM_STEER_TERMS):
        coeffs = theta[col:col + BASIS_DIM]
        terms = " + ".join(
            [f"{coeffs[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)]
        )
        print(f"  steer[k-{d}]: {terms}")
        col += BASIS_DIM
        

    print("\nExogenous:")
    for name in EXO_VARS:
        coeffs = theta[col:col + BASIS_DIM]
        terms = " + ".join(
            [f"{coeffs[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)]
        )
        print(f"  {name:12}: {terms}")
        col += BASIS_DIM

    print("\nConstant term (intercept):", theta[0])    

    np.savez(
        "lpv_arx_model.npz",
        theta=theta,
        NA=NA,
        NUM_STEER_TERMS=NUM_STEER_TERMS,
        BASIS_DIM=BASIS_DIM,
        EXO_VARS=np.array(EXO_VARS)
    )

    rmse = np.sqrt(np.mean((X @ theta - y) ** 2))
    print(f"One-step RMSE: {rmse:.4f} m/s²")


    # Pick one or more files you want to visualize
    example_file = "data_excitation/00010_excitation_sine.csv"   # change to any valid route
    plot_simulation_on_file(theta, example_file, NA)
