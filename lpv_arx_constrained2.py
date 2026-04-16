"""
TODO:
Add:
interaction terms: u * v, u * ay, v * ay, ...
Nonlinear AR terms: ay[k-1]^2
Nonlinear steer features: u, u^3, |u|
Steer[k-1] as a separate regressor with its own basis?
...


Nonlinear LPV-ARX Model Identification for TinyPhysics Lateral Dynamics

This script identifies a physics-informed Linear Parameter-Varying (LPV) model 
to predict next lateral acceleration (ay). Key improvements include:

1. SPEED-DEPENDENT BIAS: 
   Identifies a baseline offset (Bias = θ₀ + θ₁v + θ₂/v) to capture road crown, 
   aerodynamic effects, and sensor offsets independently of control inputs.

2. NONLINEAR STEERING (Tire Saturation):
   Uses a Linear + Cubic steering structure (u and u³) to capture high-slip 
   dynamics where steering effectiveness diminishes at large angles.

3. PHYSICS-INFORMED BASIS [1, v, 1/v]:
   Replaces the quadratic basis with an inverse-speed basis. This better 
   reflects vehicle "Understeer Gradient" physics and improves numerical 
   conditioning at high speeds.

4. DYNAMIC/STATIC EXOGENOUS SPLIT:
   - aEgo: Speed-dependent (captures longitudinal-lateral coupling).
   - Roll: Speed-independent (captures pure gravitational lateral pull).

Key steps:
1. Loads CSV files from data folder (limited to MAX_ROUTES).
2. Filters data for validity (vEgo > MIN_SPEED, CONTROL_START_IDX, finite values).
3. Builds a structured regression matrix X using the [1, v, 1/v] basis for 
   dynamic terms and a constant basis for gravitational roll.
4. Fits parameters theta using Ridge Regression (regularized L2) to prevent 
   overfitting to sensor noise.
5. Saves model and metadata to 'lpv_arx_model.npz'.
6. Validates via One-step RMSE and full-horizon simulation plotting.

Usage: Run script -> extract theta -> update mpc.py predict_step with new structure.
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
MAX_ROUTES = 500
LAMBDA_RIDGE = 0.01         # Small penalty to prevent overfitting (higher = simpler model).

NA = 1                      # Use 1 past ay values
NUM_STEER_TERMS = 1         # Only current steer (Assumes lag = 0)
BASIS_DIM = 3               # Number of basis functions per regressor (const + v + v²). BASIS_DIM = 1 disregards v and v².

FEATURE_DIM = (1 + NA + (NUM_STEER_TERMS * 2) + 1) * BASIS_DIM + 1  # (1 bias + NA lags + 2*Steer terms + 1 aEgo) * BASIS_DIM + 1 roll

MIN_SPEED = 1.0

# ============================
# LPV basis
# ============================

def lpv(v):
    v_safe = np.clip(v, MIN_SPEED, None)
    if BASIS_DIM == 1:
        return np.ones((len(v), 1))
    elif BASIS_DIM == 2:
        return np.stack([np.ones_like(v), v], axis=1)
    elif BASIS_DIM == 3:
        # Physics-informed basis: [Constant, Linear Speed, Inverse Speed]
        return np.stack([np.ones_like(v), v, 1.0/v_safe], axis=1)
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

        # ---- 1. Speed-Dependent Bias (The Intercept) ----
        phi[:, col:col+BASIS_DIM] = v_lpv
        col += BASIS_DIM        

        # ---- AR terms (past ay) ----
        for i_lag in range(1, NA + 1):
            ay_lag = ay[k0 - i_lag : N - i_lag]
            phi[:, col:col+BASIS_DIM] = v_lpv * ay_lag[:, None]
            col += BASIS_DIM

        # ---- Steering terms (current + past) ----
        for d_lag in range(NUM_STEER_TERMS):
            steer_lag = steer[k0 - d_lag : N - d_lag]
            # Linear effect
            phi[:, col:col+BASIS_DIM] = v_lpv * steer_lag[:, None]
            col += BASIS_DIM
            # Cubic effect (captures tire saturation/diminishing returns)
            phi[:, col:col+BASIS_DIM] = v_lpv * (steer_lag**2)[:, None]
            col += BASIS_DIM            

        # 4. EXOGENOUS: aEgo (Speed dependent)
        a_curr = a[k0:]
        phi[:, col:col+BASIS_DIM] = v_lpv * a_curr[:, None]
        col += BASIS_DIM

        # 5. EXOGENOUS: Roll (Pure Linear / Speed Independent)
        # Gravity doesn't care how fast you are going.
        phi[:, col] = roll[k0:]
        col += 1

        X_blocks.append(phi)
        y_blocks.append(ay[k0:])

        total_samples += Ns

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
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

        # ---- 1. Speed-Dependent Bias ----
        pred += np.dot(v_lpv, theta[col:col+BASIS_DIM])
        col += BASIS_DIM

        # ---- 2. Past ay ----
        for i in range(1, na + 1):
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * ay[k - i]
            col += BASIS_DIM

        # ---- 3. Steering (Linear + Cubic) ----
        for d in range(NUM_STEER_TERMS):
            u = steer[max(0, k - d)]
            # Linear
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * u
            col += BASIS_DIM
            # Cubic
            pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * (u**3)
            col += BASIS_DIM

        # ---- 4. Exogenous: aEgo ----
        pred += np.dot(v_lpv, theta[col:col+BASIS_DIM]) * a[k]
        col += BASIS_DIM

        # ---- 5. Exogenous: Roll ----
        pred += theta[col] * roll[k]
        col += 1

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
    print(f"NA={NA} | Steer terms={NUM_STEER_TERMS} | BASIS_DIM={BASIS_DIM}")
    print(f"Total parameters: {len(theta)}")

    # Update basis names to match [1, v, 1/v]
    if BASIS_DIM == 3:
        basis_names = ["1", "v", "1/v"]
    elif BASIS_DIM == 2:
        basis_names = ["1", "v"]
    else:
        basis_names = ["1"]

    col = 0

    # 1. Bias
    coeffs = theta[col:col + BASIS_DIM]
    terms = " + ".join([f"{coeffs[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)])
    print(f"\nSpeed-Dep Bias:  {terms}")
    col += BASIS_DIM

    # 2. Past ay lags
    print("\nPast ay lags:")
    for lag in range(1, NA + 1):
        coeffs = theta[col:col + BASIS_DIM]
        terms = " + ".join([f"{coeffs[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)])
        print(f"  ay_{lag}: {terms}")
        col += BASIS_DIM

    # 3. Steering terms
    print("\nSteering terms (Linear + Cubic):")
    for d in range(NUM_STEER_TERMS):
        # Linear
        coeffs_l = theta[col:col + BASIS_DIM]
        terms_l = " + ".join([f"{coeffs_l[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)])
        print(f"  steer[k-{d}] Linear: {terms_l}")
        col += BASIS_DIM
        
        # Cubic
        coeffs_c = theta[col:col + BASIS_DIM]
        terms_c = " + ".join([f"{coeffs_c[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)])
        print(f"  steer[k-{d}] Cubic:  {terms_c}")
        col += BASIS_DIM

    # 4. Exogenous: aEgo
    coeffs = theta[col:col + BASIS_DIM]
    terms = " + ".join([f"{coeffs[i]:.6f}*{basis_names[i]}" for i in range(BASIS_DIM)])
    print(f"\naEgo (Speed-Dep): {terms}")
    col += BASIS_DIM

    # 5. Exogenous: Roll
    print(f"Roll (Static):    {theta[col]:.6f}*1")
    col += 1

    print("\nConstant term (intercept):", theta[0])    

    np.savez(
        "lpv_arx_model.npz",
        theta=theta,
        NA=NA,
        NUM_STEER_TERMS=NUM_STEER_TERMS,
        BASIS_DIM=BASIS_DIM,
        feature_order=["bias", "ay_lags", "steer_linear_cubic", "aEgo", "roll_static"]
    )

    rmse = np.sqrt(np.mean((X @ theta - y) ** 2))
    print(f"One-step RMSE: {rmse:.4f} m/s²")


    # Pick one or more files you want to visualize
    example_file = "data_excitation/00010_excitation_sine.csv"   # change to any valid route
    # plot_simulation_on_file(theta, example_file, NA)
