"""
analyze_residuals.py

Standalone script to:
1. Load the fitted LPV-ARX parameters from lpv_arx_theta.npy
2. Compute one-step prediction residuals on specified hold-out routes
3. Plot histogram + normal fit
4. Compute statistics and normality tests
5. Estimate measurement noise variance R
6. Investigate outlier causes:
   - Residuals vs time (per file)
   - Residuals vs speed (vEgo)
   - Residuals vs |actual ay|
   - Residuals vs absolute steer rate (sudden changes)

Requires:
- lpv_arx_theta.npy (from running lpv_arx.py)
- lpv_arx.py in the same directory (for lpv, MIN_SPEED, CONTROL_START_IDX)
- data/*.csv files

Usage:
    python analyze_residuals.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from pathlib import Path
import glob


# ──────────────────────────────────────────────
# Import shared constants & lpv basis from lpv_arx.py
# ──────────────────────────────────────────────

from lpv_arx import (
    MIN_SPEED,
    CONTROL_START_IDX,
    lpv,
    NA,
    NDELTA,
    NUM_EXO_VAR,   # or NUM_EXO if that's the name you used
    BASIS_DIM
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

THETA_FILE = "lpv_arx_theta.npy"

# Hold-out files (example: routes after first 1000)
HOLD_OUT_FILES = sorted(glob.glob("data/*.csv"))[1000:1200]

# ──────────────────────────────────────────────
# Load one route and apply filtering
# ──────────────────────────────────────────────

def load_and_filter_route(file_path):
    df = pd.read_csv(file_path)

    required = ["targetLateralAcceleration", "steerCommand", "vEgo", "aEgo", "roll"]
    if not all(col in df.columns for col in required):
        print(f"Skipping {Path(file_path).name} — missing columns")
        return None

    ay    = df["targetLateralAcceleration"].values
    steer = df["steerCommand"].values
    v     = df["vEgo"].values
    a     = df["aEgo"].values
    roll  = df["roll"].values

    valid = (
        (np.arange(len(df)) < CONTROL_START_IDX) &
        (v >= MIN_SPEED) &
        np.isfinite(ay) &
        np.isfinite(steer) &
        np.isfinite(v) &
        np.isfinite(a) &
        np.isfinite(roll)
    )

    if np.sum(valid) < 50:
        print(f"No sufficient valid rows in {Path(file_path).name}")
        return None

    return (
        ay[valid],
        np.nan_to_num(steer[valid], nan=0.0),
        v[valid],
        a[valid],
        roll[valid]
    )

# ──────────────────────────────────────────────
# One-step prediction for one route
# ──────────────────────────────────────────────

def predict_one_step(ay, steer, v, a, roll, theta):
    N = len(ay)
    k0 = max(NA, NDELTA)
    if N <= k0:
        return np.array([])

    pred = np.zeros(N - k0)

    v_lpv = lpv(v[k0:])  # (Ns, BASIS_DIM)

    col = 0

    for k in range(k0, N):
        p = 0.0
        c = col

        for i in range(1, NA + 1):
            p += np.dot(v_lpv[k - k0], theta[c:c+BASIS_DIM]) * ay[k - i]
            c += BASIS_DIM

        for d in range(1, NDELTA + 1):
            p += np.dot(v_lpv[k - k0], theta[c:c+BASIS_DIM]) * steer[k - d]
            c += BASIS_DIM

        for z in [v[k], a[k], roll[k]]:
            p += np.dot(v_lpv[k - k0], theta[c:c+BASIS_DIM]) * z
            c += BASIS_DIM

        pred[k - k0] = p

    return pred

# ──────────────────────────────────────────────
# Analyze outlier causes
# ──────────────────────────────────────────────

def analyze_outliers(ay_list, steer_list, v_list, pred_list, res_list):
    """
    Plot residuals vs time, speed, |ay|, |steer rate|.
    Aggregates from all hold-out files.
    """
    # Concatenate everything
    ay_all    = np.concatenate(ay_list)
    steer_all = np.concatenate(steer_list)
    v_all     = np.concatenate(v_list)
    pred_all  = np.concatenate(pred_list)
    res_all   = np.concatenate(res_list)

    # Steer rate (absolute change per step, assume Δt=0.1 s)
    # Length matches res_all exactly
    steer_diff = np.diff(steer_all, prepend=steer_all[0])  # first diff=0
    steer_rate = np.abs(steer_diff) / 0.1

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("Residuals vs Various Factors (Outlier Analysis)")

    # 1. vs time (first file only, for visual clarity)
    if res_list:
        axes[0,0].scatter(np.arange(len(res_list[0])), res_list[0], s=8, alpha=0.6)
        axes[0,0].axhline(0, color='gray', linestyle='--')
        axes[0,0].set_title("Residuals vs Time (first hold-out file)")
        axes[0,0].set_xlabel("Time step")
        axes[0,0].set_ylabel("Residual [m/s²]")
        axes[0,0].grid(alpha=0.3)

    # 2. vs speed
    axes[0,1].scatter(v_all, res_all, s=6, alpha=0.4)
    axes[0,1].axhline(0, color='gray', linestyle='--')
    axes[0,1].set_title("Residuals vs vEgo")
    axes[0,1].set_xlabel("Speed [m/s]")
    axes[0,1].grid(alpha=0.3)

    # 3. vs |actual ay|
    axes[1,0].scatter(np.abs(ay_all), res_all, s=6, alpha=0.4)
    axes[1,0].axhline(0, color='gray', linestyle='--')
    axes[1,0].set_title("Residuals vs |Actual ay|")
    axes[1,0].set_xlabel("|ay| [m/s²]")
    axes[1,0].grid(alpha=0.3)

    # 4. vs |steer rate| — now same length
    axes[1,1].scatter(steer_rate, res_all, s=6, alpha=0.4)
    axes[1,1].axhline(0, color='gray', linestyle='--')
    axes[1,1].set_title("Residuals vs |Steer Rate|")
    axes[1,1].set_xlabel("|Δsteer / Δt| [rad/s]")
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Quick printed insights
    print("\nOutlier Analysis Insights:")
    print(f"  Max |residual| = {np.max(np.abs(res_all)):.4f}")
    print(f"  Mean |residual| at |ay| > 0.4: {np.mean(np.abs(res_all[np.abs(ay_all)>0.4])):.4f}")
    print(f"  Mean |residual| at v < 5 m/s: {np.mean(np.abs(res_all[v_all<5])):.4f}")
    print(f"  Mean |residual| at |steer rate| > 0.5 rad/s: {np.mean(np.abs(res_all[steer_rate>0.5])):.4f}")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load theta
    if not Path(THETA_FILE).exists():
        print(f"Error: {THETA_FILE} not found. Run lpv_arx.py first.")
        exit(1)

    theta = np.load(THETA_FILE)
    print(f"Loaded theta: shape {theta.shape}")

    # Lists to collect for outlier plots
    ay_list    = []
    steer_list = []
    v_list     = []
    pred_list  = []
    res_list   = []

    # Compute residuals
    for fpath in HOLD_OUT_FILES:
        data = load_and_filter_route(fpath)
        if data is None:
            continue

        ay, steer, v, a, roll = data
        N = len(ay)

        if N < max(NA, NDELTA) + 20:
            print(f"Skipping {Path(fpath).name} — too short ({N} steps)")
            continue

        pred = predict_one_step(ay, steer, v, a, roll, theta)
        true_y = ay[max(NA, NDELTA):]

        res = pred - true_y

        # Store for plots
        ay_list.append(ay[max(NA, NDELTA):])
        steer_list.append(steer[max(NA, NDELTA):])
        v_list.append(v[max(NA, NDELTA):])
        pred_list.append(pred)
        res_list.append(res)

        print(f"{Path(fpath).name:12} | {len(res):5d} residuals | mean: {np.mean(res):.6f}")

    if not res_list:
        print("No valid residuals collected")
        exit(1)

    residuals = np.concatenate(res_list)
    print(f"\nTotal residuals collected: {len(residuals):,}")

    # ─── Basic stats & Gaussian tests ────────────────────────────────
    mu   = np.mean(residuals)
    var  = np.var(residuals)
    std  = np.std(residuals)
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)

    sw_n = min(5000, len(residuals))
    sw_stat, sw_p = stats.shapiro(residuals[:sw_n])

    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(mu, std))

    print("\nResidual Statistics:")
    print(f"  Mean:          {mu:.6f}")
    print(f"  Variance (R):  {var:.6f}")
    print(f"  Std dev:       {std:.6f}")
    print(f"  Skewness:      {skew:.4f}  (0 = symmetric)")
    print(f"  Kurtosis:      {kurt:.4f}  (0 = Gaussian)")
    print(f"  Shapiro-Wilk p: {sw_p:.4f}  (>0.05 → fail to reject Gaussian)")
    print(f"  KS statistic:  {ks_stat:.4f}")

    # ─── Histogram ───────────────────────────────────────────────────
    plt.figure(figsize=(9, 5.5))
    plt.hist(residuals, bins=80, density=True, alpha=0.6, color='cornflowerblue')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    plt.plot(x, norm.pdf(x, mu, std), 'k', lw=2, label='Fitted Gaussian')
    plt.title(f"Residual Histogram (n = {len(residuals):,})\nEstimated R = {var:.6f}")
    plt.xlabel("Prediction error [m/s²]")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ─── Outlier cause analysis ──────────────────────────────────────
    print("\nGenerating outlier analysis plots...")
    analyze_outliers(ay_list, steer_list, v_list, pred_list, res_list)

    print(f"\nRecommended R for Kalman filter:")
    print(f"  Raw estimate: {var:.6f}")
    print(f"  Conservative (inflate 3–5×): {3*var:.6f} – {5*var:.6f}")