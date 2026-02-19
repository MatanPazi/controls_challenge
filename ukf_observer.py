"""
ukf_observer.py

Implements an Unscented Kalman Filter (UKF) for state estimation using the fitted LPV-ARX model.
Estimates the full state [ay_k, ay_{k-1}, δ_{k-1}, δ_{k-2}, δ_{k-3}, b_k] from noisy ay measurements.

Requires:
- lpv_arx_theta.npy (fitted parameters)
- lpv_arx.py in same folder (for lpv basis, MIN_SPEED, CONTROL_START_IDX)
- A hold-out .csv file from data/

Usage:
    python ukf_observer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lpv_arx import (
    lpv,                    # LPV basis function
    MIN_SPEED,
    CONTROL_START_IDX,
    BASIS_DIM
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

THETA_FILE = "lpv_arx_theta.npy"
TEST_FILE  = "data/00010.csv"  # change to any hold-out route you like

# Noise covariances (tune these!)
R = 0.003           # Measurement noise variance (inflated ~5× from raw measured 0.00058)
Q_diag = [0.015, 0.008, 1e-6, 1e-6, 1e-6, 1e-5]  # Process noise on [ay_k, ay_{k-1}, δ1, δ2, δ3, b_k]

# UKF hyperparameters (standard defaults, rarely need to change)
alpha = 1e-3        # Spread of sigma points
beta  = 2.0         # For Gaussian, optimal = 2
kappa = 0.0         # Secondary scaling

# ──────────────────────────────────────────────
# Load fitted theta
# ──────────────────────────────────────────────

theta = np.load(THETA_FILE)
print(f"Loaded theta from {THETA_FILE} — shape: {theta.shape}")

# ──────────────────────────────────────────────
# Helper: predict next state (deterministic part of your model)
# ──────────────────────────────────────────────

def predict_state(x, u, zeta, theta):
    """
    x: current state (6,)
    u: current steer command δ_k
    zeta: current exogenous [v_k, a_k, r_k]
    theta: fitted parameters (24,)
    Returns: predicted x_{k+1} (6,)
    """
    # Extract components
    ay_k      = x[0]
    ay_km1    = x[1]
    delta_km1 = x[2]
    delta_km2 = x[3]
    delta_km3 = x[4]
    b_k       = x[5]

    v_k, a_k, r_k = zeta

    v_lpv = lpv(np.array([v_k]))[0]  # (3,)

    pred = np.zeros(6)

    col = 0

    # ay_{k+1} = ay_k * basis + ay_{k-1} * basis + ...
    pred[0] = (np.dot(v_lpv, theta[col:col+BASIS_DIM]) * ay_k +
               np.dot(v_lpv, theta[col+BASIS_DIM:col+2*BASIS_DIM]) * ay_km1)
    col += 2 * BASIS_DIM

    # delta contributions to ay_{k+1}
    pred[0] += (np.dot(v_lpv, theta[col:col+BASIS_DIM]) * delta_km1 +
                np.dot(v_lpv, theta[col+BASIS_DIM:col+2*BASIS_DIM]) * delta_km2 +
                np.dot(v_lpv, theta[col+2*BASIS_DIM:col+3*BASIS_DIM]) * delta_km3)
    col += 3 * BASIS_DIM

    # exogenous contributions to ay_{k+1}
    pred[0] += (np.dot(v_lpv, theta[col:col+BASIS_DIM]) * v_k +
                np.dot(v_lpv, theta[col+BASIS_DIM:col+2*BASIS_DIM]) * a_k +
                np.dot(v_lpv, theta[col+2*BASIS_DIM:col+3*BASIS_DIM]) * r_k)
    col += 3 * BASIS_DIM

    # Shift states (companion form style)
    pred[1] = ay_k          # ay_{k} becomes ay_{k}
    pred[2] = u             # δ_k becomes δ_{k}
    pred[3] = delta_km1     # δ_{k-1} becomes δ_{k-1}
    pred[4] = delta_km2     # δ_{k-2} becomes δ_{k-2}
    pred[5] = b_k           # bias assumed random walk (update in noise)

    return pred

# ──────────────────────────────────────────────
# UKF Class (simplified discrete-time version)
# ──────────────────────────────────────────────

class UKF:
    def __init__(self, n, R, Q_diag,
                 alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = n
        self.R = float(R)
        self.Q = np.diag(Q_diag)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lmbda = alpha**2 * (n + kappa) - n
        self.gamma = n + self.lmbda

        # Weights
        self.Wm = np.full(2*n + 1, 1/(2*self.gamma))
        self.Wc = np.full(2*n + 1, 1/(2*self.gamma))
        self.Wm[0] = self.lmbda / self.gamma
        self.Wc[0] = self.lmbda / self.gamma + (1 - alpha**2 + beta)

        # State and covariance
        self.x = np.zeros(n)
        self.P = np.eye(n) * 0.1

        self.jitter = 1e-9

    # ──────────────────────────────────────────
    def sigma_points(self, x, P):
        P = 0.5 * (P + P.T)                      # enforce symmetry
        P += self.jitter * np.eye(self.n)        # jitter

        S = np.linalg.cholesky(self.gamma * P)
        X = np.zeros((2*self.n + 1, self.n))
        X[0] = x

        for i in range(self.n):
            X[i+1]        = x + S[:, i]
            X[self.n+i+1] = x - S[:, i]

        return X

    # ──────────────────────────────────────────
    def predict(self, u, zeta):
        X = self.sigma_points(self.x, self.P)

        Xp = np.array([
            predict_state(X[i], u, zeta, theta)
            for i in range(2*self.n + 1)
        ])

        # Mean
        self.x = np.sum(self.Wm[:, None] * Xp, axis=0)

        # Covariance
        P = self.Q.copy()
        for i in range(2*self.n + 1):
            dx = Xp[i] - self.x
            P += self.Wc[i] * np.outer(dx, dx)

        self.P = 0.5 * (P + P.T)   # enforce symmetry

    # ──────────────────────────────────────────
    def update(self, y_meas):
        X = self.sigma_points(self.x, self.P)

        # Measurement model: y = ay_k = x[0]
        Z = X[:, 0]

        z_hat = np.sum(self.Wm * Z)

        # Innovation covariance
        S = self.R
        for i in range(2*self.n + 1):
            dz = Z[i] - z_hat
            S += self.Wc[i] * dz * dz

        # Cross covariance
        Pxz = np.zeros(self.n)
        for i in range(2*self.n + 1):
            dx = X[i] - self.x
            dz = Z[i] - z_hat
            Pxz += self.Wc[i] * dx * dz

        # Kalman gain
        K = Pxz / S

        innovation = y_meas - z_hat

        # State update
        self.x = self.x + K * innovation

        # Covariance update (Joseph-safe for scalar case)
        self.P = self.P - np.outer(K, K) * S
        self.P = 0.5 * (self.P + self.P.T) + self.jitter * np.eye(self.n)

        return innovation


# ──────────────────────────────────────────────
# Main: run UKF on one hold-out file
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load theta
    theta = np.load(THETA_FILE)
    print(f"Loaded theta: shape {theta.shape}")

    # Load test file
    df = pd.read_csv(TEST_FILE)

    ay    = df["targetLateralAcceleration"].values
    steer = df["steerCommand"].values
    v     = df["vEgo"].values
    a     = df["aEgo"].values
    roll  = df["roll"].values

    valid = (
        (np.arange(len(df)) < CONTROL_START_IDX) &  # use post-control for closed-loop feel
        (v >= MIN_SPEED) &
        np.isfinite(ay) &
        np.isfinite(v)
    )

    ay    = ay[valid]
    steer = np.nan_to_num(steer[valid], nan=0.0)
    v     = v[valid]
    a     = a[valid]
    roll  = roll[valid]

    if len(ay) < CONTROL_START_IDX // 2:
        print("Test file too short")
        exit()

    # Initialize UKF
    n = 6
    ukf = UKF(n, R, Q_diag)

    # Run filter
    filtered_ay = []
    innovations = []

    for k in range(len(ay)):
        # Predict
        u_k = steer[k]
        zeta_k = np.array([v[k], a[k], roll[k]])
        ukf.predict(u_k, zeta_k)

        # Update with measurement
        y_k = ay[k]
        inn = ukf.update(y_k)
        innovations.append(inn)

        # Store filtered ay_k (x[0])
        filtered_ay.append(ukf.x[0])

    filtered_ay = np.array(filtered_ay)
    innovations = np.array(innovations)

    # ─── Plots ───────────────────────────────────────────────────────
    plt.figure(figsize=(12, 8))

    # Raw vs filtered ay
    plt.subplot(2,1,1)
    plt.plot(ay, label='Raw ay (noisy measurement)', alpha=0.7)
    plt.plot(filtered_ay, label='Filtered ay (UKF)', linewidth=2)
    plt.title(f"Raw vs Filtered Lateral Acceleration\n{Path(TEST_FILE).name}")
    plt.xlabel("Time step")
    plt.ylabel("ay [m/s²]")
    plt.legend()
    plt.grid(alpha=0.3)

    # Innovations (should be white noise if filter is well-tuned)
    plt.subplot(2,1,2)
    plt.plot(innovations)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Innovations (y - predicted y)")
    plt.xlabel("Time step")
    plt.ylabel("Innovation")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final filtered bias estimate: {ukf.x[5]:.6f}")
    print("Done. Tune R/Q and re-run until filtered trace is smooth but responsive.")