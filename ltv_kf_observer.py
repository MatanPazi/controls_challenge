"""
ltv_kf_observer.py

Implements a discrete Linear Time-Varying (LTV) Kalman Filter for state estimation
using the fitted LPV-ARX model.

State vector: [ay_k, ay_{k-1}, δ_{k-1}, δ_{k-2}, δ_{k-3}, b_k]

Requires:
- lpv_arx_theta.npy
- lpv_arx.py (for lpv basis, MIN_SPEED, CONTROL_START_IDX, BASIS_DIM)

Usage:
    python ltv_kf_observer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lpv_arx import (
    lpv,                    # LPV basis function: returns shape (3,) for quadratic in v
    MIN_SPEED,
    CONTROL_START_IDX,
    BASIS_DIM
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

THETA_FILE = "lpv_arx_theta.npy"
TEST_FILE  = "data/00010.csv"  # change to any hold-out route

# Noise covariances — tune these!
R = 0.003                  # Measurement variance on (ay + bias)
Q_diag = [0.015, 0.008, 1e-6, 1e-6, 1e-6, 1e-5]  # on [ay_k, ay_{k-1}, δ1, δ2, δ3, b_k]

# Small regularization to help with numerical stability
P0_diag = [0.1, 0.05, 0.01, 0.01, 0.01, 0.02]
# P0_diag = None

# ──────────────────────────────────────────────
# Model functions
# ──────────────────────────────────────────────

def get_A(v_k, theta):
    v_lpv = lpv(np.array([v_k]))  # (3,)
    A = np.zeros((6, 6))

    col = 0
    # ay_{k+1} ← ay_k
    A[0, 0] = np.dot(v_lpv, theta[col:col+BASIS_DIM]).item()
    # ay_{k+1} ← ay_{k-1}
    A[0, 1] = np.dot(v_lpv, theta[col+BASIS_DIM:col+2*BASIS_DIM]).item()
    col += 2 * BASIS_DIM

    # ay_{k+1} ← δ_{k-1}, δ_{k-2}, δ_{k-3}
    A[0, 2] = np.dot(v_lpv, theta[col:col+BASIS_DIM]).item()
    A[0, 3] = np.dot(v_lpv, theta[col+BASIS_DIM:col+2*BASIS_DIM]).item()
    A[0, 4] = np.dot(v_lpv, theta[col+2*BASIS_DIM:col+3*BASIS_DIM]).item()
    col += 3 * BASIS_DIM

    # exogenous → already in E, not in A

    # Companion / shift part
    A[1, 0] = 1.0           # ay_k   → ay_{k} (previous for next step)
    A[3, 2] = 1.0           # δ_{k-1} → δ_{k-2}
    A[4, 3] = 1.0           # δ_{k-2} → δ_{k-3}
    A[5, 5] = 1.0           # bias random walk (no deterministic change)

    return A


def get_B(theta):
    B = np.zeros((6, 1))
    B[2, 0] = 1.0  # current u_k = δ_k goes into x_next[2] (the new δ_{k-1} position for next step)
    return B


def get_E(v_k: float, theta: np.ndarray) -> np.ndarray:
    """Exogenous input matrix E(v_k) — shape (6,3) for [v, a, roll]"""
    v_lpv = lpv(np.array([v_k]))           # (3,)
    E = np.zeros((6, 3))

    col = 2 * BASIS_DIM + 3 * BASIS_DIM    # after ay + delta coeffs
    # exogenous → ay_{k+1}
    E[0, 0] = np.dot(v_lpv, theta[col:col+BASIS_DIM]).item()                # v_k
    E[0, 1] = np.dot(v_lpv, theta[col+BASIS_DIM:col+2*BASIS_DIM]).item()    # a_k
    E[0, 2] = np.dot(v_lpv, theta[col+2*BASIS_DIM:col+3*BASIS_DIM]).item()  # r_k

    return E


def predict_state(x, u, zeta, theta):
    v_k, a_k, r_k = zeta
    A = get_A(v_k, theta)          # ← new full A (see below)
    B = get_B(theta)               # input matrix for current u = δ_k
    E = get_E(v_k, theta)

    u_vec = np.array([u])
    z_vec = np.array([v_k, a_k, r_k])   

    x_next = A @ x + B.flatten() * u + E @ z_vec
    
    return x_next


def measure_state(x: np.ndarray, theta=None) -> np.ndarray:
    """h(x) = ay_k + b_k"""
    return np.array([x[0] + x[5]])


# ──────────────────────────────────────────────
# Linear Time-Varying Kalman Filter
# ──────────────────────────────────────────────

class LTV_KF:
    def __init__(self, n: int, R: float, Q_diag: list, theta: np.ndarray,
                 x0: np.ndarray = None, P0_diag: list = None):
        self.n = n
        self.m = 1                      # scalar measurement
        self.theta = theta

        self.Q = np.diag(Q_diag)
        self.R = np.array([[R]]) if np.isscalar(R) else np.atleast_2d(R)

        # Initial state / covariance
        self.x = x0 if x0 is not None else np.zeros(n)
        self.P = np.diag(P0_diag) if P0_diag is not None else np.eye(n) * 0.1

        # For logging / debugging
        self.last_innovation = None
        self.last_K = None
        self.last_S = None

    def predict(self, u: float, zeta: np.ndarray):
        """Prediction step: x_{k|k-1}, P_{k|k-1}"""
        # Deterministic dynamics
        x_pred = predict_state(self.x, u, zeta, self.theta)

        # Jacobian is A(v_k) — we need it for covariance propagation
        v_k = zeta[0]
        A = get_A(v_k, self.theta)

        # P = A P A^T + Q
        self.P = A @ self.P @ A.T + self.Q
        self.x = x_pred

    def update(self, y: float):
        """Measurement update"""
        H = np.array([[1.0, 0, 0, 0, 0, 1.0]])  # observe ay + bias

        # Innovation
        z_hat = (H @ self.x)[0]
        innovation = y - z_hat

        # Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + self.R
        S = (S + S.T) / 2  # ensure symmetry

        # Kalman gain
        K = (self.P @ H.T) @ np.linalg.solve(S, np.eye(self.m))
        K = K.flatten()  # shape (n,)

        # Update state
        self.x += K * innovation

        # Joseph form covariance update (more stable)
        I_KH = np.eye(self.n) - np.outer(K, H[0])
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R[0,0]

        # Save for debugging
        self.last_innovation = innovation
        self.last_K = K
        self.last_S = S

        return innovation


# ──────────────────────────────────────────────
# Main: run LTV-KF on one hold-out file
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load parameters
    theta = np.load(THETA_FILE)
    print(f"Loaded theta: shape {theta.shape}")

    # Load test data
    df = pd.read_csv(TEST_FILE)

    ay    = df["targetLateralAcceleration"].values
    steer = df["steerCommand"].values
    v     = df["vEgo"].values
    a     = df["aEgo"].values
    roll  = df["roll"].values

    # Filter to valid region
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

    # Initialize filter
    n = 6
    kf = LTV_KF(
        n=n,
        R=R,
        Q_diag=Q_diag,
        theta=theta,
        P0_diag=P0_diag
    )

    # Run filter
    filtered_ay = []
    innovations = []
    bias = []

    for k in range(len(ay)):
        u_k = steer[k]
        zeta_k = np.array([v[k], a[k], roll[k]])

        # Predict
        kf.predict(u_k, zeta_k)

        # Update
        y_k = ay[k]
        inn = kf.update(y_k)
        innovations.append(inn)

        # Save filtered ay (without bias)
        filtered_ay.append(kf.x[0])

        # Save bias
        bias.append(kf.x[5])

    filtered_ay = np.array(filtered_ay)
    innovations = np.array(innovations)

    # ─── Plotting ───────────────────────────────────────────────────────
    plt.figure(figsize=(12, 9))

    plt.subplot(3,1,1)
    plt.plot(ay, label='Measured ay (targetLatAccel)', alpha=0.7)
    plt.plot(filtered_ay, label='Filtered ay (KF)', linewidth=2.2)
    plt.title(f"Raw vs Filtered Lateral Acceleration\n{Path(TEST_FILE).name}")
    plt.ylabel("ay [m/s²]")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3,1,2)
    plt.plot(bias,  # approx with final bias
             label='bias', color='C3', ls='--')
    plt.ylabel("bias est.")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3,1,3)
    plt.plot(innovations, label='Innovation (y - ŷ)')
    plt.axhline(0, color='gray', ls='--', alpha=0.6)
    plt.title("Innovations — should look like white noise if well tuned")
    plt.xlabel("Time step")
    plt.ylabel("Innovation [m/s²]")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final state estimate:\n{kf.x}")
    print(f"Final bias estimate: {kf.x[5]:.6f}")
    print("Done. Tune Q / R until innovations are white and filtered trace is smooth yet responsive.")