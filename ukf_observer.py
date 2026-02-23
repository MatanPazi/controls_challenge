"""
ukf_observer.py

Lovely video explaining the particle filter, "Particle Filter Explained without Equations":
https://youtu.be/aUkBa1zMKv4?si=IRPw7ddYIXQdBopd
Need to make one for the UKF..

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

# UKF hyperparameters.
# Values are mathematically derived, see: S. J. Julier “The Scaled Unscented Transformation”
alpha = 1e-3        # Spread of sigma points
beta  = 0.0         # For Gaussian, optimal = 2
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
    State transition function for the LPV-ARX model.

    Parameters
    ----------
    x : ndarray, shape (6,)
        Current state vector:
        [ay_k, ay_{k-1}, δ_{k-1}, δ_{k-2}, δ_{k-3}, b_k]

    u : float
        Current steering command δ_k.

    zeta : ndarray, shape (3,)
        Exogenous inputs [v_k, a_k, roll_k].

    theta : ndarray
        Identified LPV-ARX parameter vector.

    Returns
    -------
    x_next : ndarray, shape (6,)
        Predicted next state x_{k+1}.
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

def measure_state(x, theta):
    """
    Measurement function h(x).
    Maps the state vector to the observed measurement space.
    """
    ay_k = x[0]
    b_k  = x[5]
    
    # If the measurement is simply the first state:
    return np.array([ay_k]) 
    
    # If the bias in the state vector is additive to the measurement:
    # return np.array([ay_k + b_k])

# ──────────────────────────────────────────────
# UKF Class (simplified discrete-time version)
# ──────────────────────────────────────────────

class UKF:
    def __init__(self, n, R, Q_diag, theta = None,
                 predict_state=None, measure_state = measure_state,
                 alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = n
        self.R = np.atleast_2d(R)  # Ensures [[val]] if scalar or [m,m] if matrix
        self.m = self.R.shape[0]   # This is your measurement dimension
        self.Q = np.diag(Q_diag)
        self.theta = theta

        # Store the user-provided prediction function
        if predict_state is None:
            raise ValueError("predict_state function must be provided")
        self.predict_state = predict_state        

        if measure_state is None:
            def default_measure(s, theta): return s[0]  # observe position
            self.measure_state = default_measure
        else:
            self.measure_state = measure_state

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
        """
        Generate Unscented Transform sigma points.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Current state mean.

        P : ndarray, shape (n, n)
            Current state covariance.

        Returns
        -------
        sigmas : ndarray, shape (2n+1, n)
            Sigma points representing the distribution N(x, P).
        """        
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
        """
        UKF prediction step.

        Propagates the current state estimate and covariance forward
        through the nonlinear process model using sigma points.

        Parameters
        ----------
        u : float
            Control input at time k.

        zeta : ndarray
            Exogenous inputs at time k.

        Updates
        -------
        self.x : ndarray
            Predicted state mean x_{k|k-1}.

        self.P : ndarray
            Predicted state covariance P_{k|k-1}.
        """        
        X = self.sigma_points(self.x, self.P)

        Xp = np.array([
            self.predict_state(X[i], u, zeta, self.theta)
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
        """
        UKF measurement update step.

        Incorporates a scalar measurement into the predicted state
        estimate using the Unscented Transform.

        Parameters
        ----------
        y_meas : float
            Measured lateral acceleration ay_k.

        Returns
        -------
        innovation : float
            Measurement innovation (y_k - ŷ_k).
        """            
        X = self.sigma_points(self.x, self.P)

        # 1. Map sigma points to measurement space
        # Z shape: (2n+1, m)
        Z = np.array([self.measure_state(sig, self.theta) for sig in X])
        
        # 2. Predicted measurement mean (weighted sum)
        # z_hat shape: (m,)
        z_hat = np.sum(self.Wm[:, None] * Z, axis=0)

        # 3. Innovation covariance S (m x m)
        # S = sum(Wc * (Z-z_hat)(Z-z_hat)^T) + R
        S = np.zeros((self.m, self.m))
        for i in range(2*self.n + 1):
            dz = Z[i] - z_hat
            S += self.Wc[i] * np.outer(dz, dz)
        S += self.R  # Add measurement noise matrix

        # 4. Cross covariance Pxz (n x m)
        Pxz = np.zeros((self.n, self.m))
        for i in range(2*self.n + 1):
            dx = X[i] - self.x
            dz = Z[i] - z_hat
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # 5. Kalman gain (K = Pxz * inv(S))
        # Use solve for better numerical stability than inv()
        K = np.linalg.solve(S.T, Pxz.T).T

        # 6. Innovation (y - z_hat)
        # Ensure y_meas is a numpy array for vector subtraction
        innovation = np.atleast_1d(y_meas) - z_hat

        # 7. State update
        self.x = self.x + K @ innovation

        # 8. Covariance update (Joseph form or standard)
        self.P = self.P - K @ S @ K.T
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
    ukf = UKF(n, R, Q_diag, theta, predict_state)

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
