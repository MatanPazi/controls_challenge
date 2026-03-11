from . import BaseController
import numpy as np
from ltv_kf_observer import LTV_KF


class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    # PID parameters    
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0

    # Load your LPV-ARX model parameters
    self.theta = np.load('lpv_arx_theta.npy')  # Adjust path relative to controllers/ if needed    
        
    # Observer parameters
    self.n = 6
    self.R = 0.003                  # Measurement variance on (ay + bias)
    self.Q_diag = [0.015, 0.008, 1e-6, 1e-6, 1e-6, 1e-5]  # on [ay_k, ay_{k-1}, δ1, δ2, δ3, b_k]    
    self.P0_diag = [0.1, 0.05, 0.01, 0.01, 0.01, 0.02]  
    
    self.last_steer_commanded = 0.0

    # Initialize the observer
    self.observer = LTV_KF(
        n=self.n,
        R=self.R,
        Q_diag=self.Q_diag,
        theta=self.theta,
        P0_diag=self.P0_diag
    )                

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # ─── Observer step ────────────────────────────────────────
    u_k = self.last_steer_commanded           # the steer that was actually applied last step
    zeta_k = np.array([state.v_ego,
                        state.a_ego,
                        state.roll_lataccel])  # or state.roll_lataccel — check attribute name

    # Predict
    self.observer.predict(u_k, zeta_k)    

    # Update with noisy measurement
    y_k = current_lataccel
    _ = self.observer.update(y_k)                   # we usually don't need the innovation here

    # Get clean estimate (bias-corrected)
    estimated_ay = self.observer.x[0]    
    error = (target_lataccel - estimated_ay)


    # error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
