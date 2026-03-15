from . import BaseController
import json
from pathlib import Path

class Controller(BaseController):
    """
    Tunable PID — gains are loaded from pid_gains.json in the repo root
    """
    def __init__(self):
        gains_path = Path(__file__).parent.parent / "pid_gains.json"
        
        if gains_path.exists():
            with open(gains_path) as f:
                gains = json.load(f)
            self.p = gains.get("p", 0.195)
            self.i = gains.get("i", 0.100)
            self.d = gains.get("d", -0.053)
            print(f"[tunable_pid] Loaded gains → P={self.p:.4f} I={self.i:.4f} D={self.d:.4f}")
        else:
            # fallback to original defaults
            self.p = 0.195
            self.i = 0.100
            self.d = -0.053
            print("[tunable_pid] pid_gains.json not found — using defaults")

        self.error_integral = 0.0
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff