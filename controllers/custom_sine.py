# custom_sine.py
from . import BaseController
import numpy as np

class Controller(BaseController):
    def __init__(self):
        self.step_idx = 0
        self.amplitude = 0.8
        self.period = 80
        self.n_cycles = 5
        self.omega = 2 * np.pi / self.period

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_idx += 1
        if self.step_idx > self.n_cycles * (2 * np.pi / self.omega):
            return 0.0
        return self.amplitude * np.sin(self.omega * self.step_idx)