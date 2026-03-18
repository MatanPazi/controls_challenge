# custom_chirp.py
from . import BaseController
import numpy as np

class Controller(BaseController):
    def __init__(self):
        self.step_idx = 0
        self.amp = 0.6
        self.f_start = 0.05
        self.f_end = 0.8
        self.duration = 400
        self.k = (self.f_end - self.f_start) / self.duration

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_idx += 1
        if self.step_idx > self.duration:
            return 0.0
        phase = 2 * np.pi * (self.f_start * self.step_idx + 0.5 * self.k * self.step_idx**2)
        return self.amp * np.sin(phase)