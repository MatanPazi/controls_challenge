from . import BaseController

class Controller(BaseController):
    def __init__(self):
        self.step_idx = 0
        self.duration = 400

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_idx += 1
        t_norm = min(self.step_idx / self.duration, 1.0)
        return 0.0 + t_norm * 1.2