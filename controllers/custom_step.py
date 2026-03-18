from . import BaseController

class Controller(BaseController):
    def __init__(self):
        self.step_idx = 0
        self.command_mag = 0.1      # +/- steering magnitude
        self.hold_samples = 10      # samples per level

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_idx += 1

        # Full cycle length: + then -
        cycle_len = 2 * self.hold_samples
        cycle_pos = self.step_idx % cycle_len

        if cycle_pos < self.hold_samples:
            return self.command_mag
        else:
            return -self.command_mag
