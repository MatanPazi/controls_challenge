from . import BaseController

class Controller(BaseController):
    """Constant step steer after control starts"""
    def __init__(self):
        self.steer_value = 0.5

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.steer_value