from . import BaseController

class Controller(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return 0.6