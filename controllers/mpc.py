from . import BaseController
import numpy as np


def predict_next_lataccel(state, steer, zeta, theta):
    ay, ay_prev, d1, d2, d3, _ = state
    v, a, roll = zeta

    basis = np.array([1.0, v, v * v])

    col = 0

    # AR
    ay_pred = (
        np.dot(basis, theta[col:col+3]) * ay +
        np.dot(basis, theta[col+3:col+6]) * ay_prev
    )
    col += 6

    # Input delays
    ay_pred += (
        np.dot(basis, theta[col:col+3]) * d1 +
        np.dot(basis, theta[col+3:col+6]) * d2 +
        np.dot(basis, theta[col+6:col+9]) * d3
    )
    col += 9

    # Exogenous
    ay_pred += (
        np.dot(basis, theta[col:col+3]) * v +
        np.dot(basis, theta[col+3:col+6]) * a +
        np.dot(basis, theta[col+6:col+9]) * roll
    )

    return ay_pred


class Controller(BaseController):
    def __init__(self):
        self.theta = np.load("lpv_arx_theta.npy")

        self.prev_ay = 0.0
        self.steer_hist = [0.0, 0.0, 0.0]

        self.step_idx = 0

        # MPC parameters
        self.N = 10
        self.steer_candidates = np.linspace(-1, 1, 101)
        self.lambda_u = 1.0

        with open("mpc_log.txt", "w") as f:
            f.write("step,ay,ay_ref,steer,ay_pred\n")  # header        

    def simulate_rollout(self, x, zeta, u_seq, target):
        cost = 0.0

        steer_hist = list(x[2:5])
        ay = x[0]
        ay_prev = x[1]

        for k in range(len(u_seq)):
            u = u_seq[k]

            state = np.array([
                ay,
                ay_prev,
                steer_hist[-1],
                steer_hist[-2],
                steer_hist[-3],
                0.0
            ])

            ay_pred = predict_next_lataccel(state, u, zeta, self.theta)

            # cost: tracking
            cost += (ay_pred - target) ** 2

            # cost: smoothness
            if k > 0:
                cost += self.lambda_u * (u - u_seq[k-1]) ** 2

            # shift state
            ay_prev = ay
            ay = ay_pred

            steer_hist.append(u)
            steer_hist = steer_hist[-3:]

        return cost

    def update(self, target_lataccel, current_lataccel, state, future_plan):

        # initial state
        x0 = np.array([
            current_lataccel,
            self.prev_ay,
            self.steer_hist[-1],
            self.steer_hist[-2],
            self.steer_hist[-3],
            0.0
        ])

        zeta = np.array([
            future_plan.v_ego[0],
            future_plan.a_ego[0],
            future_plan.roll_lataccel[0]
        ])

        # brute-force MPC (receding horizon, greedy sequence)
        best_cost = np.inf
        best_u0 = 0.0

        for u0 in self.steer_candidates:
            # build constant sequence (simple but robust)
            u_seq = [u0] * self.N

            cost = self.simulate_rollout(x0, zeta, u_seq, target_lataccel)

            if cost < best_cost:
                best_cost = cost
                best_u0 = u0

        steer = best_u0

        # logging (optional)
        ay_pred = predict_next_lataccel(x0, steer, zeta, self.theta)

        with open("mpc_log.txt", "a") as f:
            f.write(
                f"{self.step_idx},"
                f"{current_lataccel},"
                f"{target_lataccel},"
                f"{steer},"
                f"{ay_pred}\n"
            )

        # update history
        self.prev_ay = current_lataccel

        self.steer_hist.append(steer)
        self.steer_hist = self.steer_hist[-3:]

        self.step_idx += 1

        return steer