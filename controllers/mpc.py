from . import BaseController
import numpy as np
import scipy.optimize as opt

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
        self.N = 5
        self.steer_candidates = np.linspace(-1, 1, 101)
        self.lambda_u = 1.0
        self.last_u_seq = None          # warm-start for optimizer       

        with open("mpc_log.txt", "w") as f:
            f.write("step,ay,ay_ref,steer,ay_pred\n")  # header        

    def simulate_rollout(self, x, future_zetas, u_seq, target):
        cost = 0.0

        # Safety guard
        horizon = min(len(u_seq), len(future_zetas))
        if horizon == 0:
            return 0.0

        u_seq = u_seq[:horizon]

        steer_hist = [x[4], x[3], x[2]]   # oldest, mid, newest
        ay = x[0]
        ay_prev = x[1]

        for k in range(horizon):
            u = u_seq[k]

            state = np.array([
                ay,
                ay_prev,
                steer_hist[-1],             # Newest     
                steer_hist[-2],
                steer_hist[-3],             # Oldest
                0.0
            ])

            zeta_k = future_zetas[k]
            ay_pred = predict_next_lataccel(state, u, zeta_k, self.theta)

            cost += (ay_pred - target) ** 2
            if k > 0:
                cost += self.lambda_u * (u - u_seq[k-1]) ** 2

            ay_prev = ay
            ay = ay_pred
            steer_hist.append(u)
            steer_hist = steer_hist[-3:]

        # Terminal cost
        cost += 10.0 * (ay - target) ** 2

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

        # === SAFE horizon handling (fixes both previous IndexError and this new bounds error) ===
        N = self.N
        v_ego = np.asarray(future_plan.v_ego)
        a_ego = np.asarray(future_plan.a_ego)
        roll   = np.asarray(future_plan.roll_lataccel)

        actual_horizon = min(N, len(v_ego))

        if actual_horizon == 0:
            # End of route: just return last steer or 0
            steer = self.steer_hist[-1] if self.steer_hist else 0.0
        else:
            future_zetas = np.column_stack((
                v_ego[:actual_horizon],
                a_ego[:actual_horizon],
                roll[:actual_horizon]
            ))  # shape (actual_horizon, 3)

            def mpc_cost(u_vec: np.ndarray) -> float:
                return self.simulate_rollout(x0, future_zetas, u_vec.tolist(), target_lataccel)

            # warm-start
            if self.last_u_seq is None or len(self.last_u_seq) != actual_horizon:
                u0_guess = np.zeros(actual_horizon)
            else:
                u0_guess = self.last_u_seq[:actual_horizon].copy()

            res = opt.minimize(
                mpc_cost,
                x0=u0_guess,
                bounds=[(-1.0, 1.0)] * actual_horizon,
                method='L-BFGS-B',
                options={'maxiter': 100, 'disp': False}
            )

            if res.success:
                u_seq_opt = res.x
                steer = float(u_seq_opt[0])          # apply only first move
                self.last_u_seq = u_seq_opt          # save for next warm-start
            else:
                steer = 0.0

        # logging — safe even when horizon==0
        if actual_horizon > 0:
            zeta0 = future_zetas[0]
        else:
            # fallback zeta when no future data
            zeta0 = np.array([
                future_plan.v_ego[-1] if len(future_plan.v_ego) > 0 else 0.0,
                future_plan.a_ego[-1] if len(future_plan.a_ego) > 0 else 0.0,
                future_plan.roll_lataccel[-1] if len(future_plan.roll_lataccel) > 0 else 0.0
            ])
        ay_pred = predict_next_lataccel(x0, steer, zeta0, self.theta)

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
