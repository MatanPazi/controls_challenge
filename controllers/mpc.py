from . import BaseController
import numpy as np
import scipy.optimize as opt


# ============================
# LPV basis (dynamic)
# ============================

def build_basis(v, BASIS_DIM):
    if BASIS_DIM == 1:
        return np.array([1.0])
    elif BASIS_DIM == 2:
        return np.array([1.0, v])
    elif BASIS_DIM == 3:
        return np.array([1.0, v, v * v])
    else:
        raise ValueError("Unsupported BASIS_DIM")


# ============================
# Dynamic predictor
# ============================

def predict_next_lataccel(state, steer, zeta, model):
    theta = model["theta"]
    NA = model["NA"]
    NUM_STEER_TERMS = model["NUM_STEER_TERMS"]
    BASIS_DIM = model["BASIS_DIM"]
    EXO_VARS = model["EXO_VARS"]

    ay_hist = state["ay_hist"]
    steer_hist = state["steer_hist"]

    v = zeta["vEgo"]
    basis = build_basis(v, BASIS_DIM)

    col = 0
    ay_pred = 0.0

    # ---- AR terms ----
    for i in range(NA):
        ay_pred += np.dot(basis, theta[col:col+BASIS_DIM]) * ay_hist[-(i+1)]
        col += BASIS_DIM

    # ---- Steering terms ----
    for d in range(NUM_STEER_TERMS):
        if d == 0:
            u = steer
        else:
            u = steer_hist[-d]
        ay_pred += np.dot(basis, theta[col:col+BASIS_DIM]) * u
        col += BASIS_DIM

    # ---- Exogenous ----
    for name in EXO_VARS:
        val = zeta.get(name, 0.0)
        ay_pred += np.dot(basis, theta[col:col+BASIS_DIM]) * val
        col += BASIS_DIM

    return ay_pred


# ============================
# Controller
# ============================

class Controller(BaseController):
    def __init__(self):
        data = np.load("lpv_arx_model.npz", allow_pickle=True)

        self.model = {
            "theta": data["theta"],
            "NA": int(data["NA"]),
            "NUM_STEER_TERMS": int(data["NUM_STEER_TERMS"]),
            "BASIS_DIM": int(data["BASIS_DIM"]),
            "EXO_VARS": list(data["EXO_VARS"])
        }

        # histories
        self.ay_hist = [0.0] * self.model["NA"]
        self.steer_hist = [0.0] * self.model["NUM_STEER_TERMS"]

        self.step_idx = 0

        # MPC params
        self.N = 10
        self.lambda_u = 10.0   # ↑ increase to reduce oscillations
        self.last_u_seq = None

        with open("mpc_log.txt", "w") as f:
            f.write("step,ay,ay_ref,steer,ay_pred\n")


    def simulate_rollout(self, future_zetas, u_seq, future_targets):
        cost = 0.0

        ay_hist = self.ay_hist.copy()
        steer_hist = self.steer_hist.copy()

        for k in range(len(u_seq)):
            u = u_seq[k]

            zeta_k = future_zetas[k]

            state = {
                "ay_hist": ay_hist,
                "steer_hist": steer_hist
            }

            ay_pred = predict_next_lataccel(state, u, zeta_k, self.model)

            # cost
            target_k = future_targets[k]
            cost += (ay_pred - target_k) ** 2

            if k > 0:
                cost += self.lambda_u * (u - u_seq[k-1]) ** 2

            # update histories
            ay_hist.append(ay_pred)
            ay_hist = ay_hist[-self.model["NA"]:]

            steer_hist.append(u)
            steer_hist = steer_hist[-self.model["NUM_STEER_TERMS"]:]

        cost += 10.0 * (ay_hist[-1] - future_targets[-1]) ** 2

        return cost


    def update(self, target_lataccel, current_lataccel, state, future_plan):

        # update ay history
        self.ay_hist.append(current_lataccel)
        self.ay_hist = self.ay_hist[-self.model["NA"]:]

        v_ego = np.asarray(future_plan.v_ego)
        a_ego = np.asarray(future_plan.a_ego)
        roll  = np.asarray(future_plan.roll_lataccel)
        lataccel_ref = np.asarray(future_plan.lataccel)

        horizon = min(self.N, len(v_ego), len(lataccel_ref))
        future_targets = lataccel_ref[:horizon]        

        if horizon == 0:
            return self.steer_hist[-1] if self.steer_hist else 0.0

        future_zetas = []
        for i in range(horizon):
            z = {
                "vEgo": v_ego[i],
                "aEgo": a_ego[i] if len(a_ego) > i else 0.0,
                "roll": roll[i] if len(roll) > i else 0.0
            }
            future_zetas.append(z)

        # warm start
        if self.last_u_seq is None or len(self.last_u_seq) != horizon:
            u0 = np.zeros(horizon)
        else:
            u0 = self.last_u_seq[:horizon]

        def cost(u_vec):
            return self.simulate_rollout(future_zetas, u_vec.tolist(), future_targets)

        res = opt.minimize(
            cost,
            x0=u0,
            bounds=[(-1, 1)] * horizon,
            method='L-BFGS-B'
        )

        if res.success:
            u_seq = res.x
            steer = float(u_seq[0])
            self.last_u_seq = u_seq
        else:
            steer = 0.0

        # log
        z0 = future_zetas[0]
        state_now = {
            "ay_hist": self.ay_hist,
            "steer_hist": self.steer_hist
        }

        ay_pred = predict_next_lataccel(state_now, steer, z0, self.model)

        with open("mpc_log.txt", "a") as f:
            f.write(f"{self.step_idx},{current_lataccel},{target_lataccel},{steer},{ay_pred}\n")

        # update steer history
        self.steer_hist.append(steer)
        self.steer_hist = self.steer_hist[-self.model["NUM_STEER_TERMS"]:]

        self.step_idx += 1

        return steer