from . import BaseController
import numpy as np
import osqp
from scipy import sparse

# ============================
# LPV basis
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
# Controller
# ============================

class Controller(BaseController):
    def __init__(self):
        # Load model parameters
        data = np.load("lpv_arx_model.npz", allow_pickle=True)
        self.model = {
            "theta": data["theta"],
            "NA": int(data["NA"]),
            "NB": int(data["NUM_STEER_TERMS"]),
            "BD": int(data["BASIS_DIM"]),
            "EXO_VARS": list(data["EXO_VARS"])
        }

        # MPC Hyperparameters
        self.N = 14           # Horizon steps
        self.weight_y = 12.0    # Tracking error weight
        self.weight_u = 0.0    # Absolute steering magnitude penalty
        self.weight_du = 25.0  # Slew rate penalty (CRITICAL for stability)

        # Histories
        self.ay_hist = [0.0] * self.model["NA"]
        self.steer_hist = [0.0] * self.model["NB"]
        
        self.step_idx = 0
        self.last_u_seq = None

    def predict_step(self, ay_h, steer_val, steer_h, zeta):
        """Pure function to predict next ay based on provided history buffers."""
        theta = self.model["theta"]
        NA, NB, BD = self.model["NA"], self.model["NB"], self.model["BD"]
        basis = build_basis(zeta["vEgo"], BD)
        
        col = 0
        pred = 0.0

        # AR terms (past ay)
        for i in range(NA):
            pred += np.dot(basis, theta[col:col+BD]) * ay_h[-(i+1)]
            col += BD

        # Steering terms (Current + Past)
        for d in range(NB):
            u = steer_val if d == 0 else steer_h[-d]
            pred += np.dot(basis, theta[col:col+BD]) * u
            col += BD

        # Exogenous terms
        for name in self.model["EXO_VARS"]:
            val = zeta.get(name, 0.0)
            pred += np.dot(basis, theta[col:col+BD]) * val
            col += BD
            
        return pred

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # 1. Update internal history with true measurement
        self.ay_hist.append(current_lataccel)
        self.ay_hist = self.ay_hist[-self.model["NA"]:]

        # 2. Prepare Horizon data
        v = np.asarray(future_plan.v_ego)
        a = np.asarray(future_plan.a_ego)
        roll = np.asarray(future_plan.roll_lataccel)
        y_ref = np.asarray(future_plan.lataccel)
        
        N = min(self.N, len(v), len(y_ref))
        if N < 2: return 0.0

        zetas = []
        for i in range(N):
            zetas.append({
                "vEgo": v[i],
                "aEgo": a[i] if len(a) > i else 0.0,
                "roll": roll[i] if len(roll) > i else 0.0
            })

        # 3. Build Free Response (f) and Control Jacobian (G)
        # Y = G*U + f
        f = np.zeros(N)
        G = np.zeros((N, N))

        # Build 'f' (Free Response where all future U = 0)
        curr_ay_h = list(self.ay_hist)
        curr_u_h = list(self.steer_hist)
        for k in range(N):
            pred_f = self.predict_step(curr_ay_h, 0.0, curr_u_h, zetas[k])
            f[k] = pred_f
            curr_ay_h.append(pred_f)
            curr_u_h.append(0.0)
            curr_ay_h.pop(0)
            curr_u_h.pop(0)

        # Build 'G' (Step Response)
        # We calculate how a unit change in U at time 'j' affects Y at time 'k'
        for j in range(N):
            curr_ay_h = list(self.ay_hist)
            curr_u_h = list(self.steer_hist)
            for k in range(N):
                u_in = 1.0 if k == j else 0.0
                pred_y = self.predict_step(curr_ay_h, u_in, curr_u_h, zetas[k])
                
                # The effect is the difference from the free response
                G[k, j] = pred_y - f[k]
                
                # Propagate this specific impulse through the horizon
                curr_ay_h.append(pred_y)
                curr_u_h.append(u_in)
                curr_ay_h.pop(0)
                curr_u_h.pop(0)

        # 4. Formulate QP Matrices
        # Cost = (G*u + f - y_ref)^2 + weight_u*u^2 + weight_du*(D*u - d_prev)^2
        
        # Difference matrix D for slew rate (u_k - u_{k-1})
        D = sparse.eye(N) - sparse.eye(N, k=-1)
        d_prev = np.zeros(N)
        d_prev[0] = self.steer_hist[-1] # Initial condition for slew rate

        P = 2 * (G.T @ G * self.weight_y + 
                 sparse.eye(N) * self.weight_u + 
                 D.T @ D * self.weight_du)
        
        # q vector
        q = 2 * (G.T @ (f - y_ref[:N]) * self.weight_y - 
                 (D.T @ d_prev) * self.weight_du)

        # 5. Solve QP
        prob = osqp.OSQP()
        A_con = sparse.eye(N)
        lb = np.full(N, -1.0)
        ub = np.full(N, 1.0)
        
        prob.setup(sparse.csc_matrix(P), q, A_con, lb, ub, verbose=False, warm_start=True)
        
        # Warm start from last solution
        if self.last_u_seq is not None and len(self.last_u_seq) == N:
            prob.warm_start(x=self.last_u_seq)
            
        res = prob.solve()

        if res.info.status == 'solved':
            self.last_u_seq = res.x
            steer = float(res.x[0])
        else:
            steer = self.steer_hist[-1] # Fallback to previous

        # 6. Update history and return
        self.steer_hist.append(steer)
        self.steer_hist = self.steer_hist[-self.model["NB"]:]
        self.step_idx += 1
        
        return steer
