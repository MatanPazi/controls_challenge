import subprocess
import re
import json
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# ========================= CONFIG =========================
NUM_SEGS = 1          # increase to 50–100 for final tuning (slower but more accurate)
INITIAL_GAINS = [0.195, 0.100, -0.053]
TOL = 1e-4
MAXITER = 150
# ========================================================

GAINS_FILE = Path("pid_gains.json").resolve()
MODEL_PATH = Path("models/tinyphysics.onnx")
DATA_DIR   = Path("data")
FIXED_FILE = "00000.csv"                   # change to whichever route you want

def objective(gains: np.ndarray) -> float:
    p, i, d = gains
    # Write current gains
    with open(GAINS_FILE, "w") as f:
        json.dump({"p": float(p), "i": float(i), "d": float(d)}, f, indent=2)

    # Build correct path to the single file
    single_data_path = DATA_DIR / FIXED_FILE

    cmd = [
        "python", "tinyphysics.py",
        "--model_path", str(MODEL_PATH),
        "--data_path", str(single_data_path),     # points to one .csv file
        "--num_segs", str(NUM_SEGS),
        "--controller", "tunable_pid"
    ]

    print(f"→ Testing gains P={p:.4f} I={i:.4f} D={d:.4f} ...", end=" ")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    # Parse the exact line printed by tinyphysics.py
    match = re.search(r"average total_cost:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", result.stdout, re.IGNORECASE)
    if match:
        cost = float(match.group(1))
        print(f"total_cost = {cost:.4f}")
        return cost
    else:
        print("ERROR parsing output")
        print(result.stdout[-500:])  # debug
        return 1e6  # big penalty


if __name__ == "__main__":
    if not GAINS_FILE.exists():
        with open(GAINS_FILE, "w") as f:
            json.dump({"p": INITIAL_GAINS[0], "i": INITIAL_GAINS[1], "d": INITIAL_GAINS[2]}, f, indent=2)

    print("Starting PID gain optimization with Nelder-Mead...")
    print(f"Objective = average total_cost over {NUM_SEGS} routes\n")

    res = minimize(
        objective,
        x0=np.array(INITIAL_GAINS),
        method="Nelder-Mead",
        tol=TOL,
        options={"maxiter": MAXITER, "disp": True}
    )

    best_p, best_i, best_d = res.x
    best_cost = res.fun

    print("\n" + "="*60)
    print("OPTIMIZATION FINISHED")
    print(f"Best gains  → P = {best_p:.6f}   I = {best_i:.6f}   D = {best_d:.6f}")
    print(f"Best average total_cost = {best_cost:.4f}")
    print("="*60)

    # Final save
    with open(GAINS_FILE, "w") as f:
        json.dump({"p": float(best_p), "i": float(best_i), "d": float(best_d)}, f, indent=2)