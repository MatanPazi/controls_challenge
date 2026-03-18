"""
generate_csv.py - Uses modified TinyPhysicsSimulator with logging
"""

from pathlib import Path
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, run_rollout, CONTEXT_LENGTH
import pandas as pd

OUTPUT_DIR = Path("data_excitation")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = "models/tinyphysics.onnx"
MAX_STEPS = 600

# Base CSVs to use. Automatically take first N files from data/ folder
N_BASES = 50
all_csvs = sorted(Path("data").glob("*.csv"))  # finds 00000.csv, 00001.csv, …
BASE_CSVS = [str(p) for p in all_csvs[:N_BASES]]

maneuvers = [
    ("custom_const",        "const_06"),
    ("custom_step",         "step"),
    ("custom_chirp",        "chirp"),   
    ("custom_ramp",         "ramp"),   
    ("custom_sine",         "sine"),   
]

def main():
    print(f"Generating excitation for {len(BASE_CSVS)} base CSVs × {len(maneuvers)} maneuvers\n")

    for base_path in BASE_CSVS:
        base_name = Path(base_path).stem  # e.g. "00000"
        print(f"Processing base: {base_name}")

        for ctrl_type, maneuver_name in maneuvers:
            out_filename = f"{base_name}_excitation_{maneuver_name}.csv"
            out_path = OUTPUT_DIR / out_filename

            print(f"  → {ctrl_type} → {out_filename}")

            result = run_rollout(
                data_path=base_path,
                controller_type=ctrl_type,
                model_path=MODEL_PATH,
                debug=False
            )

            # Extract the DataFrame from the tuple (adjust key if needed)
            # From your previous debug: tuple[0] = costs dict, tuple[1] = target_hist, tuple[2] = current_hist
            # But since we need full log, we still need to run sim manually for full columns

            # So: create sim ourselves (as before)
            tiny_model = TinyPhysicsModel(model_path=MODEL_PATH, debug=False)
            ctrl_module = __import__(f"controllers.{ctrl_type}", fromlist=[""])
            controller = ctrl_module.Controller()
            sim = TinyPhysicsSimulator(tiny_model, base_path, controller, debug=False)
            sim.enable_logging()  # your added method

            # Run rollout
            for _ in range(CONTEXT_LENGTH, len(sim.data)):
                sim.step()

            df = sim.get_log_df()
            df = df.iloc[:MAX_STEPS]

            df.to_csv(out_path, index=False)
            print(f"     Saved: {out_path} ({len(df)} rows)")

    print("\nAll done! Files ready in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()