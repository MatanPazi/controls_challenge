"""
test_delay.py - Sample Delay Analysis for TinyPhysics Model

This script measures the effective sample delay in the TinyPhysics ONNX model 
by comparing lateral acceleration trajectories between a zero-steer controller 
and a constant step-steer controller on real driving segments.

It computes how many simulation steps pass after control start (CONTROL_START_IDX) 
before the step response noticeably diverges from the zero-steer baseline.

Requirements:
- tinyphysics.py and its dependencies in the same directory
- ./models/tinyphysics.onnx
- ./data/*.csv (comma-steering-control style segments)
- A 'step' controller defined in controllers/ (constant steer output)

"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from tinyphysics import run_rollout, CONTROL_START_IDX

# ──────────────────────────────────────────────
# Config
MODEL_PATH    = "./models/tinyphysics.onnx"
DATA_DIR      = Path("./data")
NUM_SEGMENTS  = 100                     # how many routes to analyze for stats
STEP_VALUE    = 0.5
THRESHOLD     = 0.05                   # m/s² — tune if needed

# Which single files to show nice overlay plot for. Add routes of interest
SHOW_OVERLAY_FOR = [
    # "00000.csv",
    # "00001.csv",
    "00010.csv",
    # "00123.csv",
]

# ──────────────────────────────────────────────

def detect_delay(zero_lat, step_lat, control_start=CONTROL_START_IDX):
    """
    Detect the number of samples after control_start until the step controller's 
    lateral acceleration differs noticeably from the zero controller.

    Args:
        zero_lat (list or np.array): Lateral accel history from zero controller
        step_lat (list or np.array): Lateral accel history from step controller
        control_start (int): Index where active control begins (default: 100)

    Returns:
        int or None: Number of samples of delay, or None if no clear divergence
                     is detected within the threshold.
    """    
    zero_lat = np.array(zero_lat)
    step_lat = np.array(step_lat)
    diff = np.abs(step_lat - zero_lat)
    response_idxs = np.where(diff[control_start:] > THRESHOLD)[0]
    if len(response_idxs) > 0:
        return response_idxs[0]   # samples after control_start
    return None                   # no clear response


def plot_overlay(zero_lat, step_lat, filename, delay_samples=None):
    """
    Create a single overlaid plot comparing zero-steer vs step-steer lateral 
    acceleration trajectories, zoomed around control start.

    Args:
        zero_lat (list or np.array): Zero-controller lateral accel history
        step_lat (list or np.array): Step-controller lateral accel history
        filename (str): Name of the data file (for title)
        delay_samples (int or None): Detected delay to highlight (optional)
    """    
    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.arange(len(zero_lat))

    # Lines with small dots at every sample
    ax.plot(t, zero_lat, label="zero", color="C0", lw=1.2, alpha=0.9,
            marker='.', markersize=10, markevery=1,
            markerfacecolor='C0', markeredgewidth=0)

    ax.plot(t, step_lat, label=f"step = {STEP_VALUE}", color="C3", lw=1.5,
            marker='.', markersize=10, markevery=1,
            markerfacecolor='C3', markeredgewidth=0)

    ax.axvline(CONTROL_START_IDX, color="k", ls="--", lw=1.3, alpha=0.75, label="control start")

    if delay_samples is not None:
        resp_idx = CONTROL_START_IDX + delay_samples
        ax.axvline(resp_idx, color="darkgreen", ls=":", lw=2.2, alpha=0.85,
                   label=f"first response (delay {delay_samples})")

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Lateral accel [m/s²]")
    ax.set_title(f"Zero vs Step Response – {filename}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    # Zoom
    zoom_start = max(0, CONTROL_START_IDX - 20)
    zoom_end   = min(len(t), CONTROL_START_IDX + 20)
    ax.set_xlim(zoom_start, zoom_end)

    # Minor x-ticks + faint vertical grid for counting
    ax.set_xticks(np.arange(zoom_start, zoom_end + 1, 1), minor=True)
    ax.tick_params(axis='x', which='minor', length=3, width=0.7, color='gray')
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4, alpha=0.35, color='lightgray')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main execution flow of the delay analysis script.

    Behavior:
    1. Finds and processes up to NUM_SEGMENTS CSV files in ./data/
    2. Runs zero and step controllers on each segment
    3. Computes delay for each
    4. Prints per-file results
    5. Optionally shows detailed overlay plots for selected files
    6. Prints summary table and generates histogram of delay distribution

    Configuration is done via global constants at the top of the file.
    """    
    csv_files = sorted(DATA_DIR.glob("*.csv"))[:NUM_SEGMENTS]
    if not csv_files:
        print("No csv files found in", DATA_DIR)
        return

    print(f"Analyzing {len(csv_files)} segments for delay distribution...\n")

    delays = []
    for csv_path in csv_files:
        name = csv_path.name

        zero_hist  = run_rollout(str(csv_path), "zero", MODEL_PATH)[2]
        step_hist  = run_rollout(str(csv_path), "step", MODEL_PATH)[2]

        delay = detect_delay(zero_hist, step_hist)

        if delay is not None:
            delays.append(delay)
            print(f"{name:12} → delay = {delay} samples")
        else:
            print(f"{name:12} → no clear response")

        # Optional: show overlay for selected files
        if name in SHOW_OVERLAY_FOR:
            print(f"   → showing overlay plot for {name}")
            plot_overlay(zero_hist, step_hist, name, delay)

    if not delays:
        print("\nNo delays detected on any segment.")
        return

    # ─── Summary & Histogram ────────────────────────────────────
    counts = Counter(delays)
    total = len(delays)
    min_d, max_d = min(delays), max(delays)

    print("\n" + "="*50)
    print(f"Delay distribution over {total} routes (of {NUM_SEGMENTS} attempted):")
    print("samples | count | percent | bar")
    print("-"*40)
    for d in range(min_d - 1, max_d + 2):
        c = counts.get(d, 0)
        pct = 100 * c / total if total > 0 else 0
        bar = "█" * int(round(pct / 3.5))   # scale to reasonable width
        print(f"{d:7} | {c:5} | {pct:6.1f}% | {bar}")

    print(f"\nMean   = {np.mean(delays):.2f} samples")
    print(f"Median = {np.median(delays):.0f}")
    print(f"Std    = {np.std(delays):.2f}")
    print(f"Most common = {counts.most_common(1)[0][0]} ({counts.most_common(1)[0][1]}×)")

    # Full histogram plot
    plt.figure(figsize=(9, 5))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    plt.hist(delays, bins=bins, edgecolor='black', align='mid')
    plt.title("Sample Delay Distribution (after control start)")
    plt.xlabel("Delay [samples]")
    plt.ylabel("Number of routes")
    plt.xticks(range(min_d, max_d + 2))
    plt.grid(axis='y', alpha=0.3)

    # Count labels on bars
    n, bins_out, _ = plt.hist(delays, bins=bins)  # re-get counts
    for i, count in enumerate(n):
        if count > 0:
            plt.text(bins_out[i] + 0.5, count + 0.2, str(int(count)),
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()