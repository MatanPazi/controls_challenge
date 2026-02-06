# Script function:
# Creates a histogram showing the sample delay in each of the selected routes.

# Result:
# There seems to be a decaying function of sample delay. Most routes have 0 delay, then it decays to very little routes have a sample delay of 4.
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
from io import BytesIO

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, download_dataset, CONTROL_START_IDX

class ZeroController:
    def __init__(self):
        pass

    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        return 0.0

class StepController:
    def __init__(self):
        self.const_value = 1.0  # Hardcoded constant value

    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        return self.const_value

# Hardcoded inputs
model_path = './models/tinyphysics.onnx'
data_dir = Path('./data')
num_routes = 100
epsilon = 0.001

# Download dataset if not present
if not data_dir.exists():
    url = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
    print("Downloading dataset...")
    with urllib.request.urlopen(url) as f:
        zip_data = BytesIO(f.read())
    with zipfile.ZipFile(zip_data) as z:
        z.extractall(data_dir.parent)
    print("Dataset downloaded and extracted.")

# Get list of CSV paths (first 100)
csv_paths = sorted(data_dir.glob('*.csv'))[:num_routes]

delays = []

for csv_path in csv_paths:
    # Zero controller simulation
    model = TinyPhysicsModel(model_path, debug=False)
    zero_controller = ZeroController()
    sim_zero = TinyPhysicsSimulator(model, str(csv_path), zero_controller, debug=False)
    sim_zero.rollout()
    zero_hist = np.array(sim_zero.current_lataccel_history)

    # Step controller simulation
    step_controller = StepController()
    sim_step = TinyPhysicsSimulator(model, str(csv_path), step_controller, debug=False)
    sim_step.rollout()
    step_hist = np.array(sim_step.current_lataccel_history)

    # Ensure histories are the same length
    assert len(zero_hist) == len(step_hist), "History lengths do not match"

    # Compute differences starting from CONTROL_START_IDX
    diff = np.abs(zero_hist - step_hist)
    post_control_diff = diff[CONTROL_START_IDX:]

    if np.any(post_control_diff > epsilon):
        diverge_idx = np.where(post_control_diff > epsilon)[0][0] + CONTROL_START_IDX
        delay = diverge_idx - CONTROL_START_IDX
        delays.append(delay)
    else:
        print(f"No divergence found for route {csv_path.name}. Skipping.")
        # Optionally, handle cases with no divergence

# Create histogram
if delays:
    plt.figure(figsize=(10, 6))
    plt.hist(delays, bins=20, edgecolor='black')
    plt.xlabel('Delay (steps)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Delays across 100 Routes')
    plt.show()
else:
    print("No delays found across all routes.")