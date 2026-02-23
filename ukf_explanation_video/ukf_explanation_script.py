import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
from scipy.stats import norm  # For Gaussian PDF plotting
import sys
import os


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
# Now I can import the desired class
from ukf_observer import UKF

# Load theta once at startup (shared by all UKF instances)
theta_path = os.path.join(parent_dir, 'lpv_arx_theta.npy')
theta = np.load(theta_path)

# Terrain constants
HILL_AMP = 30
WATER_HEIGHT = 4
Y_LIM = 75

# UKF and simulation constants
STATE_DIM = 2           # [position, velocity]
V_SPEED = 1.0           # Constant speed (units per step)
DT = 1.0                # Time step
Q_VAR = [0.02, 0.001]     # Q_diag for pos and vel noise
R_VAR = 0.020             # Measurement noise variance
INITIAL_X = 15.0        # start pos
V_SPEED = 1.0           # nominal vel
INITIAL_P = np.diag([50.0, 5.0])  # separate variances
NUM_STEPS = 80          # Simulation steps (to cross terrain)
TRUE_START_X = 10.0     # True bird starting position

def h(x):
    """
    Physical surface height seen by the bird.
    """
    x = np.asarray(x)
    was_scalar = x.ndim == 0
    if was_scalar:
        x = x[None]

    y_hills = np.interp(x, TERRAIN_X, TERRAIN_Y)
    y_surface = np.maximum(y_hills, WATER_HEIGHT)

    return float(y_surface[0]) if was_scalar else y_surface


def generate_terrain(x):
    """
    Raw terrain height (hills only).
    Water is NOT applied here.
    """
    x = np.asarray(x)

    w1 = 2 * np.pi / 70
    w2 = 2 * np.pi / 23
    w3 = 2 * np.pi / 11

    amp = 15 + 0.15 * x + 3 * np.sin(2 * np.pi * x / 90)
    bias = 2 * np.sin(2 * np.pi * x / 120)

    hills = (
        1.1 * np.sin(w1 * x + 0.3) +
        0.6 * np.sin(w2 * x + 1.1) +
        0.2 * np.sin(w3 * x + 2.4)
    )

    y = amp * hills + bias

    return y


def plot_terrain(show=True, figsize=(12, 6)):
    x = np.linspace(0, 100, 2000)
    y_hills = np.interp(x, TERRAIN_X, TERRAIN_Y, left=0.0, right=0.0)  # raw hills for plotting

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    # Blue water background – full width
    ax.fill_between(x, 0, WATER_HEIGHT, color='#1E90FF', zorder=1)

    # Green hills – overwrites blue where hills exist
    ax.fill_between(x, 0, y_hills, color='#228B22', zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis('off')

    if show:
        plt.tight_layout(pad=0)
        plt.show()

    return fig, ax, x, y_hills


def mark_position(x_pos, fig=None, ax=None, color='red', label=True):
    if fig is None or ax is None:
        fig, ax, x_grid, y_grid = plot_terrain(show=False)
    else:
        try:
            x_grid = ax.get_lines()[0].get_xdata() if ax.get_lines() else np.linspace(0, 100, 2000)
        except:
            x_grid = np.linspace(0, 100, 2000)

    # Use h() for consistency (includes WATER_HEIGHT min)
    terrain_y = h(x_pos)   # scalar
    max_height = terrain_y   # already max(hill, WATER_HEIGHT)

    # Vertical line
    ax.axvline(x=x_pos, ymin=max_height/Y_LIM, ymax=Y_LIM,
               color=color, ls='--', alpha=0.7, zorder=10, lw=1.2)

    # Marker at surface height
    ax.plot(x_pos, max_height, 'o', color=color, ms=10, mec='white', mew=2, zorder=12)

    return fig, ax



def simulate_trajectory():
    true_state = np.zeros((NUM_STEPS, 2))  # [pos, vel]
    measurements = np.zeros(NUM_STEPS)
    
    true_state[0] = [TRUE_START_X, V_SPEED]  # initial pos + vel
    measurements[0] = h(true_state[0, 0]) + np.random.normal(0, np.sqrt(R_VAR))
    
    for k in range(1, NUM_STEPS):
        pos, vel = true_state[k-1]
        
        # Add process noise to both states
        noise = np.random.normal(0, np.sqrt(Q_VAR))  # shape (2,)
        pos_next = pos + vel * DT + noise[0]
        vel_next = vel + noise[1]  # velocity can drift slowly
        
        true_state[k] = [pos_next, vel_next]
        measurements[k] = h(pos_next) + np.random.normal(0, np.sqrt(R_VAR))
    
    true_x = true_state[:, 0]  # only return positions for plotting
    return true_x, measurements

def bird_predict_state(x, u, zeta, theta=None):
    pos, vel = x
    # Simple kinematic model
    pos_next = pos + vel * DT + u  # u can be 0 or acceleration
    vel_next = vel  # constant velocity, or add acceleration if needed
    return np.array([pos_next, vel_next])

def bird_measure_state(x, theta=None):
    """
    Measurement model:
    z = terrain height at x-position
    """
    pos = x[0]
    return np.array([h(pos)])  # must return 1D array


def add_png_bird(ax, png_file, x_pct=0.20, y_pct=0.80, zoom=0.12, flip=True, zorder=30):
    """
    Draw a PNG image (your bird) on the axes at a percentage position.
    flip=True mirrors the PNG horizontally so it faces left.
    zoom controls its size.
    """
    img = plt.imread(png_file)

    # Flip horizontally if requested
    if flip:
        img = np.fliplr(img)

    imagebox = OffsetImage(img, zoom=zoom, zorder=zorder)

    # AnnotationBbox takes (x, y) in DATA coordinates, so convert percentage → data
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_data = x0 + x_pct * (x1 - x0)
    y_data = y0 + y_pct * (y1 - y0)

    ab = AnnotationBbox(
        imagebox,
        (x_data, y_data),
        frameon=False,
        bboxprops=None,
        zorder=zorder
    )
    ax.add_artist(ab)


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────

if __name__ == "__main__":

    # Precompute terrain height at every integer x from 0 to 100 inclusive
    TERRAIN_GRID_POINTS = 5000                     # 5000–10000 is usually plenty
    TERRAIN_X = np.linspace(0, 100, TERRAIN_GRID_POINTS)
    TERRAIN_Y = generate_terrain(TERRAIN_X)        # called only once


# --- Animation Setup ---

fig, ax = plt.subplots(figsize=(12, 6))

# Plot static terrain once
x_plot = np.linspace(0, 100, 2000)
y_plot = np.interp(x_plot, TERRAIN_X, TERRAIN_Y)
ax.fill_between(x_plot, 0, WATER_HEIGHT, color='#1E90FF', zorder=1)
ax.fill_between(x_plot, 0, y_plot, color='#228B22', zorder=2)
ax.set_xlim(0, 100)
ax.set_ylim(0, Y_LIM)
ax.axis('off')

# Dynamic Elements
bird_marker = [] # Container for the AnnotationBbox
est_dot, = ax.plot([], [], 'o', color='red', ms=10, mec='white', mew=2, zorder=12, label='UKF Estimate')
est_line = ax.axvline(0, color='red', ls='--', alpha=0.7, zorder=10, lw=1.2)

# Initialize Simulation State
true_pos = TRUE_START_X
true_vel = V_SPEED

ukf = UKF(n=STATE_DIM, R=R_VAR, Q_diag=Q_VAR, theta=theta, 
          predict_state=bird_predict_state, measure_state=bird_measure_state,
          alpha=1e-3, beta=2.0, kappa=0.0)
ukf.x = np.array([INITIAL_X, V_SPEED])
ukf.P = INITIAL_P.copy()

uncertainty_fill = ax.axvspan(0, 0, color='red', alpha=0.2)
def update(frame):
    global true_pos, true_vel, bird_marker

    # 1. True Bird Movement (Simulate)
    noise = np.random.normal(0, np.sqrt(Q_VAR))
    true_pos += true_vel * DT + noise[0]
    true_vel += noise[1]
    
    # 2. Measurement (Terrain height bird "feels")
    measurement = h(true_pos) + np.random.normal(0, np.sqrt(R_VAR))

    # 3. UKF Update
    ukf.predict(u=0, zeta=None)
    ukf.update(measurement)
    est_pos = ukf.x[0]

    # --- Update Visuals ---

    # Update UKF Marker
    surface_h = h(est_pos)
    est_dot.set_data([est_pos], [surface_h])
    est_line.set_xdata([est_pos])
    # Adjust vertical line height
    est_line.set_ydata([surface_h/Y_LIM, 1.0]) 

    est_pos = ukf.x[0]
    std_dev = np.sqrt(ukf.P[0,0])
    
    # Update the "Confidence Cloud"
    # Show 2 standard deviations (95% confidence)
    left_bound = est_pos - 2 * std_dev
    right_bound = est_pos + 2 * std_dev
    
    global uncertainty_fill
    uncertainty_fill.remove() # Clear old fill
    uncertainty_fill = ax.axvspan(left_bound, right_bound, color='red', alpha=0.15)    

    # Update Bird PNG
    if bird_marker:
        bird_marker[0].remove()
        bird_marker.clear()

    img = plt.imread('ukf_explanation_video/Stork_silhouette.png')
    img = np.fliplr(img) # Face forward
    imagebox = OffsetImage(img, zoom=0.04)
    # The bird flies at a fixed altitude (e.g., 60)
    ab = AnnotationBbox(imagebox, (true_pos, 60), frameon=False, zorder=30)
    ax.add_artist(ab)
    bird_marker.append(ab)

    return est_dot, est_line

# Start Animation
ani = FuncAnimation(fig, update, frames=NUM_STEPS, interval=100, blit=False)

plt.tight_layout(pad=0)
plt.show()
