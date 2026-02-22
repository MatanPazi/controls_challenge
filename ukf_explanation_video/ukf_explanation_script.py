import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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

# Terrain constants
HILL_AMP = 30
WATER_HEIGHT = 4
Y_LIM = 75

# UKF and simulation constants
STATE_DIM = 1           # 1D position x
V_SPEED = 1.0           # Constant speed (units per step)
DT = 1.0                # Time step
Q_VAR = 0.5             # Process noise variance
R_VAR = 2.0             # Measurement noise variance
INITIAL_X = 50.0        # Initial UKF mean (uncertain)
INITIAL_P = 10000.0     # Large initial covariance
NUM_STEPS = 80          # Simulation steps (to cross terrain)
TRUE_START_X = 10.0     # True bird starting position

def generate_terrain(x):
    """
    Compute terrain height y(x) — same logic as before.
    Returns array of heights (green hills), zero elsewhere.
    """
    period = 66
    omega = 2 * np.pi / period

    y_template = (
        1.00 * np.sin(1 * omega * x) +
        0.40 * np.sin(5 * omega * x) +
        0.25 * np.sin(7 * omega * x)
    )

    mask_left = (x <= 33)
    y_left = y_template.copy()
    y_left[~mask_left] = 0.0

    min_val = np.min(y_left[mask_left])
    max_val = np.max(y_left[mask_left])
    y_left_scaled = HILL_AMP * (y_left - min_val) / (max_val - min_val)

    # Mirror right side
    y_right_scaled = np.zeros_like(x)
    left_len = np.sum(mask_left)
    y_right_scaled[-left_len:] = y_left_scaled[:left_len][::-1]

    return y_left_scaled + y_right_scaled


def plot_terrain(show=True, figsize=(12, 6)):
    """
    Creates the base terrain plot (green hills + blue bottom strip).
    Returns fig and ax so you can add more elements.
    """
    x = np.linspace(0, 100, 2000)
    y = generate_terrain(x)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    # Blue rectangle across the bottom (background)
    ax.fill_between(x, 0, WATER_HEIGHT, color='#1E90FF', zorder=1)

    # Green hills on top
    ax.fill_between(x, 0, y, where=(y > 0), color='#228B22', zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis('off')

    if show:
        plt.tight_layout(pad=0)
        plt.show()

    return fig, ax, x, y


def mark_position(x_pos, fig=None, ax=None, color='red', label=True):
    """
    Marks a specific x position on an existing terrain plot:
      - vertical line at x_pos
      - dot at terrain height y(x_pos)
      - dot at blue rectangle top
      - optional text labels with the values
    """
    if fig is None or ax is None:
        fig, ax, x_grid, y_grid = plot_terrain(show=False)
    else:
        # Reuse existing grid if available, otherwise recompute
        try:
            x_grid = ax.get_lines()[0].get_xdata() if ax.get_lines() else np.linspace(0, 100, 2000)
            y_grid = generate_terrain(x_grid)
        except:
            x_grid = np.linspace(0, 100, 2000)
            y_grid = generate_terrain(x_grid)

    # Find closest point
    idx = np.argmin(np.abs(x_grid - x_pos))
    x_val = x_grid[idx]
    terrain_y = y_grid[idx]
    max_height = max(terrain_y, WATER_HEIGHT)

    # Vertical line
    ax.axvline(x=x_val, ymin = max_height/Y_LIM, ymax = Y_LIM, color=color, ls='--', alpha=0.7, zorder=10, lw=1.2)

    # Marker at terrain height
    ax.plot(x_val, max_height, 'o', color=color, ms=10, mec='white', mew=2, zorder=12)

    return fig, ax




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

    # 1. Just show the base terrain
    # plot_terrain()

    # 2. Or create base plot and mark one or more positions
    fig, ax, _, _ = plot_terrain(show=False)

    # # Mark a few example positions
    mark_position(15, fig, ax, color='red')
    mark_position(28, fig, ax, color='darkorange')
    # mark_position(75, fig, ax, color='purple')
    # mark_position(50, fig, ax, color='black')  # middle gap

    
    # Add a bird at 20% along x and 80% of y-range
    add_png_bird(
        ax,
        png_file='ukf_explanation_video\Stork_silhouette.png',  # <-- your saved PNG
        x_pct=0.20,
        y_pct=0.80,
        zoom=0.03,      # Adjust size to taste
        flip=True       # Flip horizontally so it flies along -x
    )

    plt.tight_layout(pad=0)
    plt.show()
