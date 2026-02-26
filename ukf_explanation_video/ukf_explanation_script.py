import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
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
STATE_DIM = 2           # [position, velocity]
V_SPEED = 1.0           # Constant speed (units per step)
DT = 1                # Time step
Q_VAR = [0.02, 0.015]     # Q_diag for pos and vel noise
R_VAR = np.diag([0.02, 0.1]) # Measurement noise variance
INITIAL_X = 15.0        # start pos
V_SPEED = 1.0           # nominal vel
INITIAL_P = np.diag([60.0, 0.5])  # separate variances
NUM_STEPS = 80          # Simulation steps (to cross terrain)
TRUE_START_X = 5.0     # True bird starting position
BIRD_HEIGHT = 55
MEAS_RULER_X = 50       # fixed x for measurement space ruler


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

def dhdx(x, eps=1e-2):
    """
    Numerical derivative of terrain height wrt x.
    Measures local observability strength.
    """
    return (h(x + eps) - h(x - eps)) / (2 * eps)    


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


def bird_predict_state(x, u, zeta, theta=None):
    pos, vel = x
    # Simple kinematic model
    pos_next = pos + vel * DT + u  # u can be 0 or acceleration
    vel_next = vel  # constant velocity, or add acceleration if needed
    return np.array([pos_next, vel_next])

def bird_measure_state(x, theta=None):
    pos, vel = x
    height   = h(pos)
    velocity = vel
    return np.array([height, velocity])


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

if __name__ == "__main__":

    # Precompute terrain
    TERRAIN_GRID_POINTS = 5000
    TERRAIN_X = np.linspace(0, 100, TERRAIN_GRID_POINTS)
    TERRAIN_Y = generate_terrain(TERRAIN_X)

    # ── Figure & Axis ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x_plot = np.linspace(0, 100, 2000)
    y_plot = np.interp(x_plot, TERRAIN_X, TERRAIN_Y)
    ax.fill_between(x_plot, 0, WATER_HEIGHT, color='#1E90FF', zorder=1)
    ax.fill_between(x_plot, 0, y_plot, color='#228B22', zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, Y_LIM)
    ax.axis('off')

    # ── Artists ──
    bird_marker = []
    est_dot, = ax.plot([], [], 'o', color='red', ms=10, mec='white', mew=2, zorder=18, label='Posterior Estimate')
    est_line = ax.axvline(0, color='red', ls='--', alpha=0.85, zorder=17, lw=1.5)

    pred_dot, = ax.plot([], [], 'o', color='purple', ms=9, mec='white', mew=1.5, zorder=16, label='Predicted Estimate')
    pred_line = ax.axvline(0, color='purple', ls=':', alpha=0.7, zorder=15, lw=1.2)

    sigma_dots = ax.scatter([], [], s=20, c='gray', alpha=0.65, zorder=14)
    projection_lines = []

    meas_dot, = ax.plot([], [], 'o', color='blue', ms=9, zorder=19, label='Measurement')
    meas_proj_line, = ax.plot([], [], 'b--', lw=0.9, alpha=0.55, zorder=10)

    pred_meas_dot, = ax.plot([], [], 'o', color='magenta', ms=9, zorder=19, label='Predicted Meas')

    innovation_lines = []

    # Measurement space ruler (vertical line + labels)
    ruler_line, = ax.plot([], [], 'k--', lw=0.5, alpha=0.6, zorder=10, label='Meas Space')
    ruler_sigma = ax.scatter([], [], s=15, c='gray', alpha=0.6, zorder=14, label='Z σ points')
    ruler_zhat_dot, = ax.plot([], [], 'kx', ms=8, mec='black', mew=1, zorder=19, label='ẑ')
    # Measurement uncertainty error bar
    ruler_s_err = ax.errorbar(
        [], [], yerr=[],
        fmt='none',
        ecolor='purple',
        elinewidth=2.0,
        capsize=6,
        alpha=0.6,
        zorder=11
    )

    ruler_label = ax.text(MEAS_RULER_X + 0.5, Y_LIM - 5, '', fontsize=8, ha='left', va='top', color='gray', alpha=0.8, zorder=25)    

    # High zorder + solid background annotation
    annotation = ax.text(0.03, 0.97, '', transform=ax.transAxes, va='top', ha='left',
                         fontsize=11, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.6',
                                                alpha=0.94), zorder=25)

    uncertainty_fill = ax.axvspan(0, 0, color='red', alpha=0.16, zorder=5)

    # Initial state
    true_pos = TRUE_START_X
    true_vel = V_SPEED

    ukf = UKF(n=STATE_DIM, R=R_VAR, Q_diag=Q_VAR,
              predict_state=bird_predict_state, measure_state=bird_measure_state,
              alpha=0.8, beta=2.0, kappa=0.0)
    ukf.x = np.array([INITIAL_X, V_SPEED])
    ukf.P = INITIAL_P.copy()

    prior_x = ukf.x.copy()
    prior_P = ukf.P.copy()
    predicted_x = ukf.x.copy()
    predicted_P = ukf.P.copy()

    est_pos = ukf.x[0]
    surface_h = h(est_pos)
    est_dot.set_data([est_pos], [surface_h])
    est_line.set_data([est_pos], [surface_h / Y_LIM, 1.0])

    # Initialize active state for manual stepping
    current_x = ukf.x.copy()
    current_P = ukf.P.copy()    

    img = plt.imread('ukf_explanation_video/Stork_silhouette.png')
    img = np.fliplr(img)
    imagebox = OffsetImage(img, zoom=0.03)
    ab = AnnotationBbox(imagebox, (true_pos, BIRD_HEIGHT), frameon=False, zorder=30)
    ax.add_artist(ab)
    bird_marker.append(ab)    

    base_text = f"Initial Est: {est_pos:.2f} | Initial True: {true_pos:.2f}"
    annotation.set_text(base_text)

    sub_steps = 8  # Extra step for Kalman gain visualization
    # _first_call = True
    current_frame = 0

    def update(frame):
        # global _first_call
        global true_pos, true_vel, bird_marker, prior_x, prior_P, predicted_x, predicted_P, uncertainty_fill, measurement
        global est_pos, surface_h, current_x, current_P
        global ruler_s_err

        # if _first_call:
        #     print("Initialization pass (frame=0)")
        #     _first_call = False
        #     return        

        # ── Clear Innovation lines ──
        for ln in innovation_lines:
            ln.remove()
        innovation_lines.clear()

        k = frame // sub_steps
        sub = frame % sub_steps

        # ── Which estimate is active? ──
        show_posterior = sub >= 7
        show_predicted  = 2 <= sub <= 7

        if sub == 0:
            noise = np.random.normal(0, np.sqrt(Q_VAR))
            true_pos += true_vel * DT + noise[0]
            true_vel += noise[1]
            true_vel = np.clip(true_vel, 0.8, 1.2)

            true_height = h(true_pos)
            true_velocity = true_vel
            measurement = np.array([true_height, true_velocity]) + \
                          np.random.multivariate_normal([0, 0], R_VAR)

            prior_x = ukf.x.copy()
            prior_P = ukf.P.copy()
            ukf.predict(u=0, zeta=None)
            predicted_x = ukf.x.copy()
            predicted_P = ukf.P.copy()
            ukf.update(measurement)            
            est_pos = ukf.x[0]
            surface_h = h(est_pos)
            current_x = ukf.x if show_posterior else predicted_x if show_predicted else prior_x
            current_P = ukf.P if show_posterior else predicted_P if show_predicted else prior_P            

        # Posterior (red) ─ only moves in last step
        if show_posterior:
            est_dot.set_data([est_pos], [surface_h])
            est_line.set_data([est_pos], [surface_h / Y_LIM, 1.0])

        # Predicted (purple) ─ visible during prediction & innovation
        if show_predicted:
            pred_pos = predicted_x[0]
            pred_h = h(pred_pos)
            pred_dot.set_data([pred_pos], [pred_h])
            pred_line.set_data([pred_pos], [pred_h / Y_LIM, 1.0])
        else:
            pred_dot.set_data([], [])
            pred_line.set_data([], [])

        # Uncertainty follows the active estimate
        if sub >= 1:
            std_dev = np.sqrt(current_P[0, 0])
            left = current_x[0] - 2 * std_dev
            right = current_x[0] + 2 * std_dev
            uncertainty_fill.set_xy([
                [left,   0],
                [right,  0],
                [right,  Y_LIM],
                [left,   Y_LIM],
                [left,   0]
            ])

        # ── Sub-step text & sigma ──
        sigma = None
        sigma_label = ""

        if sub in [0, 1]:
            base_text = f"Step {k}: Prior\nEst: {prior_x[0]:.2f} | True: {true_pos:.2f}"
            if sub == 1:
                sigma = ukf.last_prior_sigma
                sigma_label = " (prior σ)\n(σ points: orange = position, teal = velocity, grey = mean)"
        elif sub in [2, 3]:
            base_text = f"Step {k}: Predicted\nEst: {predicted_x[0]:.2f} | True: {true_pos:.2f}"
            if sub == 2:
                sigma = ukf.last_pred_sigma
                sigma_label = " (predicted σ)\n(σ points: orange = position, teal = velocity, grey = mean)"
            elif sub == 3:
                sigma = ukf.last_meas_sigma
                sigma_label = " (meas-mapped σ)\n(σ points: orange = position, teal = velocity, grey = mean)"
        elif sub == 4:
            base_text = (
                f"Step {k}: S = {ukf.last_S[0,0]:.2f} | Pxz = {ukf.last_Pxz[0,0]:.2f}"
                "\n"
                "S - Certainty in predicted measurements | Pxz - x uncertainty to measurement uncertainty correlation"
            )
        elif sub == 5:
            base_text = (
                f"Step {k}: Innovation = {ukf.last_innovation[0]:.2f}"
                "\n"
                "Innovation - Actual measurement minus predicted measurement"
            )
        elif sub == 6:  # Kalman Gain / Trust step
            if hasattr(ukf, 'last_K') and ukf.last_K is not None:
                gain = ukf.last_K[0,0]
                trust_pred = 1 - gain
                base_text = (f"Step {k}: Apply Kalman Gain\n"
                             f"Gain: {gain:.2f}\n"
                             f"Trust Pred Meas: {trust_pred:.2f} | Trust Meas: {gain:.2f}")
            else:
                base_text = f"Step {k}: Apply Kalman Gain\n(Gain not saved)"
        elif sub == 7:
            dx_corr = ukf.last_K @ ukf.last_innovation

            base_text = (
                rf"Step {k}: Posterior"
                "\n"
                r"$x_{\mathrm{post}}"
                r" = x_{\mathrm{pred}}"
                r" + K\,(y_{\mathrm{meas}} - \hat z_{\mathrm{pred\ meas}})$"
                "\n"
                rf"$x_{{\mathrm{{post}}}}"
                rf" = {predicted_x[0]:.2f}"
                rf" + ({dx_corr[0]:+.2f})"
                rf" = {est_pos:.2f}$"
            )
        else:
            base_text = f"Step {k}: Done"

        annotation.set_text(base_text + sigma_label)

        # ── Sigma points & Measurement ruler ──
        for line in projection_lines:
            line.remove()
        projection_lines.clear()

        if sigma is not None:
            # Mask: keep only center + position sigmas (indices 0,1,2)
            mask = [True, True, False, True, False]  # exclude velocity ±
            sigma_x_masked = sigma[mask, 0]
            sigma_y_masked = np.full_like(sigma_x_masked, BIRD_HEIGHT)

            # Always compute state-space sizes & colors (consistent logic)
            abs_w = np.abs(ukf.Wm)
            w_norm = (abs_w - abs_w.min()) / (abs_w.max() - abs_w.min() + 1e-8)
            sizes = 30 + 100 * w_norm
            colors_masked  = ['gray'] * len(sigma_x_masked)
            colors_masked[1] = colors_masked[2] = 'orange'   # position axis
            sigma_dots.set_sizes(sizes)
            sigma_dots.set_facecolor(colors_masked)            

            if sub == 3 and hasattr(ukf, 'last_Z') and ukf.last_Z is not None:
                # Switch to measurement space (ruler mode)
                Z = ukf.last_Z
                z_heights = Z[:,0]  # height component
                z_heights_masked = z_heights[mask]
                ruler_x = np.full_like(z_heights_masked, MEAS_RULER_X)

                sigma_dots.set_offsets(np.column_stack((ruler_x, z_heights_masked)))

                # Horizontal dashed lines: from ruler Z height back to original sigma x
                for i in range(len(sigma_x_masked)):
                    z_h = z_heights_masked[i]
                    ln, = ax.plot([sigma_x_masked[i], MEAS_RULER_X], [z_h, z_h], 'k--', lw=0.6, alpha=0.3, zorder=8)
                    projection_lines.append(ln)

                # Show z_hat on the ruler
                z_hat_h = ukf.last_z_hat[0]
                ruler_zhat_dot.set_data([MEAS_RULER_X], [z_hat_h])
                ln, = ax.plot([sigma_x_masked[0], MEAS_RULER_X], [z_hat_h, z_hat_h], 'k--', lw=0.6, alpha=0.3, zorder=8)
                projection_lines.append(ln)

                # S band
                std_S = np.sqrt(ukf.last_S[0, 0])
                # Remove old error bar artists
                ruler_s_err.remove()

                # Draw new error bar
                ruler_s_err = ax.errorbar(
                    [MEAS_RULER_X], [z_hat_h],
                    yerr=2 * std_S,
                    fmt='none',
                    ecolor='purple',
                    elinewidth=2.0,
                    capsize=6,
                    alpha=0.6,
                    zorder=11
                )

                # Ruler line & label
                ruler_line.set_data([MEAS_RULER_X, MEAS_RULER_X], [0, Y_LIM])
                ruler_label.set_text('Meas Space\n(Heights)')

            else:
                # Normal state-space sigma mode
                sigma_dots.set_offsets(np.column_stack((sigma_x_masked, sigma_y_masked)))

                for sx in sigma_x_masked:
                    ln, = ax.plot([sx, sx], [h(sx), BIRD_HEIGHT], 'k--', lw=0.6, alpha=0.3, zorder=8)
                    projection_lines.append(ln)

                # Hide ruler elements when in state mode
                ruler_sigma.set_offsets(np.empty((0,2)))
                ruler_sigma.set_sizes([])
                ruler_sigma.set_facecolor('none')
                ruler_zhat_dot.set_data([], [])
                ruler_line.set_data([], [])
                ruler_s_err.remove()
                ruler_s_err = ax.errorbar([], [], yerr=[], fmt='none')                
                ruler_label.set_text('')

        else:
            sigma_dots.set_offsets(np.empty((0,2)))
            sigma_dots.set_sizes([])
            ruler_sigma.set_offsets(np.empty((0,2)))
            ruler_sigma.set_sizes([])
            ruler_sigma.set_facecolor('none')
            ruler_zhat_dot.set_data([], [])
            ruler_line.set_data([], [])
            ruler_s_err.remove()
            ruler_s_err = ax.errorbar([], [], yerr=[], fmt='none')            
            ruler_label.set_text('')

        # ── Measurement ──
        if sub >= 4:
            meas_dot.set_data([true_pos], [h(true_pos)])
            meas_proj_line.set_data([true_pos, true_pos], [h(true_pos), BIRD_HEIGHT])
        else:
            meas_dot.set_data([], [])
            meas_proj_line.set_data([], [])

        # ── Predicted measurement ──
        if sub >= 4:
            pred_h = ukf.last_z_hat[0]
            pred_meas_dot.set_data([predicted_x[0]], [pred_h])   # x = predicted pos, y = z_hat[0]
        else:
            pred_meas_dot.set_data([], [])

        if sub < 6:
            # ── Blue & Magenta dot sizes — ONLY in Kalman gain step (sub==6) ──
            meas_dot.set_markersize(9)
            pred_meas_dot.set_markersize(9)

        if sub == 6 and hasattr(ukf, 'last_K') and ukf.last_K is not None:
            gain = ukf.last_K[0,0]
            trust_meas = gain
            trust_pred = 1 - gain

            meas_size = 8 + 18 * trust_meas   # up to ~26 when trust=1
            pred_size = 8 + 18 * trust_pred

            meas_dot.set_markersize(meas_size)
            pred_meas_dot.set_markersize(pred_size)      

        if sub == 5:
            pred_h = ukf.last_z_hat[0]
            meas_h = h(true_pos)
            y_low  = min(pred_h, meas_h)
            y_high = max(pred_h, meas_h)
            mid_x  = (predicted_x[0] + true_pos) / 2

            ln1, = ax.plot([predicted_x[0], true_pos], [pred_h, pred_h], 'purple', lw=1.3, alpha=0.8, zorder=14)
            ln2, = ax.plot([true_pos, predicted_x[0]], [meas_h, meas_h], 'purple', lw=1.3, alpha=0.8, zorder=14)
            innovation_lines.extend([ln1, ln2])

            dy = y_high - y_low
            if dy > 0.01:
                arrow = ax.arrow(mid_x, y_low, 0, dy, head_width=0.6, width=0.25,
                                 length_includes_head=True, color='purple', alpha=0.9, zorder=16)
                arrow_rev = ax.arrow(mid_x, y_high, 0, -dy, head_width=0.6, width=0.25,
                                     length_includes_head=True, color='purple', alpha=0.9, zorder=16)
                innovation_lines.extend([arrow, arrow_rev])

        # ── Bird ──
        if bird_marker:
            bird_marker[0].remove()
            bird_marker.clear()
        img = plt.imread('ukf_explanation_video/Stork_silhouette.png')
        img = np.fliplr(img)
        imagebox = OffsetImage(img, zoom=0.03)
        ab = AnnotationBbox(imagebox, (true_pos, BIRD_HEIGHT), frameon=False, zorder=30)
        ax.add_artist(ab)
        bird_marker.append(ab)

        return (est_dot, est_line, pred_dot, pred_line, sigma_dots, meas_dot, meas_proj_line,
                pred_meas_dot, annotation, *projection_lines, *innovation_lines)

    total_frames = NUM_STEPS * sub_steps
    # ani = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)

    # Advance animation via spacebar
    def on_key(event):
        global current_frame
        if event.key == ' ':
            update(current_frame)
            fig.canvas.draw_idle()
            current_frame += 1
    fig.canvas.mpl_connect('key_press_event', on_key)

    ## If I want to advance animation via mouse click
    # def on_click(event):
    #     global current_frame
    #     update(current_frame)
    #     fig.canvas.draw_idle()
    #     current_frame += 1

    # fig.canvas.mpl_connect('button_press_event', on_click)    

    plt.tight_layout(pad=0)
    plt.show()


# TODO: 
# 1. Put a semi transparent bird above the estimated position.
# 2. Always put a dashed line beneath the true bird and show measurement dot.
# 3. Estimator should work before the bird starts moving. The initial motion of the bird is unnecessary.
# 4. The initial sigma point shouldn't land on the actual bird.
# 5. Use the phrases predictor corrector phrasing.
# General notes:
# There's a non linear unknown function between the position and height.
