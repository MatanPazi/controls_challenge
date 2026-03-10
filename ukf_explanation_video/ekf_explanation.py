import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

np.random.seed(42)  # for reproducibility

# ────────────────────────────────────────────────
#  Common parameters
# ────────────────────────────────────────────────
mu = 0.0
P = 1.5
sigma = np.sqrt(P)

# Nonlinear function
def f(x):
    return x + 0.15 * x**2

# Jacobian (derivative)
def df_dx(x):
    return 1 + 0.3 * x

# ────────────────────────────────────────────────
#  EKF approximation (linearization at mean)
# ────────────────────────────────────────────────
ekf_mean = f(mu)                # = 0
F = df_dx(mu)                   # = 1
ekf_var = F**2 * P              # = 1.5
ekf_std = np.sqrt(ekf_var)

# ────────────────────────────────────────────────
#  True distribution via Monte-Carlo
# ────────────────────────────────────────────────
N = 200_000
x_samples = np.random.normal(mu, sigma, N)
y_samples = f(x_samples)

true_mean = np.mean(y_samples)
true_var  = np.var(y_samples)
true_std  = np.sqrt(true_var)

kde = gaussian_kde(y_samples)

# Output evaluation grid
x_min = true_mean - 4.5 * max(true_std, ekf_std)
x_max = true_mean + 4.5 * max(true_std, ekf_std)
x_out = np.linspace(x_min, x_max, 800)

true_pdf = kde(x_out)
ekf_pdf  = (1 / np.sqrt(2 * np.pi * ekf_var)) * np.exp(-0.5 * ((x_out - ekf_mean)**2 / ekf_var))

# Interpolation for accurate heights
interp_true = interp1d(x_out, true_pdf, kind='linear', fill_value=0.0, bounds_error=False)
interp_ekf  = interp1d(x_out, ekf_pdf,  kind='linear', fill_value=0.0, bounds_error=False)

h_true_mean  = interp_true(true_mean)
h_true_plus  = interp_true(true_mean + true_std)
h_true_minus = interp_true(true_mean - true_std)

h_ekf_mean   = interp_ekf(ekf_mean)
h_ekf_plus   = interp_ekf(ekf_mean + ekf_std)
h_ekf_minus  = interp_ekf(ekf_mean - ekf_std)

# ────────────────────────────────────────────────
#  Input Gaussian PDF (for left plot)
# ────────────────────────────────────────────────
x_prior = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
pdf_prior = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((x_prior - mu)**2 / P))

# Heights for input vertical lines
interp_prior = interp1d(x_prior, pdf_prior, kind='linear', fill_value=0.0, bounds_error=False)
h_mu_input    = interp_prior(mu)
h_sigma_input = interp_prior(mu + sigma)

# ────────────────────────────────────────────────
#  Plot – two subplots side by side
# ────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# ── Left: Input Gaussian (linearization point) ──────────────────────────
ax_left.plot(x_prior, pdf_prior, lw=2.2, color='#1f77b4', label='Gaussian PDF')

ax_left.vlines(mu,           0, h_mu_input,    color='darkred',   ls='-', lw=2.6, label=r"Mean $\mu$ (linearization point)")
ax_left.vlines(mu + sigma,   0, h_sigma_input, color='seagreen', ls='--', lw=2.3, label=r"$\pm\sigma$ covariance")
ax_left.vlines(mu - sigma,   0, h_sigma_input, color='seagreen', ls='--', lw=2.3)

# Mark the linearization point at bottom (same style as UKF sigma points)
ax_left.scatter([mu], [1e-4], s=140, color='darkred', edgecolor='black', zorder=10, clip_on=False,
                label="Linearization point μ")

ax_left.set_title("EKF Input: Gaussian prior (linearization at mean)", fontsize=13, pad=12)
ax_left.set_xlabel("$x$", fontsize=11)
ax_left.set_ylabel("Probability Density", fontsize=11)
ax_left.set_yticks([])
ax_left.set_ylim(0, None)
ax_left.set_xticks([mu - sigma, mu, mu + sigma])
ax_left.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=11)
ax_left.legend(loc='upper right', fontsize=10, framealpha=0.92)
ax_left.grid(alpha=0.15, ls=':')

# ── Right: Transformed distribution ─────────────────────────────────────
ax_right.plot(x_out, true_pdf, lw=2.2, color='#1f77b4', label="True distribution (MC + KDE)")
ax_right.plot(x_out, ekf_pdf, ls="--", lw=2.2, color='#ff7f0e', label="Gaussian approximation (EKF)")

# True statistics
ax_right.vlines(true_mean,           0, h_true_mean,  color='#d62728', ls='-', lw=2.6, label=f"True mean = {true_mean:.3f}")
ax_right.vlines(true_mean + true_std, 0, h_true_plus, color='#d62728', ls="--", lw=2.0, label=f"True ±σ (var = {true_var:.3f})")
ax_right.vlines(true_mean - true_std, 0, h_true_minus,color='#d62728', ls="--", lw=2.0)

# EKF approximation
ax_right.vlines(ekf_mean,            0, h_ekf_mean,   color='#2ca02c', ls='-', lw=1.8, label=f"EKF mean = {ekf_mean:.3f}")
ax_right.vlines(ekf_mean + ekf_std,  0, h_ekf_plus,   color='#2ca02c', ls="--", lw=1.4, label=f"EKF ±σ (var = {ekf_var:.3f})")
ax_right.vlines(ekf_mean - ekf_std,  0, h_ekf_minus,  color='#2ca02c', ls="--", lw=1.4)

# Mark linearized point f(μ) at bottom
ax_right.scatter([ekf_mean], [1e-4], s=140, color='#d62728', edgecolor='black', zorder=10, clip_on=False,
                 label="Linearized f(μ)")

ax_right.set_title("EKF Output: after f(x) = x + 0.15 x²", fontsize=13, pad=12)
ax_right.set_xlabel("$y = f(x)$", fontsize=11)
ax_right.set_ylabel("Probability Density", fontsize=11)
ax_right.set_yticks([])
ax_right.set_ylim(0, None)
ax_right.set_xlim(x_min, x_max)
ax_right.set_xticks([ekf_mean - ekf_std, ekf_mean, ekf_mean + ekf_std])
ax_right.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=11)
ax_right.legend(loc='upper right', fontsize=9.8, framealpha=0.92)
ax_right.grid(alpha=0.15, ls=':')

# Overall title
fig.suptitle("Extended Kalman Filter – Before and After Nonlinear Transformation\n"
             f"Input var = {P:.2f}   |   True output var = {true_var:.3f}   |   EKF output var = {ekf_var:.3f}",
             fontsize=14, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
