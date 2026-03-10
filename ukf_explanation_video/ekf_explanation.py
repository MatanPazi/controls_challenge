import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

np.random.seed(0)

# Prior Gaussian (same as UKF)
mu = 0.0
P = 1.5
sigma = np.sqrt(P)

# Nonlinear transform (same as UKF)
def f(x):
    return x + 0.15 * x**2

# EKF approximation (linearization at mu)
ekf_mean = f(mu)  # f(mu)
# Jacobian F = df/dx at mu
def df_dx(x):
    return 1 + 0.3 * x  # derivative of f(x) = x + 0.15 x^2
F = df_dx(mu)  # at mu=0, F=1
ekf_var = F**2 * P  # in 1D: F * P * F.T
ekf_std = np.sqrt(ekf_var)

# True distribution via Monte-Carlo (same as UKF)
N = 200000
samples = np.random.normal(mu, sigma, N)
y_samples = f(samples)

# True statistics from samples (same as UKF)
true_mean = np.mean(y_samples)
true_var = np.var(y_samples)
true_std = np.sqrt(true_var)

# KDE estimate of true PDF (same as UKF)
kde = gaussian_kde(y_samples)

# Evaluation points – centered around the data
x_min = true_mean - 4.5 * max(true_std, ekf_std)
x_max = true_mean + 4.5 * max(true_std, ekf_std)
x = np.linspace(x_min, x_max, 800)

true_pdf = kde(x)
ekf_pdf = (1 / np.sqrt(2 * np.pi * ekf_var)) * np.exp(-0.5 * ((x - ekf_mean)**2 / ekf_var))

# Interpolation functions to get PDF value at any x position
interp_true = interp1d(x, true_pdf, kind='linear', fill_value=0.0, bounds_error=False)
interp_ekf = interp1d(x, ekf_pdf, kind='linear', fill_value=0.0, bounds_error=False)

# Heights for vertical lines
h_true_mean = interp_true(true_mean)
h_true_plus = interp_true(true_mean + true_std)
h_true_minus = interp_true(true_mean - true_std)

h_ekf_mean = interp_ekf(ekf_mean)
h_ekf_plus = interp_ekf(ekf_mean + ekf_std)
h_ekf_minus = interp_ekf(ekf_mean - ekf_std)

# --- Before Plot for EKF ---
fig_before, ax_before = plt.subplots(figsize=(9, 4.5))

# Original Gaussian PDF
x_orig = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 800)
orig_pdf = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((x_orig - mu)**2 / P))

ax_before.plot(x_orig, orig_pdf, linewidth=2.2, color='#1f77b4', label="Original Gaussian PDF")

# Heights for original lines
interp_orig = interp1d(x_orig, orig_pdf, kind='linear', fill_value=0.0, bounds_error=False)
h_mu = interp_orig(mu)
h_plus = interp_orig(mu + sigma)
h_minus = interp_orig(mu - sigma)

# Vertical lines for mean and ±σ (reaching the curve)
ax_before.vlines(mu, ymin=0, ymax=h_mu, color='darkred', linestyle="-", linewidth=2.6, label=r"Mean $\mu$ (linearization point)")
ax_before.vlines(mu + sigma, ymin=0, ymax=h_plus, color='green', linestyle="--", linewidth=2.3, label=r"$\pm\sigma$ covariance")
ax_before.vlines(mu - sigma, ymin=0, ymax=h_minus, color='green', linestyle="--", linewidth=2.3)

ax_before.set_title("EKF Input: Original Gaussian (Linearization at Mean)")
ax_before.set_xlabel(r"$x$")
ax_before.set_ylabel("Probability Density")
ax_before.set_yticks([])
ax_before.set_ylim(0, None)
ax_before.set_xticks([mu - sigma, mu, mu + sigma])
ax_before.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=12)
ax_before.legend(loc='upper right', fontsize=10)
ax_before.grid(True, alpha=0.3)

# --- After Plot for EKF ---
fig_after, ax_after = plt.subplots(figsize=(10, 5))

# Distributions
ax_after.plot(x, true_pdf, linewidth=2.2, color='#1f77b4', label="True Distribution (Monte-Carlo + KDE)")
ax_after.plot(x, ekf_pdf, linestyle="--", linewidth=2.2, color='#ff7f0e', label="Gaussian Approximation (EKF)")

# Vertical lines - TRUE
ax_after.vlines(true_mean, ymin=0, ymax=h_true_mean, color='#d62728', linestyle="-", linewidth=2.6, alpha=1, label=f"True mean = {true_mean:.3f}")
ax_after.vlines(true_mean + true_std, ymin=0, ymax=h_true_plus, color='#d62728', linestyle="--", linewidth=2.3, alpha=1, label=f"True ±σ (cov = {true_var:.3f})")
ax_after.vlines(true_mean - true_std, ymin=0, ymax=h_true_minus, color='#d62728', linestyle="--", linewidth=2.3, alpha=1)

# Vertical lines - EKF approximated
ax_after.vlines(ekf_mean, ymin=0, ymax=h_ekf_mean, color='#2ca02c', linestyle="-", linewidth=1.6, alpha=1, label=f"EKF mean = {ekf_mean:.3f}")
ax_after.vlines(ekf_mean + ekf_std, ymin=0, ymax=h_ekf_plus, color='#2ca02c', linestyle="--", linewidth=1.3, alpha=1, label=f"EKF ±σ (cov = {ekf_var:.3f})")
ax_after.vlines(ekf_mean - ekf_std, ymin=0, ymax=h_ekf_minus, color='#2ca02c', linestyle="--", linewidth=1.3, alpha=1)

# Mark the linearized point f(mu) (optional, since it's the EKF mean)
ax_after.scatter([ekf_mean], [0], s=140, color='#d62728', edgecolor='black', zorder=5, label="Linearized f(μ)", clip_on=False)

# Formatting
ax_after.set_title("EKF Output: Transformed Distribution via f(x) = x + 0.15 x²\n"
                   f"True mean/var: {true_mean:.3f} / {true_var:.3f}    |    "
                   f"EKF mean/var: {ekf_mean:.3f} / {ekf_var:.3f}")
ax_after.set_xlabel(r"$y = f(x)$")
ax_after.set_ylabel("Probability Density")
ax_after.set_yticks([])
ax_after.set_ylim(0, None)
ax_after.set_xlim(x_min, x_max)
ax_after.set_xticks([ekf_mean - ekf_std, ekf_mean, ekf_mean + ekf_std])
ax_after.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=12)
ax_after.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

plt.tight_layout()
plt.show()