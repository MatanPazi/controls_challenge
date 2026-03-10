import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

np.random.seed(42)

# ────────────────────────────────────────────────
#  Parameters
# ────────────────────────────────────────────────
mu = 0.0
P = 1.5
sigma = np.sqrt(P)

# Real (nonlinear) process model — what actually happens
def f_real(x):
    return x + 0.15 * x**2

# What the linear Kalman filter assumes (ignores the quadratic term)
def f_linear(x):
    return x   # ← linear KF pretends this is the model

F_linear = 1.0   # Jacobian / transition matrix (constant 1)

# Linear KF prediction
kf_mean = f_linear(mu)          # = 0
kf_var  = F_linear**2 * P       # = 1.5
kf_std  = np.sqrt(kf_var)

# ────────────────────────────────────────────────
#  True distribution — uses the REAL nonlinear function
# ────────────────────────────────────────────────
N = 200_000
x_samples = np.random.normal(mu, sigma, N)
y_samples = f_real(x_samples)           # ← nonlinear!

true_mean = np.mean(y_samples)
true_var  = np.var(y_samples)
true_std  = np.sqrt(true_var)

kde = gaussian_kde(y_samples)

# Evaluation grid — centered on the true (nonlinear) output
x_min = true_mean - 4.5 * max(true_std, kf_std)
x_max = true_mean + 4.5 * max(true_std, kf_std)
x = np.linspace(x_min, x_max, 800)

true_pdf = kde(x)
kf_pdf   = (1 / np.sqrt(2 * np.pi * kf_var)) * np.exp(-0.5 * ((x - kf_mean)**2 / kf_var))

# Interpolation for line heights
interp_true = interp1d(x, true_pdf, kind='linear', fill_value=0.0, bounds_error=False)
interp_kf   = interp1d(x, kf_pdf,   kind='linear', fill_value=0.0, bounds_error=False)

h_true_mean  = interp_true(true_mean)
h_true_plus  = interp_true(true_mean + true_std)
h_true_minus = interp_true(true_mean - true_std)

h_kf_mean    = interp_kf(kf_mean)
h_kf_plus    = interp_kf(kf_mean + kf_std)
h_kf_minus   = interp_kf(kf_mean - kf_std)

# ────────────────────────────────────────────────
#  Input Gaussian PDF (left plot — same as before)
# ────────────────────────────────────────────────
x_prior = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
pdf_prior = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((x_prior - mu)**2 / P))

interp_prior = interp1d(x_prior, pdf_prior, kind='linear', fill_value=0.0, bounds_error=False)
h_mu_input    = interp_prior(mu)
h_sigma_input = interp_prior(mu + sigma)

# ────────────────────────────────────────────────
#  Plot – two subplots side by side
# ────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# ── Left: Input Gaussian ────────────────────────────────────────────────
ax_left.plot(x_prior, pdf_prior, lw=2.2, color='#1f77b4', label='Gaussian PDF')

ax_left.vlines(mu,           0, h_mu_input,    color='darkred',   ls='-', lw=2.6, label=r"Mean $\mu$")
ax_left.vlines(mu + sigma,   0, h_sigma_input, color='seagreen', ls='--', lw=2.3, label=r"$\pm\sigma$ covariance")
ax_left.vlines(mu - sigma,   0, h_sigma_input, color='seagreen', ls='--', lw=2.3)

ax_left.set_title("Input: Gaussian prior", fontsize=13, pad=12)
ax_left.set_xlabel("$x$", fontsize=11)
ax_left.set_ylabel("Probability Density", fontsize=11)
ax_left.set_yticks([])
ax_left.set_ylim(0, None)
ax_left.set_xticks([mu - sigma, mu, mu + sigma])
ax_left.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=11)
ax_left.legend(loc='upper right', fontsize=10, framealpha=0.92)
ax_left.grid(alpha=0.15, ls=':')

# ── Right: Linear KF vs true nonlinear reality ──────────────────────────
ax_right.plot(x, true_pdf, lw=2.2, color='#1f77b4', label="True distribution (nonlinear reality)")
ax_right.plot(x, kf_pdf, ls="--", lw=2.2, color='#ff7f0e', label="Linear KF prediction (assumes y = x)")

# True (nonlinear) statistics
ax_right.vlines(true_mean,           0, h_true_mean,  color='#d62728', ls='-', lw=2.6,
                label=f"True mean = {true_mean:.3f}")
ax_right.vlines(true_mean + true_std, 0, h_true_plus, color='#d62728', ls="--", lw=2.0,
                label=f"True ±σ (var = {true_var:.3f})")
ax_right.vlines(true_mean - true_std, 0, h_true_minus, color='#d62728', ls="--", lw=2.0)

# Linear KF prediction
ax_right.vlines(kf_mean,            0, h_kf_mean,   color='#2ca02c', ls='-', lw=1.8,
                label=f"Linear KF mean = {kf_mean:.3f}")
ax_right.vlines(kf_mean + kf_std,   0, h_kf_plus,   color='#2ca02c', ls="--", lw=1.4,
                label=f"KF ±σ (var = {kf_var:.3f})")
ax_right.vlines(kf_mean - kf_std,   0, h_kf_minus,  color='#2ca02c', ls="--", lw=1.4)

ax_right.set_title("Linear Kalman Filter applied to nonlinear reality\n"
                   "f(x) = x + 0.15 x² but KF assumes y = x", fontsize=13, pad=12)
ax_right.set_xlabel("$y$", fontsize=11)
ax_right.set_ylabel("Probability Density", fontsize=11)
ax_right.set_yticks([])
ax_right.set_ylim(0, None)
ax_right.set_xlim(x_min, x_max)
ax_right.set_xticks([kf_mean - kf_std, kf_mean, kf_mean + kf_std])
ax_right.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=11)
ax_right.legend(loc='upper right', fontsize=9.8, framealpha=0.92)
ax_right.grid(alpha=0.15, ls=':')

# Overall title
fig.suptitle("Standard Linear Kalman Filter vs Nonlinear Reality\n"
             f"Input var = {P:.2f}   |   True output var = {true_var:.3f}   |   KF output var = {kf_var:.3f}",
             fontsize=14, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()