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

# ────────────────────────────────────────────────
#  Part 1: Input Gaussian + Sigma Points
# ────────────────────────────────────────────────

# Unscented Transform parameters
n = 1
alpha = 0.7
kappa = 0
beta = 2
lam = alpha**2 * (n + kappa) - n
gamma = np.sqrt(n + lam)

# Sigma points (before transformation)
sigma_points = np.array([
    mu,
    mu + gamma * sigma,
    mu - gamma * sigma
])

# Original Gaussian PDF
x_prior = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
pdf_prior = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((x_prior - mu)**2 / P))

# Heights for vertical lines (prior)
h_mu    = pdf_prior[np.argmin(np.abs(x_prior - mu))]
h_sigma = pdf_prior[np.argmin(np.abs(x_prior - (mu + sigma)))]

# ────────────────────────────────────────────────
#  Part 2: Transformation & UKF + True distribution
# ────────────────────────────────────────────────

# Transform sigma points
y_sigma = f(sigma_points)

# UKF weights
Wm = np.array([lam / (n + lam)] + [1/(2*(n+lam))] * 2)
Wc = Wm.copy()
Wc[0] += (1 - alpha**2 + beta)

# UKF mean and covariance
mu_ukf = np.sum(Wm * y_sigma)
P_ukf = np.sum(Wc * (y_sigma - mu_ukf)**2)
sigma_ukf = np.sqrt(P_ukf)

# Monte-Carlo for true distribution
N = 200_000
x_samples = np.random.normal(mu, sigma, N)
y_samples = f(x_samples)

true_mean = np.mean(y_samples)
true_var = np.var(y_samples)
true_std = np.sqrt(true_var)

# KDE
kde = gaussian_kde(y_samples)

# Evaluation grid for output plot
x_min = true_mean - 4.5 * max(true_std, sigma_ukf)
x_max = true_mean + 4.5 * max(true_std, sigma_ukf)
x_out = np.linspace(x_min, x_max, 800)

true_pdf = kde(x_out)
ukf_pdf = (1 / np.sqrt(2 * np.pi * P_ukf)) * np.exp(-0.5 * ((x_out - mu_ukf)**2 / P_ukf))

# Interpolation for line heights
interp_true = interp1d(x_out, true_pdf, kind='linear', fill_value=0.0, bounds_error=False)
interp_ukf  = interp1d(x_out, ukf_pdf,  kind='linear', fill_value=0.0, bounds_error=False)

h_true_mean  = interp_true(true_mean)
h_true_plus  = interp_true(true_mean + true_std)
h_true_minus = interp_true(true_mean - true_std)

h_ukf_mean   = interp_ukf(mu_ukf)
h_ukf_plus   = interp_ukf(mu_ukf + sigma_ukf)
h_ukf_minus  = interp_ukf(mu_ukf - sigma_ukf)

# ────────────────────────────────────────────────
#  Plotting – two subplots side by side
# ────────────────────────────────────────────────

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# ── Left: Input Gaussian + Sigma Points ────────────────────────────────
ax_left.plot(x_prior, pdf_prior, lw=2.2, color='#1f77b4', label='Gaussian PDF')

ax_left.vlines(mu,           0, h_mu,    color='darkred',   ls='-', lw=2.6, label=r"Mean $\mu$")
ax_left.vlines(mu + sigma,   0, h_sigma, color='seagreen', ls='--', lw=2.3, label=r"$\pm\sigma$ covariance")
ax_left.vlines(mu - sigma,   0, h_sigma, color='seagreen', ls='--', lw=2.3)

ax_left.scatter(sigma_points[0], 0, s=140, color='darkred',   edgecolor='k', zorder=10, clip_on=False, label="Sigma points")
ax_left.scatter(sigma_points[1:], [0,0], s=70, color='#1f77b4', edgecolor='k', zorder=10, clip_on=False)

ax_left.set_title("Input: Gaussian prior + Sigma points", fontsize=13, pad=12)
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
ax_right.plot(x_out, ukf_pdf, ls="--", lw=2.2, color='#ff7f0e', label="UKF Gaussian approximation")

# True statistics
ax_right.vlines(true_mean,           0, h_true_mean,  color='#d62728', ls='-', lw=2.6, label=f"True mean = {true_mean:.3f}")
ax_right.vlines(true_mean + true_std, 0, h_true_plus, color='#d62728', ls="--", lw=2.0, label=f"True ±σ (var = {true_var:.3f})")
ax_right.vlines(true_mean - true_std, 0, h_true_minus,color='#d62728', ls="--", lw=2.0)

# UKF approximation
ax_right.vlines(mu_ukf,              0, h_ukf_mean,   color='#2ca02c', ls='-', lw=1.8, label=f"UKF mean = {mu_ukf:.3f}")
ax_right.vlines(mu_ukf + sigma_ukf,  0, h_ukf_plus,   color='#2ca02c', ls="--", lw=1.4)
ax_right.vlines(mu_ukf - sigma_ukf,  0, h_ukf_minus,  color='#2ca02c', ls="--", lw=1.4, label=f"UKF ±σ (var = {P_ukf:.3f})")

# Transformed sigma points
ax_right.scatter(y_sigma, np.full_like(y_sigma, 1e-4), s=[140,70,70],
                 color=['#d62728','#1f77b4','#1f77b4'], edgecolor='k', zorder=10, clip_on=False,
                 label="Transformed sigma points")

ax_right.set_title("Output after f(x) = x + 0.15 x²", fontsize=13, pad=12)
ax_right.set_xlabel("$y = f(x)$", fontsize=11)
ax_right.set_ylabel("Probability Density", fontsize=11)
ax_right.set_yticks([])
ax_right.set_ylim(0, None)
ax_right.set_xlim(x_min, x_max)
ax_right.set_xticks([mu_ukf - sigma_ukf, mu_ukf, mu_ukf + sigma_ukf])
ax_right.set_xticklabels([r"$-\sigma$", r"$\mu$", r"$\sigma$"], fontsize=11)
ax_right.legend(loc='upper right', fontsize=9.8, framealpha=0.92)
ax_right.grid(alpha=0.15, ls=':')

fig.suptitle("Unscented Kalman Filter – Before and After Nonlinear Transformation\n"
             f"Input var = {P:.2f}   |   True output var = {true_var:.3f}   |   UKF output var = {P_ukf:.3f}",
             fontsize=14, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()