import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


np.random.seed(0)

# Prior Gaussian
mu = 0.0
P = 1.5
sigma = np.sqrt(P)

# Unscented Transform parameters
n = 1
alpha = 0.7
kappa = 0
beta = 2

lam = alpha**2 * (n + kappa) - n
gamma = np.sqrt(n + lam)

# Sigma points
sigma_points = np.array([
    mu,
    mu + gamma * sigma,
    mu - gamma * sigma
])

# Weights
Wm = np.array([
    lam / (n + lam),
    1/(2*(n+lam)),
    1/(2*(n+lam))
])

Wc = Wm.copy()
Wc[0] += (1 - alpha**2 + beta)

# Nonlinear transform (mild quadratic → moderate positive skew)
def f(x):
    return x + 0.15 * x**2

# Transform sigma points
y_sigma = f(sigma_points)

# UKF mean and covariance
mu_ukf = np.sum(Wm * y_sigma)
P_ukf = np.sum(Wc * (y_sigma - mu_ukf)**2)
sigma_ukf = np.sqrt(P_ukf)

# True distribution via Monte-Carlo
N = 200000
samples = np.random.normal(mu, sigma, N)
y_samples = f(samples)

# True statistics from samples
true_mean = np.mean(y_samples)
true_var  = np.var(y_samples)
true_std  = np.sqrt(true_var)

# KDE estimate of true PDF
kde = gaussian_kde(y_samples)

# Evaluation points – centered better around the mean
x_min = true_mean - 4.5 * max(true_std, sigma_ukf)
x_max = true_mean + 4.5 * max(true_std, sigma_ukf)
x = np.linspace(x_min, x_max, 800)

true_pdf  = kde(x)
gauss_pdf = (1/np.sqrt(2*np.pi*P_ukf)) * np.exp(-(x-mu_ukf)**2/(2*P_ukf))

# Interpolation functions to get PDF value at any x position
interp_true  = interp1d(x, true_pdf,  kind='linear', fill_value=0.0, bounds_error=False)
interp_gauss = interp1d(x, gauss_pdf, kind='linear', fill_value=0.0, bounds_error=False)

# Heights for vertical lines
h_true_mean   = interp_true(true_mean)
h_true_plus   = interp_true(true_mean + true_std)
h_true_minus  = interp_true(true_mean - true_std)

h_ukf_mean    = interp_gauss(mu_ukf)
h_ukf_plus    = interp_gauss(mu_ukf + sigma_ukf)
h_ukf_minus   = interp_gauss(mu_ukf - sigma_ukf)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

# Distributions
ax.plot(x, true_pdf, linewidth=2.2, color='#1f77b4', label="True Distribution (Monte-Carlo + KDE)")
ax.plot(x, gauss_pdf, linestyle="--", linewidth=2.2, color='#ff7f0e',
        label="Gaussian Approximation (UKF)")


# Vertical lines - TRUE
ax.vlines(true_mean, ymin=0, ymax=h_true_mean,
          color='#d62728', linestyle="-", linewidth=2.6, alpha=1,
          label=f"True mean = {true_mean:.3f}")
ax.vlines(true_mean + true_std, ymin=0, ymax=h_true_plus,
          color='#d62728', linestyle="-", linewidth=2.3, alpha=1,
          label=f"True ±σ (cov = {true_var:.3f})")
ax.vlines(true_mean - true_std, ymin=0, ymax=h_true_minus,
          color='#d62728', linestyle="-", linewidth=2.3, alpha=1)

# Vertical lines - UKF approximated
ax.vlines(mu_ukf, ymin=0, ymax=h_ukf_mean,
          color='#2ca02c', linestyle="--", linewidth=1.6, alpha=1,
          label=f"UKF mean = {mu_ukf:.3f}")
ax.vlines(mu_ukf + sigma_ukf, ymin=0, ymax=h_ukf_plus,
          color='#2ca02c', linestyle="--", linewidth=1.3, alpha=1,
          label=f"UKF ±σ (cov = {P_ukf:.3f})")
ax.vlines(mu_ukf - sigma_ukf, ymin=0, ymax=h_ukf_minus,
          color='#2ca02c', linestyle="--", linewidth=1.3, alpha=1)



# Transformed sigma points (at bottom)
ax.scatter(
    y_sigma,
    np.full_like(y_sigma, 1e-4),          # very small positive value (invisible on plot scale)
    s=[140, 70, 70],
    color=['#d62728', '#1f77b4', '#1f77b4'],
    edgecolor='black',
    zorder=5,
    clip_on=False                          # crucial: allows drawing outside axis limits
)

# Formatting
ax.set_title("Unscented Transform – Nonlinear Function f(x) = x + 0.15 x²\n"
             f"True mean/var: {true_mean:.3f} / {true_var:.3f}    |    "
             f"UKF mean/var: {mu_ukf:.3f} / {P_ukf:.3f}")
ax.set_xlabel(r"$y = f(x)$")
ax.set_ylabel("Probability Density")

ax.set_yticks([])
ax.set_ylim(0, None)

# Show only μ and σ on x-axis
xticks = [mu_ukf, mu_ukf + sigma_ukf, mu_ukf - sigma_ukf]
xticklabels = [r"$\mu$", r"$\sigma$", r"$-\sigma$"]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=12)

# Better centering
ax.set_xlim(x_min, x_max)

ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax.grid(True, alpha=0.18, linestyle=':')

plt.tight_layout()
plt.show()