import numpy as np
import matplotlib.pyplot as plt

# Gaussian parameters
mu = 0.0              # mean
P = 10.0              # variance
sigma = np.sqrt(P)    # standard deviation

# Generate x-axis and Gaussian PDF (extended range for better tail visualization)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
pdf = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((x - mu)**2) / P)

# Unscented Transform parameters (for reference / consistency)
n = 1
alpha = 0.5
kappa = 0
lam = alpha**2 * (n + kappa) - n

# Sigma points (for 1D)
sigma_points = np.array([
    mu,
    mu + np.sqrt((n + lam) * P),
    mu - np.sqrt((n + lam) * P)
])

# Heights where lines should stop (PDF value at each point)
h_mu    = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((mu - mu)**2) / P)
h_plus  = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((sigma_points[1] - mu)**2) / P)
h_minus = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * ((sigma_points[2] - mu)**2) / P)
h_cov_plus = (1 / np.sqrt(2 * np.pi * P)) * np.exp(-0.5 * (((mu + sigma) - mu)**2) / P)

# Plot
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(x, pdf, label="Gaussian PDF", linewidth=2, color='C0')

# Vertical lines stopping at the curve
ax.vlines(mu,           ymin=0, ymax=h_mu,    color='darkred', linestyle="--", linewidth=1.5, label=r"Mean $\mu$")
# ax.vlines(sigma_points[1], ymin=0, ymax=h_plus,  color='blue', linestyle="--", linewidth=1.2)
# ax.vlines(sigma_points[2], ymin=0, ymax=h_minus, color='blue', linestyle="--", linewidth=1.2)
ax.vlines([mu - sigma, mu + sigma], ymin=0, ymax=h_cov_plus, color='green', linestyle="--", linewidth=1.2, label=r"$\pm\sigma$ covariance")

# Sigma points markers on x-axis (with clip_on=False to avoid any edge clipping)
ax.scatter([sigma_points[0]], [0], s=120, color='darkred',   marker="o", clip_on=False)
ax.scatter(sigma_points[1:],  [0, 0], s=60,  color='blue', marker="o", clip_on=False)

ax.set_title("Gaussian Distribution with Mean and Sigma Points")
ax.set_xlabel(r"$x$")
ax.set_ylabel("Probability Density")

# Show only μ and σ on x-axis
xticks = [mu - sigma, mu, mu + sigma]
xticklabels = [r"$-\sigma$", r"$\mu$", r"$\sigma$"]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=12)

# No y-axis numeric ticks, and ensure ylim starts exactly at 0
ax.set_yticks([])
ax.set_ylim(0, None)

ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()