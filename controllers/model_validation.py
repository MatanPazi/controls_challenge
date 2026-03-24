import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(
    "mpc_log.txt",
    delimiter=",",
    names=True,      # use header row
    dtype=None,      # auto-detect types
    encoding="utf-8"
)

t = data["step"]
ay_actual = data["ay"]
ay_pred = data["ay_pred"]
ay_ref =  data["ay_ref"]


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) # sharex=True aligns the x-axes
ctrl_rmse = np.sqrt(((ay_ref - ay_actual) ** 2).mean())
pred_rmse = np.sqrt(((ay_pred - ay_actual) ** 2).mean())

# Plot data on the first (top) subplot
ax1.plot(t, ay_actual, label="Actual")
ax1.plot(t, ay_pred, label="Predicted")
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Lateral acceleration [m/s^2]')
ax1.set_title(f'Actual Vs. Predicted - RMSE = {pred_rmse:.4f}')
ax1.legend()

# Plot data on the second (bottom) subplot
ax2.plot(t, ay_actual - ay_ref, label="Error")
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Lateral acceleration [m/s^2]')
ax2.set_title(f'Error - RMSE = {ctrl_rmse:.4f}')
ax2.legend()

plt.tight_layout() 
plt.show()
