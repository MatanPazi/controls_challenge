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

plt.plot(t, ay_actual, label="Actual")
plt.plot(t, ay_pred, label="Predicted")

ctrl_rmse = np.sqrt(((ay_ref - ay_actual) ** 2).mean())
pred_rmse = np.sqrt(((ay_pred - ay_actual) ** 2).mean())

plt.legend()
plt.title(f"Control RMSE: {ctrl_rmse:.4f}, Prediction RMSE: {pred_rmse:.4f}")
plt.xlabel("Step")
plt.ylabel("Lateral Acceleration")

plt.grid()
plt.show()