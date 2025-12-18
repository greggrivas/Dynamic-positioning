import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(r"src/runtime/dp_log.csv")
fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
axs[0].plot(df.t, df.x, label="x")
axs[0].plot(df.t, df.xr, "--", label="x_ref")
axs[0].plot(df.t, df.y, label="y")
axs[0].plot(df.t, df.yr, "--", label="y_ref")
axs[0].legend()

axs[1].plot(df.t, df.psi, label="psi")
axs[1].plot(df.t, df.psir, "--", label="psi_ref")
axs[1].legend()

axs[2].plot(df.t, df.tau_x, label="tau_x")
axs[2].plot(df.t, df.tau_y, label="tau_y")
axs[2].plot(df.t, df.tau_psi, label="tau_psi")
axs[2].legend()

plt.tight_layout()
out = Path("dp_log_plot.png")
plt.savefig(out, dpi=150)
print(f"Saved {out.resolve()}")

