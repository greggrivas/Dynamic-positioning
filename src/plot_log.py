import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_dp_log(csv_path="src/runtime/dp_log.csv", out_path="dp_log_plot.png"):
    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(df.t, df.x, label="x")
    axs[0].plot(df.t, df.xr, "--", label="x_ref")
    axs[0].plot(df.t, df.y, label="y")
    axs[0].plot(df.t, df.yr, "--", label="y_ref")
    axs[0].set_ylabel("Position [m]")
    axs[0].legend()

    # Unwrap angles for cleaner plot
    psi_unwrap = np.unwrap(df.psi)
    psir_unwrap = np.unwrap(df.psir)
    axs[1].plot(df.t, psi_unwrap, label="psi")
    axs[1].plot(df.t, psir_unwrap, "--", label="psi_ref")
    axs[1].set_ylabel("Heading [rad]")
    axs[1].legend()

    axs[2].plot(df.t, df.tau_x, label="tau_x")
    axs[2].plot(df.t, df.tau_y, label="tau_y")
    axs[2].plot(df.t, df.tau_psi, label="tau_psi")
    axs[2].set_ylabel("Control")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()

    plt.tight_layout()
    out = Path(out_path)
    plt.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")

if __name__ == "__main__":
    plot_dp_log()