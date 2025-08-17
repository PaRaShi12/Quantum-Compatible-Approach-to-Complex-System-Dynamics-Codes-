import pandas as pd
import numpy as np
import os
from scipy.linalg import sqrtm
from numpy import trace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from qutip import Qobj, ptrace, fidelity
from itertools import combinations
import math
from PIL import Image

# --- Parameters ---
data_dir = "data"
#taus = list(range(50, 301, 10))  # [50, 75, 100, ..., 300]
taus = [144]
smoothing_window = 1  # No smoothing

# --- Load data ---
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
dataframes = [pd.read_csv(os.path.join(data_dir, file), skiprows=3) for file in files]  # Skip header rows

# Align data by common dates
dates = set(dataframes[0]['Date'])
for df in dataframes[1:]:
    dates = dates.intersection(set(df['Date']))
dates = sorted(list(dates))

# Extract anomalies for common dates
closes = []
for df in dataframes:
    df = df[df['Date'].isin(dates)].sort_values('Date')
    closes.append(df['Anomaly'].values)
closes = np.array(closes)

start_trim = 1
end_trim = 1362
closes = closes[:, start_trim:]
N, T = closes.shape

currency_names = [os.path.splitext(f)[0] for f in files]

# --- Generate binary price movement vectors ---
phi_vectors = [tuple((closes[:, j] > closes[:, j - 1]).astype(int)) for j in range(1, T)]
d = 2 ** N

# --- Helper ---
def vector_to_index(vec):
    return int("".join(map(str, vec)), 2)

# --- Iterate over tau values ---
for tau in taus:
    # Create output directory for this tau
    output_dir = f"tau={tau}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Build density matrices ---
    density_matrices = []
    for t_start in range(len(phi_vectors) - tau + 1):
        window = phi_vectors[t_start:t_start + tau]
        counts = {}
        for vec in window:
            counts[vec] = counts.get(vec, 0) + 1
        P = np.zeros(d)
        for vec, count in counts.items():
            P[vector_to_index(vec)] = count / tau
        psi = np.sqrt(P).reshape(-1, 1)
        rho = psi @ psi.T
        density_matrices.append(rho)

    # --- Convert to Qobj ---
    qobjs = [Qobj(rho, dims=[[2]*N, [2]*N]) for rho in density_matrices]

    # --- Full system fidelity ---
    rho0_qobj = qobjs[0]
    fidelity_full = np.array([fidelity(rho0_qobj, rho_t) for rho_t in qobjs])

    # Save full system fidelity
    np.savetxt(os.path.join(output_dir, "fidelity_full.csv"), fidelity_full, delimiter=",", header="fidelity", comments='')

    # --- Subsystem fidelity (only for q = 1) ---
    subsystem_fids = {}
    subsystem_integrals = {}

    for idx in range(N):
        rho0_A = ptrace(qobjs[0], [idx])
        fids = []
        for rho_t in qobjs:
            rho_t_A = ptrace(rho_t, [idx])
            fid = fidelity(rho0_A, rho_t_A)
            fids.append(fid)
        fids = np.array(fids)
        time_axis = np.arange(len(fids))
        subsystem_fids[idx] = (time_axis, fids)

        # Compute cumulative area above curve under y=1
        integrals = np.cumsum(1 - fids)
        subsystem_integrals[idx] = integrals

        # Save fidelity curve
        np.savetxt(os.path.join(output_dir, f"subsystem_fidelity_C{idx+1}.csv"), np.column_stack((time_axis, fids)), delimiter=",", header="time,fidelity", comments='')

        # Save integral (area above curve)
        np.savetxt(os.path.join(output_dir, f"subsystem_integral_C{idx+1}.csv"), np.column_stack((time_axis, integrals)), delimiter=",", header="time,cumulative_area", comments='')

    # --- Summary of final cumulative area per currency ---
    summary = {
        "Currency": [currency_names[idx] for idx in subsystem_integrals],
        "Final Cumulative Area": [areas[-1] for areas in subsystem_integrals.values()]
    }
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "final_integral_summary.csv"), index=False)

    # --- Plot subsystem fidelities ---
    plt.figure(figsize=(10, 6))
    for idx in range(N):
        time_axis, fids = subsystem_fids[idx]
        plt.plot(time_axis, fids, label=currency_names[idx])
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.title('Subsystem Fidelity for All Currencies')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'subsystem_fidelities.png'))
    plt.close()

    # --- Plot full system fidelity ---
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(fidelity_full)), fidelity_full)
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.title('Full System Fidelity')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fidelity_full.png'))
    plt.close()

    # --- Plot subsystem cumulative areas ---
    plt.figure(figsize=(10, 6))
    for idx in range(N):
        time_axis, integrals = subsystem_fids[idx][0], subsystem_integrals[idx]
        plt.plot(time_axis, integrals, label=currency_names[idx])
    plt.xlabel('Time')
    plt.ylabel('Cumulative Area')
    plt.title('Subsystem Cumulative Area for All Currencies')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'subsystem_integrals.png'))
    plt.close()

    # --- Plot subsystem fidelity for idx = 4 ---
    plt.figure(figsize=(10, 6))
    idx = 4
    time_axis, fids = subsystem_fids[idx]
    plt.plot(time_axis, fids, label=currency_names[idx])
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.title('Subsystem Fidelity for All Currencies')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'subsystem_fidelities.png'))
    plt.close()

    # --- Plot subsystem fidelity for idx = 7 ---
    plt.figure(figsize=(10, 6))
    idx = 7
    time_axis, fids = subsystem_fids[idx]
    plt.plot(time_axis, fids, label=currency_names[idx])
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.title('Subsystem Fidelity for All Currencies')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'subsystem_fidelities.png'))
    plt.close()

# --- Create animation ---
fig, ax = plt.subplots(figsize=(10, 6))
plot_types = ['subsystem_fidelities.png', 'fidelity_full.png', 'subsystem_integrals.png']
frames = [(tau, plot_type) for tau in sorted(taus) for plot_type in plot_types]

def update(frame):
    tau, plot_type = frame
    ax.clear()
    img_path = os.path.join(f"tau={tau}", plot_type)
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Tau = {tau}, {plot_type}")
    return ax,

ani = FuncAnimation(fig, update, frames=frames, interval=500, blit=False)
ani.save('fidelity_animation.mp4', writer='ffmpeg', dpi=100)
plt.close()

print("All results saved to CSV files and plots in respective tau folders. Animation saved as 'fidelity_animation.mp4'.")