import pandas as pd
import numpy as np
from qutip import Qobj, fidelity
import matplotlib.pyplot as plt
import time

# --- Parameters ---
tau = 144
N = 9
T = 2106
output_prefix = "noise"
np.random.seed(42)
num_iterations = 20    # Total number of successful runs desired

# --- Fidelity analysis function ---
def run_fidelity_analysis(closes):
    N, T = closes.shape
    phi_vectors = [tuple((closes[:, j] > closes[:, j - 1]).astype(int)) for j in range(1, T)]
    d = 2 ** N

    def vector_to_index(vec):
        return int("".join(map(str, vec)), 2)

    # Build density matrices
    density_matrices = []
    for t_start in range(len(phi_vectors) - tau + 1):
        window = phi_vectors[t_start:t_start + tau]
        counts = {}
        for vec in window:
            counts[vec] = counts.get(vec, 0) + 1
        P = np.zeros(d)
        for vec, count in counts.items():
            P[vector_to_index(vec)] = count / tau
        if np.sum(P) == 0 or np.any(np.isnan(P)) or np.any(np.isinf(P)):
            psi = np.ones((d, 1)) / np.sqrt(d)  # fallback to uniform state
        else:
            psi = np.sqrt(P).reshape(-1, 1)
            psi /= np.linalg.norm(psi)  # Normalize
        rho = psi @ psi.T
        density_matrices.append(rho)

    qobjs = [Qobj(rho, dims=[[2] * N, [2] * N]) for rho in density_matrices]
    rho0_qobj = qobjs[0]

    fidelity_full = []
    for idx, rho_t in enumerate(qobjs):
        try:
            f = fidelity(rho0_qobj, rho_t)
        except Exception as e:
            print(f"[ERROR] Fidelity computation failed at t={idx}: {e}")
            f = np.nan
        fidelity_full.append(f)

    return np.array(fidelity_full)

# --- Run multiple iterations ---
fidelity_matrix = []
successful_iterations = 0
attempts = 0
start_time = time.time()

print(f"▶ Starting {num_iterations} white noise fidelity simulations...\n")

while successful_iterations < num_iterations:
    attempts += 1
    print(f"Attempt {attempts}: Running iteration {successful_iterations + 1}/{num_iterations}...")

    try:
        closes = np.random.normal(loc=0, scale=1, size=(N, T))
        fidelity_full = run_fidelity_analysis(closes)

        if np.any(np.isnan(fidelity_full)) or np.any(np.isinf(fidelity_full)):
            raise ValueError("Fidelity contains NaN or Inf.")

        fidelity_matrix.append(fidelity_full)

        elapsed = time.time() - start_time
        avg_time = elapsed / (successful_iterations + 1)
        remaining = avg_time * (num_iterations - successful_iterations - 1)

        print(f"✔ Iteration {successful_iterations + 1}/{num_iterations} complete. "
              f"Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s")

        # --- Plot fidelity for this iteration ---
        plt.figure(figsize=(8, 3))
        plt.plot(fidelity_full, label=f"Fidelity Iter {successful_iterations + 1}", alpha=0.7)
        plt.ylim(0.95, 1.01)
        plt.title(f"Fidelity Curve - Iteration {successful_iterations + 1}")
        plt.xlabel("Time")
        plt.ylabel("Fidelity")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{output_prefix}_fidelity_iter_{successful_iterations + 1}.png")
        plt.close()

        # --- Plot white noise and fidelity only for first iteration ---
        if successful_iterations == 0:
            selected_idx = 2
            plt.figure(figsize=(10, 4))
            plt.plot(closes[selected_idx], label=f"White Noise C{selected_idx + 1}")
            plt.title(f"White Noise Data - Iter 1, C{selected_idx + 1}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_white_noise_C{selected_idx + 1}.png")
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.plot(fidelity_full, label="Full Fidelity (Iter 1)", color='gray')
            plt.ylim(max(0, fidelity_full.min() - 0.01), min(1.05, fidelity_full.max() + 0.01))
            plt.title("Full Fidelity Curve - Iteration 1")
            plt.xlabel("Time")
            plt.ylabel("Fidelity")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_plot_full_fidelity_iteration1.png")
            plt.close()

        successful_iterations += 1

    except Exception as e:
        print(f"[ERROR] Iteration {successful_iterations + 1} failed: {e}")
        print("Retrying with new data...\n")

# --- Convert to array ---
fidelity_matrix = np.array(fidelity_matrix)
fidelity_mean = fidelity_matrix.mean(axis=0)
fidelity_std = fidelity_matrix.std(axis=0)
time_axis = np.arange(len(fidelity_mean))

# --- Save average fidelity ---
pd.DataFrame({
    "time": time_axis,
    "mean_fidelity": fidelity_mean,
    "std_fidelity": fidelity_std
}).to_csv(f"{output_prefix}_fidelity_avg_over_{num_iterations}.csv", index=False)

# --- Plot average fidelity with error band ---
plt.figure(figsize=(10, 4))
plt.plot(time_axis, fidelity_mean, label="Mean Fidelity", color='blue')
plt.fill_between(time_axis, fidelity_mean - fidelity_std, fidelity_mean + fidelity_std,
                 alpha=0.3, color='blue', label="±1 std dev")
ymin = max(0, fidelity_mean.min() - fidelity_std.max() - 0.01)
ymax = min(1.05, fidelity_mean.max() + fidelity_std.max() + 0.01)
plt.ylim(ymin, ymax)
plt.title(f"Average Full Fidelity over {num_iterations} White Noise Runs")
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"{output_prefix}_plot_avg_fidelity_over_{num_iterations}.png")
plt.show()

# --- Save all fidelity runs ---
df_fidelities = pd.DataFrame(fidelity_matrix).T
df_fidelities.columns = [f"iter_{i}" for i in range(num_iterations)]
df_fidelities.insert(0, "time", time_axis)
df_fidelities.to_csv(f"{output_prefix}_fidelity_all_runs.csv", index=False)

print(f"✅ Averaged fidelity over {num_iterations} white noise runs saved and plotted.")
