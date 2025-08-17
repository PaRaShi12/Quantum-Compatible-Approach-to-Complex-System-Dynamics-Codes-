import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from qutip import Qobj, fidelity, ptrace, entropy_vn

# === Fidelity Analysis Function ===
def compute_fidelity_analysis(density_matrices_A, density_matrices_B, lag=0, is_partial=False):
    """
    Compute fidelity between density matrices of two datasets for a given lag.
    is_partial: Flag to indicate if the density matrices are for a single qubit (partial trace).
    """
    if len(density_matrices_A) == 0 or len(density_matrices_B) == 0:
        print(f"⚠️ Empty density matrix list for lag={lag}")
        return np.array([])

    if lag > 0:
        rho_A_trimmed = density_matrices_A[:-lag]
        rho_B_trimmed = density_matrices_B[lag:]
    elif lag < 0:
        lag_abs = abs(lag)
        rho_A_trimmed = density_matrices_A[lag_abs:]
        rho_B_trimmed = density_matrices_B[:-lag_abs]
    else:
        rho_A_trimmed = density_matrices_A
        rho_B_trimmed = density_matrices_B

    if len(rho_A_trimmed) != len(rho_B_trimmed):
        print(f"⚠️ Mismatched lengths after lag adjustment: {len(rho_A_trimmed)} vs {len(rho_B_trimmed)}")
        return np.array([])

    # Determine dimensions based on whether it's a partial trace
    dims = [[2], [2]] if is_partial else [[2] * N, [2] * N]

    # Convert to QuTiP Qobj for fidelity computation
    try:
        qobjs_A = [Qobj(rho, dims=dims) for rho in rho_A_trimmed]
        qobjs_B = [Qobj(rho, dims=dims) for rho in rho_B_trimmed]
    except Exception as e:
        print(f"⚠️ Qobj creation failed for lag={lag}: {e}")
        return np.array([np.nan] * len(rho_A_trimmed))

    fidelity_values = []
    for t, (rho_A, rho_B) in enumerate(zip(qobjs_A, qobjs_B)):
        try:
            f = fidelity(rho_A, rho_B)
            if np.isnan(f) or np.isinf(f) or f > 1.01:
                fidelity_values.append(np.nan)  # Exclude invalid fidelity
            else:
                fidelity_values.append(f)
        except Exception as e:
            print(f"⚠️ Fidelity computation failed at t={t}, lag={lag}: {e}")
            fidelity_values.append(np.nan)

    return np.array(fidelity_values)

# === Von Neumann Entropy Computation ===
def compute_vn_entropy(density_matrices, is_partial=True):
    """
    Compute von Neumann entropy for a list of density matrices.
    is_partial: Flag to indicate if the density matrices are for a single qubit.
    """
    dims = [[2], [2]] if is_partial else [[2] * N, [2] * N]
    entropy_values = []
    for rho in density_matrices:
        try:
            qobj = Qobj(rho, dims=dims)
            entropy = entropy_vn(qobj)
            if np.isnan(entropy) or np.isinf(entropy):
                entropy_values.append(np.nan)
            else:
                entropy_values.append(entropy)
        except Exception as e:
            print(f"⚠️ Entropy computation failed: {e}")
            entropy_values.append(np.nan)
    return np.array(entropy_values)

# === Density Matrix Computation ===
def compute_density_matrices(closes, tau):
    N, T = closes.shape
    phi_vectors = [tuple((closes[:, j] > closes[:, j - 1]).astype(int)) for j in range(1, T)]
    d = 2 ** N

    def vector_to_index(vec):
        return int("".join(map(str, vec)), 2)

    if not phi_vectors:
        print(f"No valid phi_vectors for shape {closes.shape}")
        return [np.ones((d, d)) / d] * (T - tau if T >= tau else 1)

    density_matrices = []
    for t_start in range(len(phi_vectors) - tau + 1):
        window = phi_vectors[t_start:t_start + tau]
        counts = Counter(window)
        P = np.zeros(d)
        for vec, count in counts.items():
            P[vector_to_index(vec)] = count / tau
        psi = np.sqrt(P).reshape(-1, 1)
        if np.sum(P) == 0 or np.any(np.isnan(P)) or np.any(np.isinf(P)):
            psi = np.ones((d, 1)) / np.sqrt(d)  # Fallback to uniform state
        else:
            psi /= np.linalg.norm(psi)  # Normalize
        rho = psi @ psi.T
        density_matrices.append(rho)
    return density_matrices

# === Settings ===
tau = 144
start_trim = 0
z = 0  # Starting realization index
num_iterations = 30  # Total realizations to process
forces = [ "0.20", "0.70", "0.60", "0.90", "1.50", "3.00"," 2.00"]  # Force levels to process
gammas = forces
# Dictionary to store average fidelities for fourth series
average_partial_fidelities = {force: [] for force in forces}

for force in forces:
    print(f"\n=== Processing force level {force} ===")
    all_fidelity_values = []  # Full system fidelity
    all_partial_fidelity_values = []  # Partial system fidelity (fourth series)
    all_entropy_A_values = []  # Entropy for modulated force
    all_entropy_B_values = []  # Entropy for fixed force

    while z < num_iterations:
        realization_id = z + 1
        realization_str = f"{realization_id}"
        print(f"\n=== Processing realization {realization_str} ===")

        # === File Paths ===
        modulated_dir = f"{force}-Lorenz96_ModulatedF4"
        fixed_dir = f"{force}-Lorenz96_FixedF4"
        file_name = f"Lorenz96_realization_{realization_str}.txt"
        file_path_A = os.path.join(modulated_dir, file_name)
        file_path_B = os.path.join(fixed_dir, file_name)

        # === Output Directory ===
        output_dir = f"trace_outputs_{force}/realization_{realization_str}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # === Load A ===
            df_A = pd.read_csv(file_path_A, sep=' ', header=None).dropna(axis=1, how='all')
            currency_names = [f"C{i+1}" for i in range(df_A.shape[1])]
            df_A.columns = currency_names
            closes_A = df_A.values.T[:, start_trim:]
            N, T = closes_A.shape
            print(f"Shape of df_A: {df_A.shape}, N: {N}, T: {T}")

            # === Load B ===
            df_B = pd.read_csv(file_path_B, sep=' ', header=None).dropna(axis=1, how='all')
            df_B.columns = currency_names
            closes_B = df_B.values.T[:, start_trim:]
            print(f"Shape of df_B: {df_B.shape}")
            assert closes_A.shape == closes_B.shape, "❌ Datasets must have the same shape"

            # === Compute Full Density Matrices ===
            density_matrices_A = compute_density_matrices(closes_A, tau)
            density_matrices_B = compute_density_matrices(closes_B, tau)
            print(f"Full density matrices A shape: {density_matrices_A[0].shape if density_matrices_A else 'empty'}")
            print(f"Full density matrices B shape: {density_matrices_B[0].shape if density_matrices_B else 'empty'}")
            assert len(density_matrices_A) == len(density_matrices_B), "❌ Time lengths mismatch"

            # === Compute Partial Density Matrices (Fourth Series) ===
            partial_density_matrices_A = []
            partial_density_matrices_B = []
            for rho_A, rho_B in zip(density_matrices_A, density_matrices_B):
                qobj_A = Qobj(rho_A, dims=[[2] * N, [2] * N])
                qobj_B = Qobj(rho_B, dims=[[2] * N, [2] * N])
                rho_A_partial = ptrace(qobj_A, 3).full()
                rho_B_partial = ptrace(qobj_B, 3).full()
                partial_density_matrices_A.append(rho_A_partial)
                partial_density_matrices_B.append(rho_B_partial)
            print(f"Partial density matrices A shape: {partial_density_matrices_A[0].shape if partial_density_matrices_A else 'empty'}")
            print(f"Partial density matrices B shape: {partial_density_matrices_B[0].shape if partial_density_matrices_B else 'empty'}")
            assert len(partial_density_matrices_A) == len(partial_density_matrices_B), "❌ Partial time lengths mismatch"

            # === Compute Von Neumann Entropy ===
            entropy_A = compute_vn_entropy(partial_density_matrices_A, is_partial=True)
            entropy_B = compute_vn_entropy(partial_density_matrices_B, is_partial=True)
            all_entropy_A_values.append(entropy_A)
            all_entropy_B_values.append(entropy_B)

            # === Plot Entropy ===
            plt.figure(figsize=(12, 6))
            valid_entropy_A = ~np.isnan(entropy_A)
            valid_entropy_B = ~np.isnan(entropy_B)
            plt.plot(np.arange(len(entropy_A))[valid_entropy_A], entropy_A[valid_entropy_A],
                     label="Modulated Force Entropy", color='tab:blue')
            plt.plot(np.arange(len(entropy_B))[valid_entropy_B], entropy_B[valid_entropy_B],
                     label="Fixed Force Entropy", color='tab:orange')
            plt.title(f"Von Neumann Entropy (4th Series, Force = {force})\nA = {file_name}, B = {file_name}")
            plt.xlabel("Time Step")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            entropy_plot_filename = os.path.join(output_dir, f"entropy_4th_series.png")
            plt.savefig(entropy_plot_filename)
            plt.close()
            print(f"📊 Saved entropy plot: {entropy_plot_filename}")

            # === Save Entropy CSV ===
            df_entropy = pd.DataFrame({
                "time_step": np.arange(len(entropy_A)),
                "entropy_modulated": entropy_A,
                "entropy_fixed": entropy_B
            })
            entropy_csv_filename = os.path.join(output_dir, f"entropy_4th_series.csv")
            df_entropy.to_csv(entropy_csv_filename, index=False)
            print(f"✅ Saved entropy CSV: {entropy_csv_filename}")

            # === Loop over lags ===
            lags_to_check = [0]
            for lag in lags_to_check:
                print(f"\n🔄 Processing lag = {lag}...")

                # --- Compute Full Fidelity ---
                fidelity_values = compute_fidelity_analysis(density_matrices_A, density_matrices_B, lag, is_partial=False)
                all_fidelity_values.append(fidelity_values)
                valid_fidelity = ~np.isnan(fidelity_values)

                # --- Plot Full Fidelity ---
                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(len(fidelity_values))[valid_fidelity], fidelity_values[valid_fidelity],
                         label=f"Fidelity, Lag = {lag}", color='tab:blue')
                plt.title(f"Full System Fidelity (Lag = {lag}, Force = {force})\nA = {file_name}, B = {file_name}")
                plt.xlabel("Time Step")
                plt.ylabel("Fidelity")
                plt.ylim(max(0, fidelity_values[valid_fidelity].min() - 0.01) if valid_fidelity.any() else 0,
                         min(1.05, fidelity_values[valid_fidelity].max() + 0.01) if valid_fidelity.any() else 1)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plot_filename = os.path.join(output_dir, f"fidelity_lag_{lag:+}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"📊 Saved full system fidelity plot: {plot_filename}")

                # --- Save Full Fidelity CSV ---
                df_fidelity = pd.DataFrame({
                    "time_step": np.arange(len(fidelity_values)),
                    "fidelity": fidelity_values
                })
                csv_filename = os.path.join(output_dir, f"fidelity_lag_{lag:+}.csv")
                df_fidelity.to_csv(csv_filename, index=False)
                print(f"✅ Saved full system fidelity CSV: {csv_filename}")

                # --- Compute Partial Fidelity (Fourth Series) ---
                partial_fidelity_values = compute_fidelity_analysis(partial_density_matrices_A, partial_density_matrices_B, lag, is_partial=True)
                all_partial_fidelity_values.append(partial_fidelity_values)
                valid_partial_fidelity = ~np.isnan(partial_fidelity_values)

                # Store average partial fidelity for this realization
                if valid_partial_fidelity.any():
                    average_partial_fidelities[force].append(np.nanmean(partial_fidelity_values))

                # --- Plot Partial Fidelity ---
                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(len(partial_fidelity_values))[valid_partial_fidelity], partial_fidelity_values[valid_partial_fidelity],
                         label=f"Partial Fidelity (4th Series), Lag = {lag}", color='tab:orange')
                plt.title(f"Partial System Fidelity (4th Series, Lag = {lag}, Force = {force})\nA = {file_name}, B = {file_name}")
                plt.xlabel("Time Step")
                plt.ylabel("Fidelity")
                plt.ylim(max(0, partial_fidelity_values[valid_partial_fidelity].min() - 0.01) if valid_partial_fidelity.any() else 0,
                         min(1.05, partial_fidelity_values[valid_fidelity].max() + 0.01) if valid_partial_fidelity.any() else 1)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                partial_plot_filename = os.path.join(output_dir, f"partial_fidelity_lag_{lag:+}.png")
                plt.savefig(partial_plot_filename)
                plt.close()
                print(f"📊 Saved partial fidelity plot: {partial_plot_filename}")

                # --- Save Partial Fidelity CSV ---
                df_partial_fidelity = pd.DataFrame({
                    "time_step": np.arange(len(partial_fidelity_values)),
                    "partial_fidelity": partial_fidelity_values
                })
                partial_csv_filename = os.path.join(output_dir, f"partial_fidelity_lag_{lag:+}.csv")
                df_partial_fidelity.to_csv(partial_csv_filename, index=False)
                print(f"✅ Saved partial fidelity CSV: {partial_csv_filename}")

        except Exception as e:
            print(f"\n❌ Fatal error in realization {realization_str} for force {force}: {e}")
            z += 1
            continue

        z += 1

    # === Compute and Plot Average Fidelity and Entropy ===
    if all_fidelity_values:
        # Full system fidelity
        max_length = max(len(fv) for fv in all_fidelity_values)
        padded_fidelity = [np.pad(fv, (0, max_length - len(fv)), constant_values=np.nan) for fv in all_fidelity_values]
        fidelity_array = np.array(padded_fidelity)
        avg_fidelity = np.nanmean(fidelity_array, axis=0)
        valid_avg_fidelity = ~np.isnan(avg_fidelity)

        # Plot average full fidelity
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(avg_fidelity))[valid_avg_fidelity], avg_fidelity[valid_avg_fidelity],
                 label=f"Average Fidelity, Lag = {lags_to_check[0]}", color='tab:green')
        plt.title(f"Average Full System Fidelity Across {num_iterations - 1} Realizations (Lag = {lags_to_check[0]}, Force = {force})")
        plt.xlabel("Time Step")
        plt.ylabel("Average Fidelity")
        plt.ylim(max(0, avg_fidelity[valid_avg_fidelity].min() - 0.01) if valid_avg_fidelity.any() else 0,
                 min(1.05, avg_fidelity[valid_avg_fidelity].max() + 0.01) if valid_avg_fidelity.any() else 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"trace_outputs_{force}", exist_ok=True)
        avg_plot_filename = os.path.join(f"trace_outputs_{force}", f"average_fidelity_lag_{lags_to_check[0]:+}.png")
        plt.savefig(avg_plot_filename)
        plt.close()
        print(f"📊 Saved average full fidelity plot: {avg_plot_filename}")

        # Save average full fidelity CSV
        df_avg_fidelity = pd.DataFrame({
            "time_step": np.arange(len(avg_fidelity)),
            "average_fidelity": avg_fidelity
        })
        avg_csv_filename = os.path.join(f"trace_outputs_{force}", f"average_fidelity_lag_{lags_to_check[0]:+}.csv")
        df_avg_fidelity.to_csv(avg_csv_filename, index=False)
        print(f"✅ Saved average full fidelity CSV: {avg_csv_filename}")

        # Partial system fidelity
        max_length_partial = max(len(fv) for fv in all_partial_fidelity_values)
        padded_partial_fidelity = [np.pad(fv, (0, max_length_partial - len(fv)), constant_values=np.nan) for fv in all_partial_fidelity_values]
        partial_fidelity_array = np.array(padded_partial_fidelity)
        avg_partial_fidelity = np.nanmean(partial_fidelity_array, axis=0)
        valid_avg_partial_fidelity = ~np.isnan(avg_partial_fidelity)

        # Plot average partial fidelity
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(avg_partial_fidelity))[valid_avg_partial_fidelity], avg_partial_fidelity[valid_avg_partial_fidelity],
                 label=f"Average Partial Fidelity (4th Series), Lag = {lags_to_check[0]}", color='tab:purple')
        plt.title(f"Average Partial System Fidelity (4th Series) Across {num_iterations - 1} Realizations (Lag = {lags_to_check[0]}, Force = {force})")
        plt.xlabel("Time Step")
        plt.ylabel("Average Fidelity")
        plt.ylim(max(0, avg_partial_fidelity[valid_avg_partial_fidelity].min() - 0.01) if valid_avg_partial_fidelity.any() else 0,
                 min(1.05, avg_partial_fidelity[valid_avg_fidelity].max() + 0.01) if valid_avg_partial_fidelity.any() else 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        avg_partial_plot_filename = os.path.join(f"trace_outputs_{force}", f"average_partial_fidelity_lag_{lags_to_check[0]:+}.png")
        plt.savefig(avg_partial_plot_filename)
        plt.close()
        print(f"📊 Saved average partial fidelity plot: {avg_partial_plot_filename}")

        # Save average partial fidelity CSV
        df_avg_partial_fidelity = pd.DataFrame({
            "time_step": np.arange(len(avg_partial_fidelity)),
            "average_partial_fidelity": avg_partial_fidelity
        })
        avg_partial_csv_filename = os.path.join(f"trace_outputs_{force}", f"average_partial_fidelity_lag_{lags_to_check[0]:+}.csv")
        df_avg_partial_fidelity.to_csv(avg_partial_csv_filename, index=False)
        print(f"✅ Saved average partial fidelity CSV: {avg_partial_csv_filename}")

        # Average entropy for modulated force
        max_length_entropy_A = max(len(ev) for ev in all_entropy_A_values)
        padded_entropy_A = [np.pad(ev, (0, max_length_entropy_A - len(ev)), constant_values=np.nan) for ev in all_entropy_A_values]
        entropy_A_array = np.array(padded_entropy_A)
        avg_entropy_A = np.nanmean(entropy_A_array, axis=0)
        valid_avg_entropy_A = ~np.isnan(avg_entropy_A)

        # Average entropy for fixed force
        max_length_entropy_B = max(len(ev) for ev in all_entropy_B_values)
        padded_entropy_B = [np.pad(ev, (0, max_length_entropy_B - len(ev)), constant_values=np.nan) for ev in all_entropy_B_values]
        entropy_B_array = np.array(padded_entropy_B)
        avg_entropy_B = np.nanmean(entropy_B_array, axis=0)
        valid_avg_entropy_B = ~np.isnan(avg_entropy_B)

        # Plot average entropy
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(avg_entropy_A))[valid_avg_entropy_A], avg_entropy_A[valid_avg_entropy_A],
                 label=f"Average Entropy (Modulated, 4th Series)", color='tab:blue')
        plt.plot(np.arange(len(avg_entropy_B))[valid_avg_entropy_B], avg_entropy_B[valid_avg_entropy_B],
                 label=f"Average Entropy (Fixed, 4th Series)", color='tab:orange')
        plt.title(f"Average Von Neumann Entropy (4th Series) Across {num_iterations - 1} Realizations (Force = {force})")
        plt.xlabel("Time Step")
        plt.ylabel("Average Entropy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        avg_entropy_plot_filename = os.path.join(f"trace_outputs_{force}", f"average_entropy_4th_series.png")
        plt.savefig(avg_entropy_plot_filename)
        plt.close()
        print(f"📊 Saved average entropy plot: {avg_entropy_plot_filename}")

        # Save average entropy CSV
        df_avg_entropy = pd.DataFrame({
            "time_step": np.arange(len(avg_entropy_A)),
            "average_entropy_modulated": avg_entropy_A,
            "average_entropy_fixed": avg_entropy_B
        })
        avg_entropy_csv_filename = os.path.join(f"trace_outputs_{force}", f"average_entropy_4th_series.csv")
        df_avg_entropy.to_csv(avg_entropy_csv_filename, index=False)
        print(f"✅ Saved average entropy CSV: {avg_entropy_csv_filename}")

    z = 1  # Reset realization index for the next force level

print(f"\n✅ Completed processing {num_iterations - 1} realizations for force levels {', '.join(forces)}.")

# === Combined Plotting Section ===
plt.rcParams["text.usetex"] = True

# === Settings ===
#gammas = [ "0.10"]
lag = 0
output_dir = "combined_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Load and Plot Average Fidelities and Entropies ===
plt.figure(figsize=(10, 5))
all_fidelity_data = {}
all_partial_fidelity_data = {}
all_entropy_A_data = {}
all_entropy_B_data = {}
min_fidelity, max_fidelity = 1.0, 0.0
min_partial_fidelity, max_partial_fidelity = 1.0, 0.0
min_entropy, max_entropy = np.inf, -np.inf

for gamma in gammas:
    # Load the average full fidelity CSV
    csv_filename = os.path.join(f"trace_outputs_{gamma}", f"average_fidelity_lag_{lag:+}.csv")
    if not os.path.exists(csv_filename):
        print(f"⚠️ Average full fidelity CSV not found for γ={gamma}: {csv_filename}")
        continue

    try:
        df = pd.read_csv(csv_filename)
        time_steps = df["time_step"].values
        avg_fidelity = df["average_fidelity"].values
        valid_fidelity = ~np.isnan(avg_fidelity)

        all_fidelity_data[gamma] = avg_fidelity
        if valid_fidelity.any():
            min_fidelity = min(min_fidelity, np.min(avg_fidelity[valid_fidelity]))
            max_fidelity = max(max_fidelity, np.max(avg_fidelity[valid_fidelity]))

        plt.plot(time_steps[valid_fidelity], avg_fidelity[valid_fidelity],
                 label=f"$\\gamma={gamma}$ (Full)",
                 linewidth=2.5, alpha=0.8)
    except Exception as e:
        print(f"❌ Error loading or plotting full fidelity data for γ={gamma}: {e}")

    # Load the average partial fidelity CSV
    partial_csv_filename = os.path.join(f"trace_outputs_{gamma}", f"average_partial_fidelity_lag_{lag:+}.csv")
    if not os.path.exists(partial_csv_filename):
        print(f"⚠️ Average partial fidelity CSV not found for γ={gamma}: {partial_csv_filename}")
        continue

    try:
        df_partial = pd.read_csv(partial_csv_filename)
        time_steps_partial = df_partial["time_step"].values
        avg_partial_fidelity = df_partial["average_partial_fidelity"].values
        valid_partial_fidelity = ~np.isnan(avg_partial_fidelity)

        all_partial_fidelity_data[gamma] = avg_partial_fidelity
        if valid_partial_fidelity.any():
            min_partial_fidelity = min(min_partial_fidelity, np.min(avg_partial_fidelity[valid_partial_fidelity]))
            max_partial_fidelity = max(max_partial_fidelity, np.max(avg_partial_fidelity[valid_partial_fidelity]))

        plt.plot(time_steps_partial[valid_partial_fidelity], avg_partial_fidelity[valid_partial_fidelity],
                 label=f"$\\gamma={gamma}$ (4th Series)",
                 linewidth=1.5, linestyle='--', alpha=0.8)
    except Exception as e:
        print(f"❌ Error loading or plotting partial fidelity data for γ={gamma}: {e}")

    # Load the average entropy CSV
    entropy_csv_filename = os.path.join(f"trace_outputs_{gamma}", f"average_entropy_4th_series.csv")
    if not os.path.exists(entropy_csv_filename):
        print(f"⚠️ Average entropy CSV not found for γ={gamma}: {entropy_csv_filename}")
        continue

    try:
        df_entropy = pd.read_csv(entropy_csv_filename)
        time_steps_entropy = df_entropy["time_step"].values
        avg_entropy_A = df_entropy["average_entropy_modulated"].values
        avg_entropy_B = df_entropy["average_entropy_fixed"].values
        valid_entropy_A = ~np.isnan(avg_entropy_A)
        valid_entropy_B = ~np.isnan(avg_entropy_B)

        all_entropy_A_data[gamma] = avg_entropy_A
        all_entropy_B_data[gamma] = avg_entropy_B
        if valid_entropy_A.any():
            min_entropy = min(min_entropy, np.min(avg_entropy_A[valid_entropy_A]))
            max_entropy = max(max_entropy, np.max(avg_entropy_A[valid_entropy_A]))
        if valid_entropy_B.any():
            min_entropy = min(min_entropy, np.min(avg_entropy_B[valid_entropy_B]))
            max_entropy = max(max_entropy, np.max(avg_entropy_B[valid_entropy_B]))
    except Exception as e:
        print(f"❌ Error loading entropy data for γ={gamma}: {e}")

# === Finalize Combined Fidelity Plot ===
plt.title("Average Fidelity Across Realizations (Full and Partial)", fontsize=14, pad=10)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Average Fidelity", fontsize=12)
plt.ylim(max(0, min(min_fidelity, min_partial_fidelity) - 0.02),
         min(1.05, max(max_fidelity, max_partial_fidelity) + 0.02))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='best')
plt.tick_params(axis='both', labelsize=10)
plt.tight_layout(pad=1.0)

plot_filename = os.path.join(output_dir, f"combined_average_fidelity_lag_{lag:+}.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 Saved combined fidelity plot: {plot_filename}")

# === Combined Entropy Plot ===
plt.figure(figsize=(10, 5))
for gamma in gammas:
    if gamma in all_entropy_A_data:
        avg_entropy_A = all_entropy_A_data[gamma]
        valid_entropy_A = ~np.isnan(avg_entropy_A)
        plt.plot(np.arange(len(avg_entropy_A))[valid_entropy_A], avg_entropy_A[valid_entropy_A],
                 label=f"$\\gamma={gamma}$ (Modulated)", linewidth=2.5, alpha=0.8)
    if gamma in all_entropy_B_data:
        avg_entropy_B = all_entropy_B_data[gamma]
        valid_entropy_B = ~np.isnan(avg_entropy_B)
        plt.plot(np.arange(len(avg_entropy_B))[valid_entropy_B], avg_entropy_B[valid_entropy_B],
                 label=f"$\\gamma={gamma}$ (Fixed)", linewidth=1.5, linestyle='--', alpha=0.8)

plt.title("Average Von Neumann Entropy (4th Series) Across Realizations", fontsize=14, pad=10)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Average Entropy", fontsize=12)
plt.ylim(max(0, min_entropy - 0.02), max_entropy + 0.02)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='best')
plt.tick_params(axis='both', labelsize=10)
plt.tight_layout(pad=1.0)

entropy_plot_filename = os.path.join(output_dir, f"combined_average_entropy_4th_series.png")
plt.savefig(entropy_plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 Saved combined entropy plot: {entropy_plot_filename}")

# === Save Combined CSV ===
max_length = max(len(data) for data in list(all_fidelity_data.values()) + list(all_partial_fidelity_data.values()) + list(all_entropy_A_data.values()) + list(all_entropy_B_data.values()))
combined_data = {"time_step": np.arange(max_length)}
for gamma, data in all_fidelity_data.items():
    padded_data = np.pad(data, (0, max_length - len(data)), constant_values=np.nan)
    combined_data[f"average_fidelity_gamma_{gamma}"] = padded_data
for gamma, data in all_partial_fidelity_data.items():
    padded_data = np.pad(data, (0, max_length - len(data)), constant_values=np.nan)
    combined_data[f"average_partial_fidelity_gamma_{gamma}"] = padded_data
for gamma, data in all_entropy_A_data.items():
    padded_data = np.pad(data, (0, max_length - len(data)), constant_values=np.nan)
    combined_data[f"average_entropy_modulated_gamma_{gamma}"] = padded_data
for gamma, data in all_entropy_B_data.items():
    padded_data = np.pad(data, (0, max_length - len(data)), constant_values=np.nan)
    combined_data[f"average_entropy_fixed_gamma_{gamma}"] = padded_data

df_combined = pd.DataFrame(combined_data)
csv_filename = os.path.join(output_dir, f"combined_average_fidelity_entropy_4th_series.csv")
df_combined.to_csv(csv_filename, index=False)
print(f"✅ Saved combined CSV: {csv_filename}")

# === Print Average Partial Fidelities ===
print("\n=== Average Partial Fidelities (4th Series) Across Realizations ===")
for force in forces:
    if average_partial_fidelities[force]:
        mean_fidelity = np.nanmean(average_partial_fidelities[force])
        print(f"Force {force}: {mean_fidelity:.4f}")
    else:
        print(f"Force {force}: No valid fidelity data")