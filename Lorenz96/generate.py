import numpy as np
from scipy.integrate import odeint
import os
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(123)

# --- Set up parameters ---
N = 7  # Number of variables
times = np.arange(0, 100.1, 0.1)  # Time vector
n_times = len(times)

# --- Gamma values to iterate over ---
gamma_values = [0.1,0.15,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,1]

# --- Forcing matrix setup ---
# Constant forcing
F_constant = np.full((n_times, N), 8.0)

# Modulated forcing (time-dependent on x_4)
F_modulated = np.full((n_times, N), 8.0)
step_rise_start = 500
step_rise_end = 550
step_fall_end = 600

idx_rise_start = step_rise_start
idx_rise_end = step_rise_end
idx_fall_end = step_fall_end

F_modulated[idx_rise_start:idx_rise_end + 1, 3] = np.linspace(8, 90, idx_rise_end - idx_rise_start + 1)
idx_fall_start = idx_rise_end + 1
F_modulated[idx_fall_start:idx_fall_end + 1, 3] = np.linspace(90, 8, idx_fall_end - idx_fall_start + 1)
if idx_fall_end + 1 < n_times:
    F_modulated[idx_fall_end + 1:, 3] = 8

# --- Lorenz-96 model with nearest-neighbor diffusion ---
def lorenz96_mod(state, t, N, gamma, F_matrix, times):
    idx = np.argmin(np.abs(times - t))
    F_t = F_matrix[idx, :]
    dx = np.zeros(N)
    for i in range(N):
        ip1 = (i + 1) % N  # Next neighbor (right)
        im1 = (i - 1) % N  # Previous neighbor (left)
        im2 = (i - 2) % N  # Second left neighbor
        nonlinear = (state[ip1] - state[im2]) * state[im1] - state[i] + F_t[i]
        diffusive = gamma * (state[ip1] + state[im1] - 2 * state[i])
        dx[i] = nonlinear + diffusive
    return dx

# --- Run simulations for each gamma ---
n_realizations = 30

for gamma in gamma_values:
    # Create output folders
    output_folder_fixed = os.path.join(os.path.expanduser("~/Desktop/New folder (4)"), f"{gamma:.2f}-Lorenz96_FixedF4")
    output_folder_modulated = os.path.join(os.path.expanduser("~/Desktop/New folder (4)"), f"{gamma:.2f}-Lorenz96_ModulatedF4")
    
    os.makedirs(output_folder_fixed, exist_ok=True)
    os.makedirs(output_folder_modulated, exist_ok=True)
    
    print(f"Processing gamma = {gamma}")
    
    # Parameters for the model
    parms = {'N': N, 'gamma': gamma, 'times': times}
    
    # Run 20 realizations for both forcing types
    for realization in range(1, n_realizations + 1):
        # Set seed for initial conditions
        np.random.seed(100 + realization)
        state0 = 3 + np.random.randn(N)
        
        # Run simulation with constant forcing
        parms['F_matrix'] = F_constant
        out_fixed = odeint(lorenz96_mod, state0, times, args=(N, gamma, F_constant, times))
        
        # Save results (exclude time column)
        output_file_fixed = os.path.join(output_folder_fixed, f"lorenz96_realization_{realization}.txt")
        np.savetxt(output_file_fixed, out_fixed, fmt='%.6f')
        print(f"Saved realization {realization} to {output_file_fixed}")
        
        # Run simulation with modulated forcing
        parms['F_matrix'] = F_modulated
        out_modulated = odeint(lorenz96_mod, state0, times, args=(N, gamma, F_modulated, times))
        
        # Save results (exclude time column)
        output_file_modulated = os.path.join(output_folder_modulated, f"lorenz96_realization_{realization}.txt")
        np.savetxt(output_file_modulated, out_modulated, fmt='%.6f')
        print(f"Saved realization {realization} to {output_file_modulated}")

print("Simulations completed.")