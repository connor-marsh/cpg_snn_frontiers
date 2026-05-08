import numpy as np
import matplotlib.pyplot as plt

dir_prefix = "vishnu_multigait_stuff/"

# Professional Plotting Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "figure.figsize": (10, 12) # Taller figure for 3 rows
})

# Helper function to keep code clean
def plot_row(axs, row_idx, x_data, y_rmse, y_runtime, x_label):
    # Plot RMSE (Left Column)
    axs[row_idx, 0].plot(x_data, y_rmse[:, 0], marker='o', label='Pure-gait', color='#1f77b4', markerfacecolor='white')
    axs[row_idx, 0].plot(x_data, y_rmse[:, 1], marker='s', label='Transition', color='#d62728', markerfacecolor='white')
    axs[row_idx, 0].set_ylabel('RMSE')
    axs[row_idx, 0].set_xlabel(x_label)
    axs[row_idx, 0].grid(True, linestyle='--')
    
    # Plot Runtime (Right Column)
    axs[row_idx, 1].plot(x_data, y_runtime, marker='o', color='#1f77b4', markerfacecolor='white')
    axs[row_idx, 1].set_ylabel('Runtime (s)')
    axs[row_idx, 1].set_xlabel(x_label)
    axs[row_idx, 1].grid(True, linestyle='--')

# Defining what we are running analysis on
params = ["hidden", "seq_len", "beta", "n_layers"]
titles = ["Model Size", "Temporal Context", "Membrane Decay Rate", "Model Length"]
x_axis_labels = [r'Hidden Neurons ($N_h$)', r'Sequence Length ($L$)', r'Decay Factor ($\beta$)', r'Hidden Layers']
dir_prefix = "vishnu_multigait_stuff/"

params = [params[0]]

fig, axs = plt.subplots(len(params), 2, sharex=False)

# Data Loading
for i, param in enumerate(params):
    rmse_data = np.load(dir_prefix + "param_sweep_" + param + "/all_rmse_matrices.npy")
    av_rmse = np.mean(rmse_data, axis=(2, 3))
    run_times = np.load(dir_prefix + "param_sweep_" + param + "/run_time.npy")
    x_axis = np.load(dir_prefix + "param_sweep_" + param + "/x_axis.npy")
    plot_row(axs, i, x_axis, av_rmse, run_times, x_axis_labels[i])
    axs[i, 0].set_title('Performance vs. ' + titles[i])
    axs[i, 1].set_title('Runtime vs. ' + titles[i])
    axs[i, 0].legend()


plt.tight_layout()

# Save for paper
plt.savefig(dir_prefix+'full_parameter_sweep.pdf', bbox_inches='tight')
plt.show()