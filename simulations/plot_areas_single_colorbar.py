import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


#Compare the difference in critical probabilities for networks with the same homophily $h$, but different local homophilies.
# Define the file paths for the pickle data files
file_paths = [
    'data/area_plot_prb_h_slice_0.5_alpha_0.14285714285714285_data_2024-11-14_11-35-29.pkl',
        'data/area_plot_prb_h_slice_1_alpha_0.14285714285714285_data_2024-11-14_11-57-00.pkl',
    'data/area_plot_prb_h_slice_0.5_alpha_0.5_data_2024-11-14_11-19-18.pkl',
    'data/area_plot_prb_h_slice_0.5_alpha_0.14285714285714285_uneq_data_2024-11-14_14-26-00.pkl'
]

# Initialize a list to store the data
data_list = []

# Load the pickle data files
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        data_list.append(data)

# Create a figure and axes for the subplots
plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Find the global min and max values for the colorbar normalization
cmap = plt.colormaps.get_cmap('coolwarm')
vmin = min(np.nanmin(data[2]) for data in data_list)
vmax = max(np.nanmax(data[2]) for data in data_list)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Plot each data set in a mesh plot
for ax, data in zip(axes, data_list):
    X,Y,results = data
    mesh = ax.contourf(X, Y, results.T, levels=100, cmap=cmap, norm=norm)
    ax.set_xlabel('$\pi_{rb}$',fontsize=20)
    ax.set_ylabel('$h$',fontsize=20)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to make space for the colorbar


# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.15, wspace=0.4, hspace=0.4)


# Add a shared colorbar
cbar_ax = fig.add_axes([0.81, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
fig.colorbar(mesh, cax=cbar_ax, label='Difference in Critical Probabilities')





# Finalize plot
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize

#plt.grid(True)

# Ensure the directory exists
output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)

# Save the figure
output_file = os.path.join(output_dir, 'cont_area_plot_same_colorbar.pdf')
plt.savefig(output_file)


plt.show()