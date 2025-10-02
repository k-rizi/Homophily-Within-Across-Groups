import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import string

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Define the file paths for the pickle data files
file_paths = [
    'data/critical_p_rr_p_bb_0.1_N_4_0.55_prpbrat_1.npy',
    'data/critical_p_rr_p_bb_0.4_N_4_0.55_prpbrat_1.npy',
    'data/critical_p_rr_p_bb_0.5_N_4_0.54_prpbrat_1.npy'
]

# Initialize a list to store the data
data_list = []

# Homophily ranges
h1_values = np.linspace(-0.3, 0.9, 25)
h2_values = np.linspace(-0.3, 0.9, 25)


# Load the data files
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        data = np.load(file)
        data_list.append(data)
print(data)

# Create a figure and axes for the subplots
plt.rcParams.update({'font.size': 22})
fig, axes = plt.subplots(3, figsize=(4, 6),sharex=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()


# plt.rcParams.update({'font.size': 20})
# plt.figure(figsize=(4, 3))
# plt.imshow(data, origin='lower', extent=[h2_values[0], h2_values[-1], h1_values[0], h1_values[-1]],
#             aspect='auto', cmap='viridis')
# plt.colorbar(label='Critical $p_{rr}^*$')
# plt.xlabel('$h_2$')
# plt.ylabel('$h_4$')

# Determine the global min and max values across all datasets
vmin = min(data.min() for data in data_list)
vmax = max(data.max() for data in data_list)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.4, vmax=vmax)



# Plot each data set in a mesh plot
for ax, data in zip(axes, data_list):
    results = data
    mesh = ax.imshow(data, origin='lower', extent=[h2_values[0], h2_values[-1], h1_values[0], h1_values[-1]],
            aspect='auto', cmap='cividis',norm=norm)
    #ax.set_xlabel('$h_2$',fontsize=20)
    ax.set_ylabel('$h_4$',fontsize=20)



# Set the x-axis label for the bottom subplot
axes[-1].set_xlabel('$h_2$', fontsize=20)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0.1, 0, 0.7, 1])  # Adjust the right margin to make space for the colorbar


# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.25, right=0.65, top=0.9, bottom=0.15, wspace=0.4, hspace=0.1)


# Add a shared colorbar
cbar_ax = fig.add_axes([0.71, 0.18, 0.03, 0.7])  # [left, bottom, width, height]
cb = fig.colorbar(mesh, cax=cbar_ax, norm = norm, label='Critical $\pi_{rr}$')
cb.ax.set_yscale('linear')




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
output_file = os.path.join(output_dir, 'cont_area_plot_same_colorbar_h2h4.pdf')
plt.savefig(output_file)


plt.show()