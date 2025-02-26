import pickle
import matplotlib.pyplot as plt
import numpy as np
import math, os
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Path to the pickle file
pickle_file_path = 'C:/Users/StegehuisC/surfdrive/Documents/Github/Weak-and-Strong-Homophily-in-Networks/Homophily-Within-Across-Groups/simulations/data/area_plot_prb_pbb_homval_0.5.pkl'

# Open the pickle file and load the data
with open(pickle_file_path, 'rb') as file:
    prb_values, p_bb_values, results = pickle.load(file)

# Now you can use prb_values, p_bb_values, and results
X, Y = np.meshgrid(prb_values, p_bb_values)

# Use a diverging colormap from ColorBrewer
cmap = plt.colormaps.get_cmap('cividis')
norm = TwoSlopeNorm(vmin=np.nanmin(results), vcenter=0, vmax=np.nanmax(results))

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(6, 6))

# Finalize plot
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=25)     # fontsize of the axes title
plt.rc('axes', labelsize=25)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize

plt.xlabel('$\pi_{rb}$')
plt.ylabel('$\pi_{bb}$')

# Plot the data
mesh = ax.pcolormesh(X, Y, results.T, cmap=cmap, norm=norm)

# Add a colorbar
cbar = fig.colorbar(mesh, ax=ax,label='Difference in $\pi_{rr}$')
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
cbar.ax.set_yscale('linear')

# Remove the right and upper parts of the box around the plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.95, 1])

# Save the figure
output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'improve_prr_pbb_plot.pdf')
plt.savefig(output_file)

plt.show()