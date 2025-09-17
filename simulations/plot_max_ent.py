import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


# Probability distributions
diamond = [0.14774453279301664, 0.07271677662869289, 0.0576034816859733, 0.07344356723902097, 0.150712649983881, 0.49777899166941525]
triangle = [0.4977789916694153, 0.15071264998388104, 0.07344356723902098, 0.05760348168597331, 0.0727167766286929, 0.14774453279301666]
square = [0.030937582332237125, 0.06932680881826901, 0.1300870848580512, 0.2044020692117611, 0.26893946007004965, 0.296306994709632]
circle = [0.29630699470963184, 0.2689394600700497, 0.20440206921176118, 0.1300870848580513, 0.06932680881826897, 0.03093758233223706]

# X-axis values
x = np.arange(6)

# Create a gradient of colors from blue to red
cmap = mcolors.LinearSegmentedColormap.from_list("gradient", ["blue", "red"])
colors = [cmap(i / 4) for i in range(4)]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each distribution with gradient-colored markers, dark grey lines, and black borders on markers
for i, val in enumerate(diamond):
    plt.plot(x[i], val, marker="D", markersize=18, color=colors[0], linestyle="dotted", alpha=1, linewidth=1.5, markeredgecolor="black")
plt.plot(x, diamond, linestyle="--", color=colors[0], linewidth=2)

for i, val in enumerate(triangle):
    plt.plot(x[i], val, marker="^", markersize=18, color=colors[1], linestyle="dotted", alpha=1, linewidth=1.5, markeredgecolor="black")
plt.plot(x, triangle, linestyle="--", color=colors[1], linewidth=2)

for i, val in enumerate(square):
    plt.plot(x[i], val, marker="s", markersize=18, color=colors[2], linestyle="--", alpha=1, linewidth=1.5, markeredgecolor="black")
plt.plot(x, square, linestyle="--", color=colors[2], linewidth=2)

for i, val in enumerate(circle):
    plt.plot(x[i], val, marker="o", markersize=18, color=colors[3], linestyle="--", alpha=1, linewidth=1.5, markeredgecolor="black")
plt.plot(x, circle, linestyle="--", color=colors[3], linewidth=2)

# Add labels with 50% larger font
plt.xlabel("Number of red nodes in the clique", fontsize=27)  # 50% larger
plt.ylabel("$\mathbf{F}_c$", fontsize=27)  # 50% larger
#plt.title("Probability Distributions", fontsize=30)  # 50% larger

# Custom legend
legend_elements = [
    plt.Line2D([0], [0], marker="^", color="black", markerfacecolor=colors[0], markersize=15, linestyle="none", label="$h=0.6$, $N_r$=0.3"),
    plt.Line2D([0], [0], marker="D", color="black", markerfacecolor=colors[1], markersize=15, linestyle="none", label="$h=0.6$, $N_r$=0.7"),
    plt.Line2D([0], [0], marker="s", color="black", markerfacecolor=colors[2], markersize=15, linestyle="none", label="$h=0.2$, $N_r$=0.7"),
    plt.Line2D([0], [0], marker="o", color="black", markerfacecolor=colors[3], markersize=15, linestyle="none", label="$h=0.2$, $N_r$=0.3")
]
plt.legend(handles=legend_elements, fontsize=21, loc="upper center")  # 50% larger font

# Show grid for better readability
plt.grid(alpha=0.3)

# Increase tick label size by 50%
plt.tick_params(axis="both", labelsize=24)  # 50% larger
plt.tight_layout()

# Save the figure as a PDF with 300 DPI
plt.savefig("maxent.pdf", dpi=300)

# Show the plot
plt.show()
