import numpy as np
import ridgeplot
import math
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clique_h_index(c, i, frac_red):
    """
    Compute the Coleman homophily index of a clique.

    Parameters:
    c (int): Number of nodes in the clique
    i (int): Number of red nodes in the clique
    frac_red (float): Fraction of red nodes globally

    Returns:
    float: Coleman homophily index of clique type-i (variant)
    """
    expin = frac_red**2 + (1 - frac_red)**2
    in_group = (math.comb(i, 2) + math.comb(c - i, 2)) / math.comb(c, 2)
    return (in_group - expin) / (1 - expin)

# Compute the maximum entropy clique distribution
def F_maximum_entropy(c, h, frac_red):
    """
    Compute the clique probability distribution vector F = [F_0,...,F_c].

    Parameters:
    c (int): Clique size
    h (float): Local homophily we want
    frac_red (float): Fraction of red nodes globally

    Returns:
    list: Clique probability distribution vector
    """
    def objective(vars):
        lam, theta = vars
        Z = sum([math.exp(theta * clique_h_index(c, i, frac_red) + i * lam) for i in range(c + 1)])
        F = [math.exp(theta * clique_h_index(c, i, frac_red) + i * lam) / Z for i in range(c + 1)]

        f1 = sum([i * F[i] for i in range(c + 1)]) - c * frac_red
        f2 = sum([clique_h_index(c, i, frac_red) * F[i] for i in range(c + 1)]) - h
        return f1**2 + f2**2

    initial_guess = [0., 0.]
    result = minimize(objective, initial_guess, bounds=[(-10, 10), (-10, 10)], tol=1e-9)

    lam_sol, theta_sol = result.x

    F = [math.exp(theta_sol * clique_h_index(c, i, frac_red) + i * lam_sol) for i in range(c + 1)]
    Z = sum(F)
    F = [[[i,Fi / Z] for (i,Fi) in enumerate(F)]]

    return F


hvals = np.linspace(1,-0.2, 15)
c= 4
nr = 0.7

# Generate data for various values of h
samples = []
for h in hvals:
    F = F_maximum_entropy(c, h, nr)
    samples.append(F)

# Convert hvals to strings for labels
labels = [str(round(h,2)) for h in hvals]
for (i,l) in enumerate(labels):
    if i%5!=0:
        labels[i]=''

# Create the ridge plot using ridgeplot library
fig = ridgeplot.ridgeplot(densities=samples, spacing=0.5, show_yticklabels=True,labels = labels,opacity=0.6,colorscale=['blue','red'],line_width = 0.2)
fig.update_layout(
    xaxis_title='number of red nodes',
    yaxis_title='$F_c$ for various $h$',
    showlegend=False,
    font_size=18,
    width=600,  # Set the width of the figure
    height=400,  # Set the height of the figure
    plot_bgcolor='white',  # Set the background color to white
    paper_bgcolor='white',  # Set the paper background color to white
    xaxis=dict(showgrid=False),  # Remove the x-axis grid
    yaxis=dict(showgrid=False)   # Remove the y-axis grid
)
fig.show()