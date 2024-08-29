import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random, os, math, itertools, time, pickle
from datetime import datetime
from joblib import Parallel, delayed
from networkx.utils import py_random_state
from networkx.algorithms import community
from scipy.optimize import fsolve, minimize

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(timestamp)

# Compute the homophily associated with a clique of size c with i red nodes.
def h_min(c, nr):
    nb = 1 - nr
    return ((c - 2) / (2 * c - 2) - nr**2 - nb**2) / (1 - nr**2 - nb**2)


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
    F = [Fi / Z for Fi in F]

    return F

def elgent_graphs(N, Nr, h2, h, ave_degree, alpha = 0.5, c = 6, c2 = 2):
    if h < h_min(c, Nr/N) or h2 < h_min(c2, Nr/N):
        raise ValueError("this configuration is impossible")           
    M = int(ave_degree * N / ((1-alpha)*(c2 * (c2 - 1))+(alpha*c * (c - 1))))

    F = F_maximum_entropy(c, h, Nr/N) 
    if len(F) != c + 1:
        raise ValueError("F must have a length of c + 1")
    else:
        F = np.array(F, dtype=float)
        F /= np.sum(F)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Randomly choose red nodes
    red_nodes = np.random.choice(N, size=Nr, replace=False)
    blue_nodes = np.setdiff1d(np.arange(N), red_nodes)

    red_nodes_set = set(red_nodes)
    blue_nodes_set = set(blue_nodes)

    for node in G.nodes:
        G.nodes[node]['color'] = 'red' if node in red_nodes_set else 'blue'

    # Sample cliques according to F
    chosen_nodes = []
    for _ in range(int(alpha*M)):
        # Determine the type of clique based on F
        clique_type = np.random.choice(c + 1, p=F)
        
        # Sample nodes for the clique
        red_sample = np.random.choice(red_nodes, clique_type, replace=False)
        blue_sample = np.random.choice(blue_nodes, c - clique_type, replace=False)
        clique_nodes = np.concatenate((red_sample, blue_sample))
        chosen_nodes.append(clique_nodes)
    
    # Generate edges
    edges = []
    for sublist in chosen_nodes:
        edges.extend(itertools.combinations(sublist, 2))
    G.add_edges_from(edges)
    #E1 = len(edges)

    F = F_maximum_entropy(c2, h2, Nr/N)
    chosen_nodes = []
    for _ in range(int((1-alpha)*M)):
        clique_type = np.random.choice(c2 + 1, p=F)
        red_sample = np.random.choice(red_nodes, clique_type, replace=False)
        blue_sample = np.random.choice(blue_nodes, c2 - clique_type, replace=False)
        clique_nodes = np.concatenate((red_sample, blue_sample))
        chosen_nodes.append(clique_nodes)
    
    # Generate edges
    edges2 = []
    for sublist in chosen_nodes:
        edges2.extend(itertools.combinations(sublist, 2))
    G.add_edges_from(edges2)
    #E2 = len(edges2)

    G.remove_edges_from(nx.selfloop_edges(G))

    #homophily = lambda h, h2, E1, E2: (E2*h2 + E1*h)/(E2 + E1)
    def homophily(alpha, c, h, c2, h2):
        numerator = alpha * c * (c - 1) * h + (1 - alpha) * c2 * (c2 - 1) * h2
        denominator = alpha * c * (c - 1) + (1 - alpha) * c2 * (c2 - 1)
        return numerator / denominator

    # Print statistics
    #print('Assortativity: ', calculate_assortativity(G))
    print('Total Homophily: ', homophily(alpha, c, h, c2, h2))

    #red_degrees = [d for n, d in G.degree() if G.nodes[n]['color'] == 'red']
   # blue_degrees = [d for n, d in G.degree() if G.nodes[n]['color'] == 'blue']
    #degrees = [d for n, d in G.degree()]
    #mean_degree = np.mean(degrees)
    #std_degree = np.std(degrees)
    
    #print('Size of the GC: ', gc_size(G))
    #print('Mean Degree: ', mean_degree, '±', std_degree)
    #print('Red nodes average degree: ', np.mean(red_degrees))
    #print('Blue nodes average degree: ', np.mean(blue_degrees))
    #if alpha == alpha_star(c):
        #print(E1, E2)
    return G


def alpha_star(c, c2=2):
    numerator = c2 * (c2 - 1)
    denominator = c2 * (c2 - 1) + c * (c - 1)
    return numerator / denominator

def calculate_assortativity(G):
    return nx.attribute_assortativity_coefficient(G, 'color')

def gc_size(g):
    largest_cc = max(nx.connected_components(g), key=len)
    return len(largest_cc) / g.number_of_nodes()

def components(g):
    return list(nx.connected_components(g))

def susceptibility(components):
    sizes = [len(component) for component in components]
    giant_size = max(sizes)
    sizes.remove(giant_size)
    if sizes:
        return np.mean(sizes)
    else:
        return 0

def label_links(g):
    edge_labels = {}
    for u, v in g.edges():
        if g.nodes[u]['color'] == 'red' and g.nodes[v]['color'] == 'red':
            edge_labels[(u, v)] = 'rr'
        elif g.nodes[u]['color'] == 'blue' and g.nodes[v]['color'] == 'blue':
            edge_labels[(u, v)] = 'bb'
        else:
            edge_labels[(u, v)] = 'rb'
    return edge_labels

def make_color_graph(g, Pi):
    edge_labels = label_links(g)
    rr_edges = [e for e in edge_labels if edge_labels[e] == 'rr']
    rb_edges = [e for e in edge_labels if edge_labels[e] == 'rb']
    bb_edges = [e for e in edge_labels if edge_labels[e] == 'bb']

    new_g = nx.Graph()
    new_g.add_nodes_from(g.nodes())

    sampled_rr_edges = random.sample(rr_edges, int(Pi[0] * len(rr_edges)))
    sampled_rb_edges = random.sample(rb_edges, int(Pi[1] * len(rb_edges)))
    sampled_bb_edges = random.sample(bb_edges, int(Pi[2] * len(bb_edges)))

    new_g.add_edges_from(sampled_rr_edges + sampled_rb_edges + sampled_bb_edges)
    return new_g


def color_percolation(g, Pi_range, ens=10):
    def Gs(g, Pi_range):
        return [gc_size(make_color_graph(g, Pi)) for Pi in Pi_range]

    gcc = Parallel(n_jobs=-1)(delayed(Gs)(g, Pi_range) for _ in range(ens))
    gcc = np.array(gcc)
    return np.mean(gcc, axis=0), np.std(gcc, axis=0)

def sus(g, Pi_range, ens=10):
    def Ss(g, Pi_range):
        S = np.array([susceptibility(components(make_color_graph(g, Pi))) for Pi in Pi_range])
        S /= np.max(S) if np.max(S) != 0 else 1
        return S

    scc = Parallel(n_jobs=-1)(delayed(Ss)(g, Pi_range) for _ in range(ens))
    scc = np.array(scc)
    return np.mean(scc, axis=0)

N = 5*10**5
ave_degree = 6
Nr = int(.6*N)
combinations = [(-0.3, 0.3), (-0.3, 0.9), (0.0, 0.0), (0.0, 0.6), (0.6, 0.0), (0.6, 0.6)]
p_range = np.linspace(0, 1, 70)
p_rb, p_bb = 0, 0
Pi_range = [[p_rr, p_rb, p_bb] for p_rr in p_range]
ens = 20

# Initialize the plot with larger font sizes
plt.figure(figsize=(10, 8))
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize

# Color and linestyle arrays
colors = ['y', 'c', 'b', 'g', 'r', 'm']  # Yellow, Cyan, Blue, Green, Red, Magenta
linestyles = ['-.', ':', '-', '--', '-.', ':']  # Dot-dash, Dotted, Solid, Dashed, Dot-dash, Dotted
markers = ['o', 's', 'D', 'P', '^', 'v']  # Circle, Square, Diamond, Plus, Triangle, Inverted Triangle

# Loop over each combination of c and h
Data = []
for idx, (h2, h) in enumerate(combinations):
    G = elgent_graphs(N, Nr, h2, h, ave_degree, alpha=alpha_star(c=6))
    mean_gcc, std_gcc = color_percolation(G, Pi_range, ens)
    mean_sus = sus(G, Pi_range, ens)

    Data.append([G, mean_gcc, std_gcc, mean_sus])

    # Plot giant component size with unique color, linestyle, and marker
    plt.plot(p_range, mean_gcc, color=colors[idx], linestyle=linestyles[idx % len(linestyles)],
             marker=markers[idx % len(markers)], label=f'$h_2$={h2}, $h_6$={h}')
    plt.fill_between(p_range, mean_gcc - std_gcc, mean_gcc + std_gcc, color=colors[idx], alpha=0.2)
    
    # Plot susceptibility with different marker and linestyle
    plt.plot(p_range, mean_sus, color=colors[idx], linestyle=linestyles[(idx + 1) % len(linestyles)],
             marker=markers[(idx + 1) % len(markers)], alpha=0.7)

# Add a horizontal line for Nr/N (fraction of red nodes)
plt.plot(p_range, [Nr/N for _ in p_range], color='k', linestyle=":")
plt.text((p_range[0] + p_range[-1]) / 2, Nr/N, '$n_r$', color='k', 
         ha='center', va='bottom', fontsize=16)

# Set labels and title
plt.xlabel('$\pi_{rr}$')
plt.ylabel('GC  |  '+'$\chi$')
plt.legend()
plt.tight_layout()

# Create figs directory if it does not exist
if not os.path.exists('figs'):
    os.makedirs('figs')

# Create data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')

# Save the figure
fig_filename = os.path.join('figs', f'color_percolation_{timestamp}.pdf')
plt.savefig(fig_filename, bbox_inches='tight')
plt.show()

# Save the data using pickle
data_filename = os.path.join('data', f'color_percolation_data_{timestamp}.pkl')
with open(data_filename, 'wb') as f:
    pickle.dump(Data, f)

print('done', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))