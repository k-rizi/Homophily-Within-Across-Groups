import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve


from scipy import linalg


# Define the symbolic variables
p_success = sp.Symbol('p_success')
p_failure = sp.Symbol('p_failure')

# Define the symbolic variables
p_rr = sp.Symbol('p_rr')
p_bb = sp.Symbol('p_bb')
p_rb = sp.Symbol('p_rb')


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

def compute_probability_con_color(dr,db, p_rr, p_bb, p_rb, start_type):
    # Base case
    if dr+db==1:
        return sp.Integer(1)
    if start_type == 'r':
        i_r = 1
        i_b = 0
    else:
        i_r = 0
        i_b = 1

    # Compute the probability using the binomial distribution
    probability = 0
    for i in range(i_r, dr+1):
        for j in range(i_b, db+1):
            if i ==dr and j == db:
                return 1-probability
            probability += sp.binomial(dr-i_r, i-i_r) *sp.binomial(db-i_b, j-i_b) * compute_probability_con_color(i,j, p_rr, p_bb, p_rb, start_type) * ((1-p_rr) ** (i * (dr-i)))* ((1-p_bb) ** (j * (db-j)))* ((1-p_rb) ** (i*(db-j)+j* (dr-i)))
    probability = 1 - probability

    return probability


def compute_probability_con_color_uneq(dr,db,kr,kb, p_rr, p_bb, p_rb, start_type):
    if start_type == 'r':
        i_r = 1
        i_b = 0
    else:
        i_r = 0
        i_b = 1
    probability = sp.binomial(dr-i_r, kr-i_r) *sp.binomial(db-i_b, kb-i_b) * compute_probability_con_color(kr,kb, p_rr, p_bb, p_rb,start_type) * ((1-p_rr) ** (kr * (dr-kr)))* ((1-p_bb) ** (kb * (db-kb)))* ((1-p_rb) ** (kr*(db-kb)+kb*(dr-kr)))

    return probability

def compute_probability_con(d, p_success, p_failure):
    # Base case
    if d == 1:
        return 1

    # Compute the probability using the binomial distribution
    probability = 0
    for i in range(1, d):
        probability += sp.binomial(d-1, i-1) * compute_probability_con(i,  p_success, p_failure) * (p_failure ** (i * (d-i)))
    probability = 1 - probability

    return probability

def compute_probability_con_uneq(d, k, p_success, p_failure):
    # Compute the probability using the binomial distribution

    probability = sp.binomial(d-1, k-1)* compute_probability_con(k,  p_success, p_failure) * (p_failure ** (k * (d-k)))

    return probability

def clique_perc_avg(k, p_success, p_failure):
    # Compute the average connectivity of a clique after percolation
    avg = 0
    for i in range(2, k+1):
        avg  += (i-1)*compute_probability_con_uneq(k, i, p_success, p_failure)
    return avg

def clique_perc_avg_color(dr,db, p_rr, p_bb, p_rb,start_type, target_type):
    # Compute the average connectivity of a clique after percolation
    avg = 0
    if start_type == 'r':
        i_r = 1
        i_b = 0
    else:
        i_r = 0
        i_b = 1

    if target_type == 'r':
        i_r_t = 1
        i_b_t = 0
    else:
        i_r_t = 0
        i_b_t = 1

    for i in range(i_r, dr+1):
        for j in range(i_b,db+1):
            avg  += ((i-i_r)*i_r_t+(j-i_b)*i_b_t)*compute_probability_con_color_uneq(dr, db, i, j, p_rr, p_bb, p_rb,start_type)
    return avg

def F_biased(F):
    F_red = [0]*len(F)
    F_blue = [0]*len(F)
    for i in range(len(F)):
        F_red[i] += i*F[i]
        F_blue[i] += (len(F)-i-1)*F[i]
    #F_red = [F_red[i]/sum(F_red) for i in range(len(F_red))]
    #F_blue = [F_blue[i]/sum(F_blue) for i in range(len(F_blue))]
    return F_red, F_blue

# # Substitute the values of p_success and p_failure
# probability = compute_probability_con_uneq(3, 0.5,1, p_success, p_failure)
# probability = probability.subs(p_success, 0.5)
# probability = probability.subs(p_failure, 1-0.5)

# print(probability)

def B_matrix(F, F2,p_rr, p_bb, p_rb,N, Nr, M1, M2):
    c = len(F)-1
    c2 = len(F2)-1
    F_red,F_blue = F_biased(F)
    F_red2,F_blue2 = F_biased(F2)
    B = sp.ones(2*(c+4),2*(c+4))
    for i in range(c+1):
        for j in range(c+1):
            B[i,j] =M1 /Nr *F_red[j]*clique_perc_avg_color(i,c-i, p_rr, p_bb, p_rb,'r','r')
    for i in range(c+1,2*(c+1)):
        for j in range(c+1):
            B[i,j] =M1 /Nr *F_red[j]*clique_perc_avg_color(i%(c+1),c-i%(c+1), p_rr, p_bb, p_rb,'b','r')
    for i in range(c+1):
        for j in range(c+1,2*(c+1)):
            B[i,j] =M1 /(N-Nr) *F_blue[j%(c+1)]*clique_perc_avg_color(i,c-i%(c+1), p_rr, p_bb, p_rb,'r','b')
    for i in range(c+1,2*(c+1)):
        for j in range(c+1,2*(c+1)):
            B[i,j] =M1 /(N-Nr) *F_blue[j%(c+1)]*clique_perc_avg_color(i%(c+1),c-i%(c+1), p_rr, p_bb, p_rb,'b','b')

    for i in range(c+1):
        for j in range(2*(c+1),2*(c+1)+c2+1):
            B[i,j] =M2 /Nr *F_red2[j%(c+1)]*clique_perc_avg_color(i,c-i, p_rr, p_bb, p_rb,'r','r')
    for i in range(c+1):
        for j in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
            B[i,j] =M2 /(N-Nr) *F_blue2[j-2*(c+2)-1]*clique_perc_avg_color(i,c-i, p_rr, p_bb, p_rb,'r','b')
    for i in range(c+1,2*(c+1)):
        for j in range(2*(c+1),2*(c+1)+c2+1):
            B[i,j] =M2 /Nr *F_red2[j-2*(c+1)]*clique_perc_avg_color(i%(c+1),c-i%(c+1), p_rr, p_bb, p_rb,'b','r')
    for i in range(c+1,2*(c+1)):
        for j in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
            B[i,j] =M2 /(N-Nr) *F_blue2[j-2*(c+2)-1]*clique_perc_avg_color(i%(c+1),c-i%(c+1), p_rr, p_bb, p_rb,'b','b')

    for i in range(2*(c+1),2*(c+1)+c2+1):
        for j in range(c+1):
            B[i,j] =M1 /Nr *F_red[j]*clique_perc_avg_color(i%(c+1),c2-i%(c+1), p_rr, p_bb, p_rb,'r','r')
    for i in range(2*(c+1),2*(c+1)+c2+1):
        for j in range(c+1,2*(c+1)):
            B[i,j] =M1 /(N-Nr) *F_blue[j%(c+1)]*clique_perc_avg_color(i%(c+1),c2-i%(c+1), p_rr, p_bb, p_rb,'r','b')
    for i in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
        for j in range(c+1):
            B[i,j] =M1 /Nr *F_red[j]*clique_perc_avg_color(i-(2*(c+1)+c2+1),c-i+(2*(c+1)+c2+1), p_rr, p_bb, p_rb,'b','r')
    for i in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
        for j in range(c+1,2*(c+1)):
            B[i,j] =M1 /(N-Nr) *F_blue[j%(c+1)]*clique_perc_avg_color(i-(2*(c+1)+c2+1),c-i+(2*(c+1)+c2+1),  p_rr, p_bb, p_rb,'b','b')

    for i in range(2*(c+1),2*(c+1)+c2+1):
        for j in range(2*(c+1),2*(c+1)+c2+1):
            B[i,j] =M2 /Nr *F_red2[j%(c+1)]*clique_perc_avg_color(i%(c+1),c2-i%(c+1), p_rr, p_bb, p_rb,'r','r')
    for i in range(2*(c+1),2*(c+1)+c2+1):
        for j in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
            B[i,j] =M2 /(N-Nr) *F_blue2[j-2*(c+2)-1]*clique_perc_avg_color(i%(c+1),c2-i%(c+1), p_rr, p_bb, p_rb,'r','b')
    for i in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
        for j in range(2*(c+1),2*(c+1)+c2+1):
            B[i,j] =M2 /Nr *F_red2[j%(c+1)]*clique_perc_avg_color(i-(2*(c+1)+c2+1),c-i+(2*(c+1)+c2+1), p_rr, p_bb, p_rb,'b','r')
    for i in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
        for j in range(2*(c+1)+c2+1,2*(c+1)+2*(c2+1)):
            B[i,j] =M2 /(N-Nr) *F_blue2[j-2*(c+2)-1]*clique_perc_avg_color(i-(2*(c+1)+c2+1),c-i+(2*(c+1)+c2+1),  p_rr, p_bb, p_rb,'b','b')

    return B

def determinant_matrix(pb,B):
    mat = B.subs(p_bb,pb[0])
    temp = np.array(mat).astype(np.float64)
    return np.linalg.det(temp)

def alpha_star(c, c2=2):
    numerator = c2 * (c2 - 1)
    denominator = c2 * (c2 - 1) + c * (c - 1)
    return numerator / denominator

def determinant_matrix_red(pr,  B):
    mat = B.subs(p_rr,pr[0])
    temp = np.array(mat).astype(np.float64)
    return np.linalg.det(temp)


c1 = 6
c2 = 2

combinations = [(-0.3, 0.3), (-0.3, 0.9), (0.0, 0.0), (0.0, 0.6), (0.6, 0.0), (0.6, 0.6)]
prb_pbb_combinations = [(0.0, 0.0), (0.0, 0.2), (0.2, 0.0), (0.2, 0.2)]
N = 5*10**5
Nr = 0.6*N
ave_degree = 6
alpha = alpha_star(c1)
M = ave_degree * N / ((1-alpha)*(c2 * (c2 - 1))+(alpha*c1 * (c1 - 1)))


critical_probabilities = []
for (h2,h1) in combinations:
    F1 = F_maximum_entropy(c1, h1, Nr / N)
    F2 = F_maximum_entropy(c2, h2, Nr / N)
    for (prb,pbb) in prb_pbb_combinations:
        mat = B_matrix(F1, F2, p_rr, pbb, prb, N, Nr, alpha*M, (1-alpha)*M) - sp.eye(2 * (c1 + c2 + 2))
        root = fsolve(determinant_matrix_red, x0 = 0.2, args = mat)
        quality = determinant_matrix_red(root,mat)
        if root>0 and root<1 and abs(quality)<1e-10:
            critical_probabilities.append((root[0], h1, h2, prb, pbb))      #the critical probability p_rr (root[0]) for given h1, h2, p_bb, p_rb



# # Compute F1, F2, and prbvec for all combinations of h1 and h2
# for combination in combinations:
#     h1 = combination[0]
#     h2 = combination[1]

#     F1 = F_maximum_entropy(c1, h1, Nr / N)
#     F2 = F_maximum_entropy(c2, h2, Nr / N)
#     print(h1, h2, F1, F2)

#     mat = B_matrix(F1, F2, p_rr, p_bb, 0.2, N, Nr, alpha*M, (1-alpha)*M) - sp.eye(2 * (c1 + c2 + 2))

#     xs = [x * 0.02 for x in range(0, 50)]
#     xs.reverse()
#     prbvec = []
#     x_zero = 0.6
#     for pr in xs:
#         mat2 = mat.subs(p_rr, pr)
#         root = fsolve(determinant_matrix, x0 = x_zero, args = mat2)
#         quality = determinant_matrix(root,mat2)
#         x_zero = max(root[0],0)
#         if root>0 and root<1 and abs(quality)<1e-10:
#             prbvec.append((pr, root[0]))
#             print(pr, root[0], quality)
#         if abs(quality)>1e-10:
#             break
#         # for pb in (x * 0.02 for x in range(0, 50)):
#         #     mat2 = mat2.subs(p_bb, pb)
#         #     if abs(mat2.det()) < 0.01:
#         #         prbvec.append((pr, pb))
#         #         print(pr, pb, mat2.det())
#         #         break

#     # Plotting the entries in prbvec
#     pr_values = [pr for pr, pb in prbvec]
#     pb_values = [pb for pr, pb in prbvec]

#     plt.plot(pr_values, pb_values, label=f'h1={h1}, h2={h2}')

# # Finalize plot
# plt.xlabel('$p_{rr}$')
# plt.ylabel('$p_{bb}$')
# plt.grid(True)
# plt.legend()
# plt.show()

# h1vec = [0.1* x for x in range(0, 11)]
# h2vec = [0.1* x for x in range(0, 11)]


# probmat = np.zeros((len(h1vec),len(h2vec)))
# for (i,h1) in enumerate(h1vec):
#     F1 = F_maximum_entropy(c1, h1, Nr / N)
#     for (j,h2) in enumerate(h2vec):
#         F2 = F_maximum_entropy(c2, h2, Nr / N)
#         mat = B_matrix(F1, F2, p_bb, p_bb, 0.1, N, Nr, alpha*M, (1-alpha)*M) - sp.eye(2 * (c1 + c2 + 2))
#         root = fsolve(determinant_matrix, x0 = 0.2, args = mat)
#         quality = determinant_matrix(root,mat)
#         if root>0 and root<1 and abs(quality)<1e-10:
#             probmat[i,j] = root[0]
#             print(h1, h2, root[0], quality)

# # Plotting the heatmap
# plt.figure(figsize=(8, 6))
# plt.imshow(probmat, cmap='hot', interpolation='nearest', aspect='auto')
# plt.colorbar(label='Probability')
# plt.xlabel('$h_2$')
# plt.ylabel('$h_1$')
# plt.xticks(ticks=np.arange(len(h2vec)), labels=[f'{h2:.1f}' for h2 in h2vec])
# plt.yticks(ticks=np.arange(len(h1vec)), labels=[f'{h1:.1f}' for h1 in h1vec])
# plt.show()