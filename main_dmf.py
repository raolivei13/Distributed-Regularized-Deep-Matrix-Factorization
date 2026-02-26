import numpy as np
from numpy.linalg import svd, norm
from helper_functions_dmf import (metropolis_hastings_weights, generate_ground_truth,
                                  generate_sensing_ops, generate_feasible_dims,
                                  compute_U_vectors, back_mat_prod, sensing_apply, stack_params, unstack_params
                                  , low_rank_init)
from dmf_consensus_w_2reg_ import l2_norm_reg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from autograd import grad
import autograd.numpy as anp

# ------ Plotting stuff

# plotting parameters such that it looks in Latex style
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams.update({
    "text.usetex": True,                   # render all text with LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,            # proper minus sign in TeX mode
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}\usepackage{lmodern}"
})

# -------------------------------------------------------------------------------------------------- #
# computation of the hessian of the local loss evaluated at ground truth points
# or simply evaluated at arbitrary points in parameter space
# really this is only the second directional derivative
def compute_hessian_of_reg_loss(n_agents, Wstar, Ustar, A_sensing, tau):
    L = len(Wstar)
    hessians = [] # this is the second directional derivative

    for j in range(n_agents):
        sum_frob = np.zeros((d, d))
        sum_reg = np.zeros((d, d))
        for i in range(L):
            A = back_mat_prod(Wstar, i + 1, L - 1)
            B = back_mat_prod(Wstar, 0, i - 1)
            sum_frob += sensing_apply(A_sensing[j], np.matmul(A, np.matmul(Ustar[i], B)))
            sum_reg += norm(Ustar[i], 'fro')**2

        # compute here the directional second derivative
        hessians.append(2*norm(sum_frob, 'fro')**2 + tau[j]*sum_reg)

    return hessians # returns the hessians for each node

def eigen_max_of_local_loss(n_agents, Wstar, A_sensing, tau):
    L = len(Wstar)
    d = Wstar[0].shape[1]  # d0 = dL = d

    lambdas_max = []

    for j in range(n_agents):

        H_sum = anp.zeros((d*d, d*d))

        for i in range(L):
            # Products with variable dims
            A_i = back_mat_prod(Wstar, i+1, L-1)   # shape (d × d_i)
            B_i = back_mat_prod(Wstar, 0, i-1)    # shape (d_i × d)

            # Compute sensing inner block
            AiAiT = A_i @ A_i.T               # (d × d)
            G_i = A_sensing[j] @ AiAiT @ A_sensing[j].T  # (d × d)

            # Kronecker Hessian block
            H_sum += anp.kron(B_i.T @ B_i, G_i)

        lambdas_max.append(2*anp.linalg.norm(H_sum,2) + tau[j])

    return lambdas_max

# initialization near minimum
def initialization_near_min(W_star, U_star, n_nodes, r):
    shapes = [W.shape for W in W_star]

    # stack ground-truth parameters and direction
    w_star = stack_params(W_star)
    u_vec = stack_params(U_star)

    # normalize direction
    u_vec = u_vec / anp.linalg.norm(u_vec)

    init_W_nodes = []
    init_theta_nodes = []

    for _ in range(n_nodes):
        eps = r  # keep simple for now and give the same perturbation for each node
        theta0 = w_star + eps * u_vec          # this is the initialization near the minimum
        W0 = unstack_params(theta0, shapes)  # list-of-matrices init

        init_W_nodes.append(W0) # append the non-stacked matrices per node
        init_theta_nodes.append(theta0) # append the stacked matrices per node


    return init_W_nodes, init_theta_nodes # returns stacked and unstacked parameters

# check the singular value of the matrix factorization
def check_sing_val(W_list):

    def prod(W_list):
        X = W_list[0]
        for W in W_list[1:]:
            X = W @ X
        return X

    X_estimated = prod(W_list)
    _, sigma_i, _ = anp.linalg.svd(X_estimated, full_matrices=False)

    return sigma_i

# -------------------------------------------------------------------------------------------------- #


# parameters
n_agents = 20
max_iter = 100 # consensus iterations
T = 1 # consensus rounds
p = 0.8 # can change later
d = 10 # sensing dimension
L = 20 # depth of the factorization
rank = 5 # select parameter for low rank solution
low_rank = False
rank_init = 10 # initialize ground truth with low rank
# tau = np.zeros(n_agents) # regularization coefficients
tau = 0.1 + (0.9 - 0.1) * anp.random.rand(n_agents)
#tau = np.zeros(n_agents)
# for i in range(n_agents):
#     tau[i] = 0.6 # this can change and can impose different regularization coefficients per node



# generates sensing operators for each node
# some generation of the data
# here we assume that there exist at least one factorization of the form X_star = W_1_star ... W_L_star
A_sensing = generate_sensing_ops(d, n_agents)
dims = generate_feasible_dims(d, L, high_factor=3) # generates the feasible for each layer of the factorization
W_list_star, X_star, Y_list_star, sigma_prod = generate_ground_truth(dims, A_sensing, low_rank, rank) # generates global minimizers
U_star = compute_U_vectors(W_list_star) # compute the optimal directions



# call the max eigen value function
lambdas_max = eigen_max_of_local_loss(n_agents, W_list_star, A_sensing, tau)


# create graph
rng_graph = np.random.RandomState(0)
adj = (rng_graph.rand(n_agents, n_agents) < p).astype(int)
adj = np.triu(adj, 1)
adj = adj + adj.T
for i in range(n_agents):
    if adj[i].sum() == 0:
        # ensure at least one neighbor
        j = (i + 1) % n_agents
        adj[i, j] = adj[j, i] = 1

# create weights on edges
V = metropolis_hastings_weights(adj)
lambda_min_V = np.min(np.linalg.eigvalsh(V))
print(lambda_min_V)

# define the learning rate such that one is below the bound and one above
eta = np.zeros(n_agents) # learning rates
high = 2.0
low = 0.1
# const = low + (high - low) * anp.random.rand(n_agents)
# const = 0.9
# c = anp.linspace(1.50, 2.00, 3)
c = [0.90, 1.30, 1.50]
etas = np.zeros((len(c), n_agents))
for const in range(len(c)):
    for i in range(n_agents):
        eta[i] = c[const] * ((1 + lambda_min_V) / lambdas_max[i]) # my bound
    etas[const, :] = eta


# when initializing the parameters, we need to initialize them close to the minimizers
# basically we need to initialize the parameters extremely close to the
r = 1e-3 # super far


W0, _ = initialization_near_min(W_list_star, U_star, n_agents, r) # initialization for each node near ground truth
#W0 = low_rank_init(dims, n_agents, rank_init) # low-rank initialization which is slightly above the rank of the ground truth


# Proposition 4:  Learning dynamics of deep matrix factorization beyond the edge of stability
# Balanced Initialization


# call the algorithm here
norm_errors_per_c = []
consensus_error_total_c = []
sharpness_per_iters_c = []
for const in range(len(c)):
    W_final, normalized_error, consensus_error_total, lambdas_per_iter = l2_norm_reg(W0, Y_list_star, A_sensing, U_star,
                n_agents,
                max_iter, T, tau, etas[const, :], V, W_list_star, X_star, verbose=True)

    norm_errors_per_c.append(normalized_error)
    consensus_error_total_c.append(consensus_error_total)
    sharpness_per_iters_c.append(lambdas_per_iter)



fig, ax = plt.subplots(figsize=(11, 11))
# bound plot
for i in range(n_agents):
    # Plot first curve and grab its color
    line, = ax.plot(
        sharpness_per_iters_c[0][:, i],
        linewidth=10,
        label=rf"$\text{{node}} \ {i+1}$"
    )
    color = line.get_color()

    # Plot second curve with the SAME color
    ax.plot(
        (1 + lambda_min_V) / etas[0, :][i] * np.ones(max_iter),
        linestyle='--',
        linewidth=10,
        label=rf"$1+\lambda_{{\text{{min}}}}(\mathbf{{V}})/\eta_{i+1}$",
        color=color
    )
ax.set_xlabel(r"$t$ Iterations", fontsize=40)
ax.set_ylabel(r"$\lambda_{\text{max}}(\nabla^2\mathcal{L}_{i}(\boldsymbol{\omega}_{i}^{t}))$", fontsize=50)
ax.tick_params(axis='both', which='major', labelsize=60)
ax.yaxis.get_offset_text().set_fontsize(50)
ax.legend(loc="best", fontsize=40)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("./saved_plots/sharpness_plot.png", dpi=600, bbox_inches="tight")
plt.show()


# check the resulting singular values
# _, s, _ = anp.linalg.svd(X_star, full_matrices=False)
# print("This is the singular value associated to the ground truth matrix: ", s)
# sigmas = check_sing_val(W_final[0])
# print("These are the singular values of the estimated factorized matrix: ", sigmas)
# print("Normalized error between estimated and true factorization: ", np.linalg.norm(anp.diag(s) - anp.diag(sigmas), 'fro')**2 / np.linalg.norm(anp.diag(s), 'fro')**2)
#
# for i in range(L):
#     _, s_est, _ = anp.linalg.svd(W_final[0][i], full_matrices=False)
#     _, s_init, _ = anp.linalg.svd(W0[0][i], full_matrices=False)
#     print("Singular values of the individual estimated matrices: ", s_est)
#     print("Singular values of the initialized matrices: ", s_init)


# show consensus error curves across al five methods
# plt.figure(figsize=(11, 11))
#
# for const in range(len(c)):
#     plt.plot(norm_errors_per_c[const], linewidth=10, label = rf"$\eta = {c[const]:.2f}\,\frac{{1+\tilde{{\lambda}}_{{\min}}(\mathbf{{V}})}}{{\lambda_{{\max}}\left(\nabla^2 \mathcal{{L}}(\omega^\star_{{i}})\right)}}$")
# plt.xlabel("$t$ Iterations", fontsize=40)
# plt.ylabel("NMSE", fontsize=50)
# plt.tick_params(axis='both', which='major', labelsize=60)  # Makes the numbers larger
# plt.yaxis.get_offset_text().set_fontsize(16) # targets the scientific notation on top of the yaxis
# plt.legend(loc="best", fontsize = 50)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("./saved_plots/normalized_errors.png", dpi = 600, bbox_inches="tight")
# plt.show()

fig, ax = plt.subplots(figsize=(11, 11))
for const in range(len(c)):
    ax.plot(
        norm_errors_per_c[const],
        linewidth=10,
        label=rf"$\eta_{{i}} = {c[const]:.2f}\,\bar{{\eta_{{i}}}}$"
        #label=rf"$\eta = {c[const]:.2f}\,\frac{{1+\tilde{{\lambda}}_{{\min}}(\mathbf{{V}})}}{{\lambda_{{\max}}\left(\nabla^2 \mathcal{{L}}(\omega^\star_{{i}})\right)}}$"
    )
ax.set_xlabel(r"$t$ Iterations", fontsize=40)
ax.set_ylabel("NMSE", fontsize=50)
ax.tick_params(axis='both', which='major', labelsize=60)
ax.yaxis.get_offset_text().set_fontsize(50)
ax.legend(loc="best", fontsize=50)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("./saved_plots/normalized_errors.png", dpi=600, bbox_inches="tight")
plt.show()



# plt.figure(figsize=(11, 11))
#
# for const in range(len(c)):
#     plt.plot(consensus_error_total_c[const], linewidth=10, label = rf"$\eta = {c[const]:.2f}\,\frac{{1+\tilde{{\lambda}}_{{\min}}(\mathbf{{V}})}}{{\lambda_{{\max}}\left(\nabla^2 \mathcal{{L}}(\omega^\star_{{i}})\right)}}$")
# plt.xlabel("$t$ Iterations", fontsize=40)
# plt.ylabel("Total Consensus Error", fontsize=50)
# plt.tick_params(axis='both', which='major', labelsize=60)  # Makes the numbers larger
# plt.axis.get_offset_text().set_fontsize(16)
# plt.legend(loc="best", fontsize = 50)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("./saved_plots/consensus_errors.png", dpi = 600, bbox_inches="tight")
# plt.show()

fig, ax = plt.subplots(figsize=(11, 11))

for const in range(len(c)):
    ax.plot(
        consensus_error_total_c[const],
        linewidth=10,
        label=rf"$\eta_{{i}} = {c[const]:.2f}\,\bar{{\eta_{{i}}}}$"
    )
    # ax.plot(
    #     consensus_error_total_c[const],
    #     linewidth=10,
    #     label=rf"$\eta = {c[const]:.2f}\,\frac{{1+\tilde{{\lambda}}_{{\min}}(\mathbf{{V}})}}{{\lambda_{{\max}}\left(\nabla^2 \mathcal{{L}}(\omega^\star_{{i}})\right)}}$"
    # )
ax.set_xlabel(r"$t$ Iterations", fontsize=40)
ax.set_ylabel("Total Consensus Error", fontsize=50)
ax.tick_params(axis='both', which='major', labelsize=60)
ax.yaxis.get_offset_text().set_fontsize(50)
ax.legend(loc="best", fontsize=50)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("./saved_plots/consensus_errors.png", dpi=600, bbox_inches="tight")
plt.show()