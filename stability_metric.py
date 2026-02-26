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


# spectral radius
def spectral_radius(A):
    eigvals = np.linalg.eigvals(A)
    return np.max(np.abs(eigvals))

def hessian_vector_product(loss, theta, v):
    g = grad(loss)
    return grad(lambda th: anp.dot(g(th), v))(theta)

def stack_params_list(W_list):
    return anp.concatenate([W.reshape(-1) for W in W_list])

def unstack_params_list(theta, shapes):
    W_list = []
    offset = 0
    for (m, n) in shapes:
        size = m * n
        W = theta[offset:offset + size].reshape((m, n))
        W_list.append(W)
        offset += size
    return W_list

def node_loss(theta, shapes, Yi, Ai, tau):
    Wi = unstack_params_list(theta, shapes)

    # forward product X = W_L ... W_1
    X = Wi[0]
    for W in Wi[1:]:
        X = W @ X


    residual = Yi- Ai @ X

    data_term = anp.trace(residual.T @ residual)
    reg_term = tau * sum(anp.sum(W**2) for W in Wi)

    return data_term + reg_term


def mse_l2_hessian_w_max_val(Wi, Yi, Ai, tau, num_iters=50, tol=1e-6):
    shapes = [W.shape for W in Wi]
    theta = stack_params_list(Wi)

    def loss(th):
        return node_loss(th, shapes, Yi, Ai, tau)

    d = theta.size
    v = anp.random.randn(d)
    v /= anp.linalg.norm(v)


    Hv = hessian_vector_product(loss, theta, v)

    return lambda_new

# -------------------------------------------------------------------------------------------------- #



# parameters
n_agents = 10
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


eta = np.zeros(n_agents) # learning rates
c = anp.linspace(0.50, 0.90, 1)
etas = np.zeros((len(c), n_agents))
for const in range(len(c)):
    for i in range(n_agents):
        eta[i] = c[const] * ((1 + lambda_min_V) / lambdas_max[i]) # my bound
    etas[const, :] = eta



first_block = np.kron(V, np.identity(len(stack_params(W_list_star))))


# regularization coefficients
tau = 0.1 + (0.9 - 0.1) * anp.random.rand(n_agents)


# stability analysis
eta_min = 0.1
eta_max = 2.0
eta_star_0 = (1 + lambda_min_V) / (lambdas_max[0]) # my bound

N1, N2 = 10, 10

etas_left  = np.linspace(eta_min, eta_star_0, N1, endpoint=False)
etas_right = np.linspace(eta_star_0, eta_max, N2)
etas = np.concatenate([etas_left, etas_right])

# get the hessians



spect_rad = []
etas = np.linalg.inv(etas)
eta_among_nodes = np.linspace(eta_min, 0.5, n_agents, endpoint=False)
for j in range(etas):

    eta_among_nodes[0] = etas[j]
    D_etas = np.diag(eta_among_nodes)


    second_block = np.kron(D_etas, np.identity(len(stack_params(W_list_star))))
    spect_rad.append()







