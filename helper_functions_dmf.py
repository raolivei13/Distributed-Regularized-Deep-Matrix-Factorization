import numpy as np
from numpy.linalg import svd, norm
from autograd import grad
import autograd.numpy as anp


def generate_feasible_dims(d, L, high_factor=3):

    dims = [d]

    # Generate intermediate dims randomly satisfying di >= d
    max_width = int(high_factor * d)  # allow wider layers
    for _ in range(L-1):
        dims.append(anp.random.randint(d, max_width + 1))

    # Ensure final equals first (square)
    dims.append(d)

    return dims


# generates sensing operators
def generate_sensing_ops(d, n_agents, normalize=True):

    A_ops = []
    for _ in range(n_agents):
        Ai = anp.random.randn(d, d) # acts on the vectorized parameters
        if normalize:
            Ai /= anp.sqrt(d)
        A_ops.append(Ai)
    return A_ops

# ---- this is where the ground truth is generated ---- #
def generate_ground_truth_low_rank(dims, A_list, rank):

    def prod(W_list):
        X = W_list[0]
        for W in W_list[1:]:
            X = W @ X
        return X


    L = len(dims) - 1  # number of factors
    W_star_list = []

    # generate true factor matrices
    sigma = []
    Uj_list = []
    VjT_list = []
    for j in range(L):
        d_in, d_out = dims[j], dims[j + 1]
        W_j = anp.random.randn(d_out, d_in) # generate at random with appropriate dimensions
        Uj, sigma_j, VjT = anp.linalg.svd(W_j, full_matrices=False)

        W_star_list.append(Uj[:, :rank] @ anp.diag(sigma_j[:rank]) @ VjT[:rank, :]) # append the ground truth full matrices
        sigma.append(np.diag(sigma_j[:rank])) # append each of the singular values for each layer
        Uj_list.append(Uj[:, :rank]) # append each of the left singular values for each layer
        VjT_list.append(VjT[:rank, :]) # append each of the right singular values for each layer

    # Product of all singular values
    # and with the assumption that we have orthogonal subspaces
    if rank > 1:
        sigma_reduced = None
        # sigma_reduced = prod(sigma)
        # u_L = Uj_list[-1]  # first left vector of last layer
        # v_1T = VjT_list[0]  # first right vector of first layer
        #
        # # construct the ground truth low rank product
        # X_star = u_L @ sigma_reduced @ v_1T

        X_star = prod(W_star_list)

        # per-node measurements
        Y_list = [Ai @ X_star for Ai in A_list]

        return W_star_list, X_star, Y_list, sigma_reduced

    else:
        s_prod = 1.0
        for s in sigma:
            s_prod *= s

        u_L = Uj_list[-1]  # first left vector of last layer
        v_1T = VjT_list[0]  # first right vector of first layer

        # construct the ground truth low rank product
        X_star = s_prod * (u_L @ v_1T)

        # per-node measurements
        Y_list = [Ai @ X_star for Ai in A_list]

        return W_star_list, X_star, Y_list, s_prod




def generate_ground_truth(dims, A_list, low_rank, rank):



    if low_rank:
        W_star_list, X_star, Y_list, sigma_prod = generate_ground_truth_low_rank(dims, A_list, rank)
        return W_star_list, X_star, Y_list, sigma_prod
    else:
        L = len(dims) - 1                         # number of factors
        W_star_list = []

        # generate true factor matrices
        for j in range(L):
            d_in, d_out = dims[j], dims[j+1]
            W_star_list.append(anp.random.randn(d_out, d_in))

        # compute ground truth product X_star = W_L ... W_1
        X_star = W_star_list[0]
        for W in W_star_list[1:]:
            X_star = anp.dot(W, X_star)

        # per-node measurements
        Y_list = [Ai @ X_star for Ai in A_list]

        sigma_prod = None

        return W_star_list, X_star, Y_list, sigma_prod
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- #


def back_mat_prod(W_list, n, m):

    if n > m:
        d_m = W_list[m].shape[0] # outer dimension
        return anp.eye(d_m)

    # Otherwise construct W_m W_{m-1} ... W_n
    X = W_list[m]
    for j in range(m-1, n-1, -1):   # go backwards down to n
        X = X @ W_list[j]
    return X



def deep_product(W_list):
    return back_mat_prod(0, len(W_list) - 1, W_list)

def sensing_apply(A, X):
    # A is a linear operator and is symmetric
    AX = np.matmul(A, X)
    return AX

# local loss
def local_loss(W_list, A, Y, tau):
    # given the measurement Y
    X = deep_product(W_list)
    # squared loss plus weight decay
    R = norm(sensing_apply(A, X) - Y, 'fro')**2 + tau*norm(X, 'fro')**2
    return R

def metropolis_hastings_weights(adj):
    n = adj.shape[0]  # input the number of agents
    W = np.zeros((n, n), dtype=float)  # initialize matrix
    deg = adj.sum(axis=1)
    # populate the matrix
    for i in range(n):
        for j in range(n):
            if i == j:  # no self-loops so you keep it as zero
                continue
            if adj[i, j]:
                W[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W


# compute the optimal directions
def compute_U_vectors(Wstar):
    L = len(Wstar)
    S_pairs = []  # store the singular value pairs

    # First pass: compute singular values
    for i in range(L):
        A_i = back_mat_prod(Wstar, i + 1, L - 1)
        B_i = back_mat_prod(Wstar, 0, i - 1)

        # largest singular value via SVD
        _, S_A, _ = anp.linalg.svd(A_i)
        _, S_B, _ = anp.linalg.svd(B_i)

        S_pairs.append((S_A[0], S_B[0]))  # top singular values

    # denominator term
    denom = anp.sqrt(anp.sum(anp.array([sL ** 2 * sR ** 2 for sL, sR in S_pairs])))

    # Second pass: compute actual U_i blocks
    U_list = []
    for i in range(L):
        A_i = back_mat_prod(Wstar, i + 1, L - 1)
        B_i = back_mat_prod(Wstar, 0, i - 1)

        uL, _, vL_t = anp.linalg.svd(A_i)
        uR, _, vR_t = anp.linalg.svd(B_i)

        sL, sR = S_pairs[i]
        scalar = (sL * sR) / denom

        U_i = scalar * anp.outer(uR[:, 0], vL_t[0, :])
        U_list.append(U_i)

    return U_list


# autograd functions
def stack_params(W_list):
    return anp.concatenate([W.reshape(-1) for W in W_list])

def unstack_params(theta, shapes):
    W_list = []
    offset = 0
    for (m, n) in shapes:
        size = m * n
        W = theta[offset:offset + size].reshape((m, n))
        W_list.append(W)
        offset += size
    return W_list


def consensus_step(W_t, V):
    n = len(W_t)
    # shapes are same for all nodes
    shapes = [W.shape for W in W_t[0]]
    P = sum(m * n for (m, n) in shapes)

    theta_cons = []

    for i in range(n):
        accum = anp.zeros(P)
        for j in range(n):
            theta_j = stack_params(W_t[j])  # vec parameters at node j
            accum = accum + V[i, j] * theta_j   # weighted average
        theta_cons.append(accum)

    return theta_cons




def low_rank_init(dims, n_agents, rank):


    L = len(dims) - 1  # number of factors
    W0_list = []

    # generate true factor matrices
    for i in range(n_agents):

        temp = []
        for j in range(L):
            d_in, d_out = dims[j], dims[j + 1]
            W_j = anp.random.randn(d_out, d_in) # generate at random with appropriate dimensions
            Uj, sigma_j, VjT = anp.linalg.svd(W_j, full_matrices=False)

            temp.append(Uj[:, :rank] @ anp.diag(sigma_j[:rank]) @ VjT[:rank, :]) # append the ground truth full matrices

        W0_list.append(temp)

    return W0_list


def consensus_error_total(W_t):
    n = len(W_t)
    L = len(W_t[0])
    err = 0.0

    for j in range(L):  # for each layer j
        # stack parameters of that layer for all nodes
        theta_list = [stack_params([W_t[i][j]]) for i in range(n)]

        # compute layer-wise consensus average
        theta_bar = sum(theta_list) / n

        # accumulate squared 2-norm differences
        for theta_i in theta_list:
            err += anp.linalg.norm(theta_i - theta_bar) ** 2

    return err / n  # consensus error, averaged per node


# effective rank computation
def effective_rank(X):
    # Compute singular values
    S = np.linalg.svd(X, compute_uv=False)
    # Normalize into probability distribution
    p = S / np.sum(S)
    # Shannon entropy
    H = -np.sum(p * np.log(p + 1e-12))  # add 1e-12 to avoid log(0)
    # Effective rank
    return np.exp(H)







