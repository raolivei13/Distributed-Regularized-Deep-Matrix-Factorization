import numpy as np
from numpy.linalg import svd, norm
from tqdm import trange
from helper_functions_dmf import stack_params, unstack_params, consensus_step, consensus_error_total, effective_rank, back_mat_prod
from autograd import grad
import autograd.numpy as anp
from autograd import hessian
# from main_dmf import eigen_max_of_local_loss


# --------- max eiagn val of hessian computation -------- #
# def mse_l2_hessian_w_max_val(Wi, Yi, Ai, tau):
#     shapes = [W.shape for W in Wi]
#     theta0 = stack_params_list(Wi)
#
#     def wrapped(th):
#         return node_loss(th, shapes, Yi, Ai, tau)
#
#     H = hessian(wrapped)(theta0)
#     return anp.linalg.eigvalsh(H)[-1]

from autograd import grad
import autograd.numpy as anp

def hessian_vector_product(loss, theta, v):
    g = grad(loss)
    return grad(lambda th: anp.dot(g(th), v))(theta)


def mse_l2_hessian_w_max_val(Wi, Yi, Ai, tau, num_iters=50, tol=1e-6):
    shapes = [W.shape for W in Wi]
    theta = stack_params_list(Wi)

    def loss(th):
        return node_loss(th, shapes, Yi, Ai, tau)

    d = theta.size
    v = anp.random.randn(d)
    v /= anp.linalg.norm(v)

    lambda_old = 0.0

    for _ in range(num_iters):
        Hv = hessian_vector_product(loss, theta, v)
        v = Hv / anp.linalg.norm(Hv)
        lambda_new = anp.dot(v, Hv)

        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new

    return lambda_new

# --------- max eigen val of hessian computation -------- #



# autograd functions
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


def gradient_dmf(Wi, Yi, Ai, tau):
    shapes = [W.shape for W in Wi]
    theta0 = stack_params_list(Wi)

    # wrap so autograd only sees θ as variable
    def wrapped(theta):
        return node_loss(theta, shapes, Yi, Ai, tau)

    grad_theta = grad(wrapped)(theta0)
    return anp.array(grad_theta)


def consensus_w_2norm_dmf(W0, Y, A, U,
                n_agents,
                max_iter, T, tau, eta, V, W_star, X_star, verbose=True):

    n = n_agents
    L = len(W0)

    def prod(W_list):
        X = W_list[0]
        for W in W_list[1:]:
            X = W @ X
        return X

    # algorithm outputs
    history = {'distance_from_gt': [], 'consensus_error': []}

    W_t = W0
    shapes = [W.shape for W in W_t[0]]

    distance_from_gt = np.mean([(np.linalg.norm(prod(W_t[i]) - X_star, 'fro')) ** 2 / np.linalg.norm(X_star, 'fro') ** 2 for i in range(n)])
    history['distance_from_gt'].append(distance_from_gt)

    lambdas_per_iter = np.zeros((max_iter, n))
    # -------- BEGINNING OF THE ALGORITHM ------- #
    for k in trange(max_iter, desc="DARN iters"):  # algorithm iterations



        # perform gradient descent at each node
        grad = anp.zeros((n, sum(W.shape[0]*W.shape[1] for W in W_t[0])))
        for i in range(n):
            Wi = W_t[i]  # get the initial guess
            Yi = Y[i]
            Ai = A[i]
            taui = tau[i]


            grad[i, :] = gradient_dmf(Wi, Yi, Ai, taui) # per node stacked gradient

        # print('stop')



        # # compute the sharpness of the loss for all nodes
        # for i in range(n):
        #     lambdas_per_iter[k, i] = mse_l2_hessian_w_max_val(W_t[i], Y[i], A[i], tau[i])


        # consensus weighted averaging
        theta_cons = consensus_step(W_t, V) # in here we iterate over each node and perform consensus

        # consensus-based gradient descent
        W_next = []
        for i in range(n):
            theta_next_i = theta_cons[i] - eta[i] * grad[i, :]
            Wi_next = unstack_params_list(theta_next_i, shapes)
            W_next.append(Wi_next)

        # compute the sharpness of the loss for all nodes
        for i in range(n):
            lambdas_per_iter[k, i] = mse_l2_hessian_w_max_val(W_next[i], Y[i], A[i], tau[i])


        # average normalized loss curve
        distance_from_gt = np.mean([(np.linalg.norm(prod(W_next[i]) - X_star, 'fro')) ** 2 / np.linalg.norm(
            X_star, 'fro') ** 2 for i in range(n)])
        history['distance_from_gt'].append(distance_from_gt)

        # total consensus error
        history['consensus_error'].append(consensus_error_total(W_next))

        # # total maximum eigenvalues
        # history['sharpness_per_node'].append(lambdas_per_iter)



        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"iter {k:4d}: obj {distance_from_gt:.6e}")

        W_t = W_next
        # -------- END OF THE ALGORITHM ------- #

    return W_t, history, lambdas_per_iter

# if __name__ == "__main__":

def l2_norm_reg(W0, Y, A, U,
                n_agents,
                max_iter, T, tau, eta, V, W_star, X_star, verbose=True):

    # inputs: 1. W0: initialization matrices for each node
    #         2.  Y: measurements for each node
    #         3.  A: measurement operators
    #         4.  U: optimal directions
    #         5. max_iter: consensus iterations
    #         6. T: consensus rounds



    # consensus algorithm run associated to that measurement
    W_final, history, lambdas_per_iter = consensus_w_2norm_dmf(W0, Y, A, U,
                n_agents, max_iter, T, tau, eta, V, W_star, X_star, verbose=True)

    # collect consensus error per consensus rounds
    normalized_error = history['distance_from_gt']
    consensus_error_total = history['consensus_error']
    # sharpness_per_iters = history['sharpness_per_node']


    return W_final, normalized_error, consensus_error_total, lambdas_per_iter