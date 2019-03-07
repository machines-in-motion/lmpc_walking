import numpy as np
# ------------------------------------------------------------------------------
# Description:
# -----------
# this function compute the canonical the quadratic objective term Q
# and the linear objective term p.T inside the cost function
# of the Quadratic Program defined as:
#        minimize
#            (1/2) * u.T * Q * u - p.T * u
#        subject to
#            A.T * u >= b

# Parameters:
# ----------
# alpha : CoM jerk regularizing parameter (scalar)
# gamma : CoP regularizing parameter (scalar)
# N     : number of sampling time intervals
# P_zs  : (Nx3 numpy.array)
# P_zu  : (Nx3 numpy.array)
# x_hat : [x_k, x_dot_k, x_ddot_k].T  (3x1 numpy.array)
# y_hat : [y_k, y_dot_k, y_ddot_k].T  (3x1 numpy.array)
# Z_ref : [z_ref_x, z_ref_y].T        (Nx2 numpy.array)

# Returns:
# -------
# Q     : (2Nx2N numpy.array)
# p_k   : (2Nx1  numpy.array)

def compute_objective_terms(alpha, gamma, N,  P_zs, P_zu, x_hat_k, y_hat_k, Z_ref_k):
    Q_prime         = np.zeros((N,N))
    Q_prime         = alpha*np.identity(N) + (gamma * np.dot(P_zu.T, P_zu))
    Q               = np.zeros((2*N, 2*N))
    Q[0:N, 0:N]     = Q_prime
    Q[N:2*N, N:2*N] = Q_prime

    p_k             = np.zeros((2*N))
    p_k[0:N]        = np.dot((gamma*P_zu.T), (np.dot(P_zs, x_hat_k) - Z_ref_k[:,0]))
    p_k[N:2*N]      = np.dot((gamma*P_zu.T), (np.dot(P_zs, y_hat_k) - Z_ref_k[:,1]))

    return Q, p_k
