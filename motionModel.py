import numpy as np

# headers
import cost_function
import constraints

# computes dynamics recursively for N time intervals based on current states
# x_k, y_k and control inputs U
# ------------------------------------------------------------------------------

def compute_recursive_dynamics(P_ps, P_vs, P_as, P_zs, P_pu, P_vu, P_au, P_zu, N, x_hat_k, y_hat_k, U):
    X         = np.zeros((N,3))
    Y         = np.zeros((N,3))

    # evaluate your states over the horizon
    X[0:N,0]  = np.dot(P_ps, x_hat_k) + np.dot(P_pu, U[0:N])   #x
    X[0:N,1]  = np.dot(P_vs, x_hat_k) + np.dot(P_vu, U[0:N])   #x_dot
    X[0:N,2]  = np.dot(P_as, x_hat_k) + np.dot(P_au, U[0:N])   #x_ddot

    Y[0:N,0]  = np.dot(P_ps, y_hat_k) + np.dot(P_pu, U[N:2*N]) #y
    Y[0:N,1]  = np.dot(P_vs, y_hat_k) + np.dot(P_vu, U[N:2*N]) #y_Dot
    Y[0:N,2]  = np.dot(P_as, y_hat_k) + np.dot(P_au, U[N:2*N]) #y_ddot

    # evaluate actual CoP
    Z_x = np.zeros((N))
    Z_y = np.zeros((N))
    Z_x = np.dot(P_zs, x_hat_k) + np.dot(P_zu, U[0:N])
    Z_y = np.dot(P_zs, y_hat_k) + np.dot(P_zu, U[N:2*N])
    return X, Y, Z_x, Z_y
