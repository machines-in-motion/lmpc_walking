import numpy as np
# Desctiption:
# -----------
# this function assembles the ZMP matrices A and b encapsulating
# the foot ZMP inequality constraints in a Quadratic Program:
#        minimize
#            (1/2) * u.T * Q * u - p.T * u
#        subject to
#            A.T * u >= b

# Parameters:
# ----------
# N           : number of time intervals
# foot_length : length of the foot along the x-axis (scalar)
# foot_width  : length of the foot along the y-axis (scalar)
# P_zs        : (Nx3 numpy.array)
# P_zu        : (NxN numpy.array)
# Z_ref       : [z_ref_x, z_ref_y].T (Nx2 numpy.array)
#                CoP reference trajectory

# Returns:
# -------
# A  : (4Nx2N numpy.array)
#      matrix defining the constraints under which we want to minimize the
#      quadratic function
# b  : (4Nx1  numpy.array)
#      vector defining the ZMP constraints

def add_ZMP_constraints(N, foot_length, foot_width, P_zs, P_zu, Z_ref_k, x_hat_k, y_hat_k):
    A = np.zeros((4*N, 2*N))
    b = np.zeros((4*N))

    A[0:N  , 0:N]   = P_zu
    A[N:2*N, 0:N]   = -P_zu

    A[2*N:3*N, N:2*N] = P_zu
    A[3*N:4*N, N:2*N] = -P_zu

    foot_length_N = np.zeros((N))
    foot_width_N  = np.zeros((N))
    foot_length_N = np.tile(foot_length,(N))
    foot_width_N  = np.tile(foot_width,(N))

    b[0:N]     = -np.dot(P_zs, x_hat_k) + Z_ref_k[0:N,0] - (0.5*foot_length_N)
    b[N:2*N]   = np.dot(P_zs, x_hat_k)  - Z_ref_k[0:N,0] - (0.5*foot_length_N)
    b[2*N:3*N] = -np.dot(P_zs, y_hat_k) + Z_ref_k[0:N,1] - (0.5*foot_width_N)
    b[3*N:4*N] =  np.dot(P_zs, y_hat_k) - Z_ref_k[0:N,1] - (0.5*foot_width_N)

    return A,b
