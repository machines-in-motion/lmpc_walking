import numpy as np
# Desctiption:
# -----------
# this function assembles the ZMP matrices A and b encapsulating
# the foot ZMP inequality constraints in Quadratic Program:
#        minimize
#            (1/2) * u.T * Q * u + p.T * u
#        subject to
#            A * u <= b

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

def add_ZMP_constraint(N, foot_length, foot_width, P_zs, P_zu, Z_ref):
    A = np.zeros((4*N, 2*N))
    b = np.zeros((4*N))

    A[0:N  , 0:N]     = P_zu
    A[N:2*N, N:2*N]   = P_zu

    A[2*N:3*N, 0:N]   = -P_zu
    A[3*N:4*N, N:2*N] = -P_zu

    x_hat_k           = np.zeros((N,3))
    x_hat_k[0:N,0]    = Z_ref[0:N,0]

    y_hat_k           = np.zeros((N,3))
    y_hat_k[0:N,0]    = Z_ref[0:N,1]

    for i in range(N): # compute ZMP constraint based on the current ZMP location
        current_ZMP_x_constraint  = x_hat_k[i,:].T
        current_ZMP_y_constraint  = y_hat_k[i,:].T

        b[i]     = foot_length + np.dot(P_zs[i,:], current_ZMP_x_constraint)
        b[i+N]   = foot_width  + np.dot(P_zs[i,:], current_ZMP_y_constraint)
        b[i+2*N] = foot_length - np.dot(P_zs[i,:], current_ZMP_x_constraint)
        b[i+3*N] = foot_width  - np.dot(P_zs[i,:], current_ZMP_y_constraint)

    return A,b
