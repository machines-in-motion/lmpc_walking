#    LMPC_walking is a python software implementation of one of the algorithms
#    refered in this paper https://hal.inria.fr/inria-00391408v2/document
#    Copyright (C) 2019 @ahmad gazar

#    LMPC_walking is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    LMPC_walking is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.linalg import toeplitz

# Description:
# -----------
# this functon computes the recursive integration matrices of the preview system
# this function is computed once offline since all parameters are all fixed
# before the beginning of the optimization

# Parameters:
# ----------
# N: preceding horizon
# T: sampling time
# h: fixed height of the CoM assuming walking on a flat terrain
# g: norm of the gravity acceleration vector

# Returns:
# -------
# P_ps, P_vs, P_as, P_zs (Nx3 numpy.array)
# P_pu, P_vu, P_au, P_zu (N-1 x N-1 numpy.array)

def compute_recursive_matrices(N, T, h, g):
    P_ps = np.zeros((N,3))
    P_vs = np.zeros((N,3))
    P_as = np.zeros((N,3))
    P_zs = np.zeros((N,3))

    P_pu = np.zeros((N,N))
    P_vu = np.zeros((N,N))
    P_au = np.zeros((N,N))
    P_zu = np.zeros((N,N))

    temp_pu = np.zeros((N))
    temp_vu = np.zeros((N))
    temp_au = np.zeros((N))
    temp_zu = np.zeros((N))

    for i in range(N):
         P_ps[i, 0:3] = np.array([1.0, (i+1.0)*T, 0.5*((i+1.0)*T)**2.0])
         P_vs[i, 0:3] = np.array([0.0,      1.0 , (i+1.0)*T])
         P_as[i, 0:3] = np.array([0.0,      0.0 , 1.0])
         P_zs[i, 0:3] = np.array([1.0, (i+1.0)*T, 0.5*(((i+1.0)*T)**2.0) - h/g])


         temp_pu[i] = np.array([(1.0 + (3.0*i) + 3.0*((i)**2.0))*(T**3.0)/6.0])
         temp_vu[i] = np.array([(1.0 + (2.0*i)) *(T**2.0)*0.5])
         temp_au[i] = np.array([T])
         temp_zu[i] = np.array([(1.0 + (3.0*i) + 3.0*((i)**2.0))*((T**3.0)/6.0) -T*h/g])

    P_pu = toeplitz(temp_pu) * np.tri(N,N)
    P_vu = toeplitz(temp_vu) * np.tri(N,N)
    P_au = toeplitz(temp_au) * np.tri(N,N)
    P_zu = toeplitz(temp_zu) * np.tri(N,N)

    return P_ps, P_vs, P_as, P_zs, P_pu, P_vu, P_au, P_zu

# Description:
# -----------
# this functon computes the future states and computed CoP along the preceding
# horizon based on the current state and input

# X_k+1   = P_ps * x_hat_k + P_pu * U_x
# Y_k+1   = P_ps * y_hat_k + P_pu * U_y
# Z_x_k+1 = P_zs * x_hat_k + P_zu * U_x
# Z_y_k+1 = P_zs * y_hat_k + P_zu * U_y


# Parameters:
# ----------
# P_ps, P_vs, P_as, P_zs                (Nx3 numpy.array)
# P_pu, P_vu, P_au, P_zu                (N-1 x N-1 numpy.array)
# N: preceding horizon                  (scalar)

# x_hat_k:= [x_k, x_dot_k, x_ddot_k].T   current x state (3, numpy.array)
# y_hat_k:= [y_k, y_dot_k, y_ddot_k].T   current y state (3, numpy.array)

# U      := [x_jerk_k, .. , x_jerk_k+N-1, .. , y_jerk_k+N-1].T
#           current control inputs (x and y jerks)
#           along the horizon (2N-2, numpy.array)

# Returns:
# -------
# X:= [x_k+1, x_dot_k+1  , x_ddot_k+1]  referred above as X_k+1 (Nx3 numpy.array)
#       .   ,   .        , .
#       .   ,   .        , .
#     [x_k+N , x_dot_k+N , x_ddot_k+N]  referred above as Y_k+1 (Nx3 numpy.array)

# Y:= [y_k+1, y_dot_k+1  , y_ddot_k+1]
#       .   ,   .        , .
#       .   ,   .        , .
#     [y_k+N , x_dot_k+N , y_ddot_k+N]

# Z_x:= [z_x_k+1, .. , z_x_k+N].T    referred as Z_x_k+1 above (Nx1 numpy.array)
# Z_y:= [z_y_k+1, .. , z_y_k+N].T    referred as Z_y_k+1 above (Nx1 numpy.array)

def compute_recursive_dynamics(P_ps, P_vs, P_as, P_zs, P_pu, P_vu, P_au, P_zu,\
                               N, x_hat_k, y_hat_k, U):
    X         = np.zeros((N,3))
    Y         = np.zeros((N,3))

    # evaluate your CoM states along the horizon
    X[0:N,0]  = np.dot(P_ps, x_hat_k) + np.dot(P_pu, U[0:N])   #x
    X[0:N,1]  = np.dot(P_vs, x_hat_k) + np.dot(P_vu, U[0:N])   #x_dot
    X[0:N,2]  = np.dot(P_as, x_hat_k) + np.dot(P_au, U[0:N])   #x_ddot

    Y[0:N,0]  = np.dot(P_ps, y_hat_k) + np.dot(P_pu, U[N:2*N]) #y
    Y[0:N,1]  = np.dot(P_vs, y_hat_k) + np.dot(P_vu, U[N:2*N]) #y_dot
    Y[0:N,2]  = np.dot(P_as, y_hat_k) + np.dot(P_au, U[N:2*N]) #y_ddot

    # evaluate computed CoP
    Z_x = np.zeros((N))
    Z_y = np.zeros((N))
    Z_x = np.dot(P_zs, x_hat_k) + np.dot(P_zu, U[0:N])
    Z_y = np.dot(P_zs, y_hat_k) + np.dot(P_zu, U[N:2*N])

    return X, Y, Z_x, Z_y
