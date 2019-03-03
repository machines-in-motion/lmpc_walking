import math
import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
import matplotlib.patches as patches

# headers
import reference_trajectories
import cost_function
import recursive_matrices
import constraints
import motionModel
import plot_utils

# regularization terms in the cost function:
# -----------------------------------------
alpha = 0*10**(-6)
gamma = 1#10**(-3)

# regularization term for Q
Q_reg = 10**(-10)

# CoM initial state: [x, xdot, x_ddot].T
#                    [y, ydot, y_ddot].T
# --------------------------------------
x_hat_0 = np.array([0.0, 0.0, 0.0])
y_hat_0 = np.array([-0.09, 0.0, 0.0])

# Inverted pendulum parameters:
# ----------------------------
h           = 0.75
g           = 9.81
foot_length = 0.2
foot_width  = 0.1

# MPC parameters:
# --------------
T                = 0.1                        # sampling time interval
step_time        = 0.8                        # time needed for every step
no_steps_per_T   = int(round(step_time/T))
no_steps         = 8                          # number of desired walking steps
walking_time     = no_steps * no_steps_per_T  # number of sampling time intervals
N                = walking_time


# compute CoP reference trajectory:
# --------------------------------
foot_step_0 = np.array([0.0, -0.09])     # initial foot step position in x-y
Foot_steps  = reference_trajectories.manual_foot_placement(foot_step_0, no_steps)
Z_ref  = reference_trajectories.create_CoP_trajectory(no_steps, Foot_steps, \
                                                      walking_time, no_steps_per_T)
#plt.plot(Z_ref[:,0], Z_ref[:,1])
#plt.show()

# construct your preview system: 'Go pokemon !'
# --------------------------------------------
[P_ps, P_vs, P_zs, P_pu, P_vu, P_zu] = recursive_matrices.compute_recursive_matrices(N, T, h, g)
[Q, p_k] = cost_function.compute_objective_terms(alpha, gamma, walking_time, \
                                                 P_zs, P_zu, x_hat_0, y_hat_0,\
                                                 Z_ref)
[A, b]   = constraints.add_ZMP_constraint(walking_time, foot_length,\
                                          foot_width, P_zs, P_zu, Z_ref)
Q        = Q_reg*np.identity(2*N) + Q   #making sure that Q is +ve definite

# call the solver del sto cazzo:
# -----------------------------
U = solve_qp(Q, -p_k, -A.T, -b)[0]

# open-loop planning: (based on the initial state x_hat_0, y_hat_0)
# ------------------------------------------------------------------------------
[X, Y, Z_x, Z_y] = motionModel.open_loop(P_ps, P_vs, P_zs, P_pu, P_vu, P_zu, \
                                         walking_time, x_hat_0, y_hat_0, U)
# ------------------------------------------------------------------------------
# visualize:
# ------------------------------------------------------------------------------
time               = np.arange(0, round(walking_time*T, 2), T)
min_admissible_CoP = Z_ref - np.tile([foot_length, foot_width], (walking_time,1))
max_admissible_cop = Z_ref + np.tile([foot_length, foot_width], (walking_time,1))

# time vs CoP and CoM in x: 'A.K.A run rabbit run !'
# -------------------------------------------------
plot_utils.plot_x(time, walking_time, min_admissible_CoP, max_admissible_cop, \
                  Z_x, X, Z_ref)

# time VS CoP and CoM in y: 'A.K.A what goes up must go down'
# ----------------------------------------------------------
plot_utils.plot_y(time, walking_time, min_admissible_CoP, max_admissible_cop, \
                  Z_y, Y, Z_ref)

# plot CoP, CoM in x Vs Cop, CoM in y:
# -----------------------------------
plot_utils.plot_xy(time, walking_time, foot_length, foot_width, Z_ref, \
                   Z_x, Z_y, X, Y)
