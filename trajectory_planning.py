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
alpha = 10**(-6)
gamma = 0*10**(-6)

# regularization term for Q
Q_reg = 0*10**(-10)

# CoM initial state: [x, xdot, x_ddot].T
#                    [y, ydot, y_ddot].T
# --------------------------------------
x_hat_0 = np.array([0.0, 0.0, 0.0])
y_hat_0 = np.array([-0.09, 0.0, 0.0])

# Inverted pendulum parameters:
# ----------------------------
h           = 0.8
g           = 9.81
foot_length = 0.2
foot_width  = 0.1

# MPC parameters:
# --------------
T                = 0.1                        # sampling time interval
step_time        = 0.8                        # time needed for every step
no_steps_per_T   = int(round(step_time/T))

# walking parameters:
# ------------------
Step_length      = 0.21                       # fixed step length in the xz-plane
no_steps         = 3                          # number of desired walking steps
walking_time     = no_steps * no_steps_per_T  # number of desired walking intervals

# compute CoP reference trajectory:
# --------------------------------
foot_step_0 = np.array([0.0, -0.09])     # initial foot step position in x-y
Foot_steps  = reference_trajectories.manual_foot_placement(foot_step_0, Step_length no_steps)
Z_ref  = reference_trajectories.create_CoP_trajectory(no_steps, Foot_steps, \
                                                      walking_time, no_steps_per_T)
#print Z_ref
#plt.plot(Z_ref[:,0], Z_ref[:,1])
#plt.show()

# construct your preview system: 'Go pokemon !'
# --------------------------------------------
[P_ps, P_vs, P_as, P_zs, P_pu, P_vu, P_au, P_zu] = recursive_matrices.compute_recursive_matrices(walking_time, T, h, g)
[Q, p_k] = cost_function.compute_objective_terms(alpha, gamma, walking_time, \
                                                 P_zs, P_zu, x_hat_0, y_hat_0,\
                                                 Z_ref)
[A, b]   = constraints.add_ZMP_constraints(walking_time, foot_length, foot_width,\
                                          P_zs, P_zu, Z_ref, x_hat_0, y_hat_0)
Q        = Q_reg*np.identity(2*walking_time) + Q   #making sure that Q is +ve definite

# call the solver del sto cazzo:
# -----------------------------
U = solve_qp(Q, -p_k, A.T, b)[0]

# open-loop planning: (based on the initial state x_hat_0, y_hat_0)
# -------------------------------------------------------------------------
[X, Y, Z_x, Z_y] = motionModel.compute_recursive_dynamics(P_ps, P_vs, P_as, P_zs, P_pu, \
                                                        P_vu, P_au, P_zu, walking_time, \
                                                        x_hat_0, y_hat_0, U)

# ------------------------------------------------------------------------------
# debugging:
# ------------------------------------------------------------------------------
#for i in range (2):
#    print 'x_hat_k = ', X[i,0], '\n'
#    print 'x_hatdot_k = ', X[i,1], '\n'
#    print 'x_hatddot_k = ', X[i,2], '\n'

#    print 'y_hat_k = ', Y[i,0], '\n'
#    print 'y_hatdot_k = ', Y[i,1], '\n'
#    print 'y_hatddot_k = ', Y[i,2], '\n'

#    print 'Z_x_k = ', Z_x[i], '\n'
#    print 'Z_y_k = ', Z_y[i], '\n'

# ------------------------------------------------------------------------------
# visualize:
# ------------------------------------------------------------------------------
time               = np.arange(0, round(walking_time*T, 2), T)
min_admissible_CoP = Z_ref - np.tile([foot_length/2, foot_width/2], (walking_time,1))
max_admissible_cop = Z_ref + np.tile([foot_length/2, foot_width/2], (walking_time,1))

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
