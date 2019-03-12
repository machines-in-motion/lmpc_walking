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
alpha       = 10**(-6)
gamma       = 0*10**(-6)

# regularization term for Q
Q_reg       = 0*10**(-10)

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

# MPC Parameters:
# --------------
T                     = 0.1                                # sampling time interval
step_time             = 0.8                                # time needed for every step
no_steps_per_T        = int(round(step_time/T))
N                     = 16                                 # preceding horizon

# walking parameters:
# ------------------
Step_length           = 0.21                               # fixed step length in the xz-plane
no_desired_steps      = 4                                  # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                 # planning 2 steps ahead
desired_walking_time  = no_desired_steps * no_steps_per_T  # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T  # number of planned walking intervals

# compute CoP reference trajectory:
# --------------------------------
foot_step_0   = np.array([0.0, -0.09])    # initial foot step position in x-y

desiredFoot_steps    = reference_trajectories.manual_foot_placement(foot_step_0, Step_length, no_desired_steps)
plannedFoot_steps    = reference_trajectories.manual_foot_placement(foot_step_0, Step_length, no_planned_steps)

desired_Z_ref = reference_trajectories.create_CoP_trajectory(no_desired_steps, desiredFoot_steps, \
                                                    desired_walking_time, no_steps_per_T)
planned_Z_ref = reference_trajectories.create_CoP_trajectory(no_planned_steps, plannedFoot_steps, \
                                                planned_walking_time, no_steps_per_T)

#print 'desired_Z_ref', desired_Z_ref, '\n'
#print 'planned_Z_ref', planned_Z_ref, '\n'
#planned_Z_ref[desired_walking_time:planned_walking_time,:] = desired_Z_ref[desired_walking_time-1,:]
#print planned_Z_ref

x_hat_k   = x_hat_0
y_hat_k   = y_hat_0
Z_ref_k   = planned_Z_ref[0:N,:]

X_k       = np.zeros((N,3))
Y_k       = np.zeros((N,3))
Z_x_k     = np.zeros((N))
Z_y_k     = np.zeros((N))

X_total   = np.zeros((desired_walking_time,3))
Y_total   = np.zeros((desired_walking_time,3))
Z_x_total = np.zeros((desired_walking_time))
Z_y_total = np.zeros((desired_walking_time))

[P_ps, P_vs, P_as, P_zs, P_pu, P_vu, P_au, P_zu] = recursive_matrices.compute_recursive_matrices(N, T, h, g)

for i in range(desired_walking_time):

    [Q, p_k] = cost_function.compute_objective_terms(alpha, gamma, N, P_zs,\
                                                      P_zu, x_hat_k, y_hat_k,\
                                                      Z_ref_k)
    [A, b]   = constraints.add_ZMP_constraints(N, foot_length, foot_width,\
                                                   P_zs, P_zu, Z_ref_k, \
                                                   x_hat_k, y_hat_k)
    Q        += Q_reg*np.identity(2*N)   #making sure that Q is +ve definite
    current_U = solve_qp(Q, -p_k, A.T, b)[0]

    # evaluate your recursive dynamics over the current horizon:
    [X_k, Y_k, Z_x_k, Z_y_k] = motionModel.compute_recursive_dynamics(P_ps, P_vs, P_as, P_zs, P_pu, \
                                                            P_vu, P_au, P_zu, N, \
                                                            x_hat_k, y_hat_k, current_U)
    # update the next state
    X_total[i,:]  = X_k[0,:]
    Y_total[i,:]  = Y_k[0,:]

    Z_x_total[i]  = Z_x_k[0]
    Z_y_total[i]  = Z_y_k[0]

    x_hat_k   = X_k[0,:]                    # update the next x state for next iteration
    y_hat_k   = Y_k[0,:]                    # update the next y state for next iteration
    Z_ref_k   = planned_Z_ref[i+1:i+N+1,:]  # update cop references for next iteration

# ------------------------------------------------------------------------------
# debugging:
# ------------------------------------------------------------------------------
    #print 'x_hat_k = ', x_hat_k[0], '\n'
    #print 'x_hatdot_k = ', X_k[0,1], '\n'
    #print 'x_hatddot_k = ', X_k[0,2], '\n'

    #print 'y_hat_k = ', y_hat_k[0], '\n'
    #print 'y_hatdot_k = ', Y_k[0,1], '\n'
    #print 'y_hatddot_k = ', Y_k[0,2], '\n'

    #print 'Z_x_k = ', Z_x_k[0], '\n'
    #print 'Z_y_k = ', Z_y_k[0], '\n'

# ------------------------------------------------------------------------------
# visualize:
# ------------------------------------------------------------------------------
time               = np.arange(0, round(desired_walking_time*T, 2), T)
min_admissible_CoP = desired_Z_ref - np.tile([foot_length/2, foot_width/2], (desired_walking_time,1))
max_admissible_cop = desired_Z_ref + np.tile([foot_length/2, foot_width/2], (desired_walking_time,1))

# time vs CoP and CoM in x: 'A.K.A run rabbit run !'
# -------------------------------------------------
plot_utils.plot_x(time, desired_walking_time, min_admissible_CoP, max_admissible_cop, \
                  Z_x_total, X_total, desired_Z_ref)

# time VS CoP and CoM in y: 'A.K.A what goes up must go down'
# ----------------------------------------------------------
plot_utils.plot_y(time, desired_walking_time, min_admissible_CoP, max_admissible_cop, \
                  Z_y_total, Y_total, desired_Z_ref)

# plot CoP, CoM in x Vs Cop, CoM in y:
# -----------------------------------
plot_utils.plot_xy(time, desired_walking_time, foot_length, foot_width, desired_Z_ref, \
                   Z_x_total, Z_y_total, X_total, Y_total)
