import math
import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
import matplotlib.patches as patches
import csv

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
gamma       = 0*10**(-3)

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
Step_length           = 0.21                             # fixed step length in the xz-plane
no_desired_steps      = 6                                 # number of desired walking steps
no_planned_steps      = 2+no_desired_steps                 # planning 2 steps ahead
desired_walking_time  = no_desired_steps * no_steps_per_T  # number of desired walking intervals
planned_walking_time  = no_planned_steps * no_steps_per_T  # number of planned walking intervals

# Adding gaussian white noise to the control input: use to emulate closed-loop behavior in simulation
U_noise = 0*np.random.normal(0, 1, 2*N)  # multiply by zero to get open-loop MPC
#print U_noise

# compute CoP reference trajectory:
# --------------------------------
foot_step_0   = np.array([0.0, -0.09])    # initial foot step position in x-y

desiredFoot_steps    = reference_trajectories.manual_foot_placement(foot_step_0, Step_length, no_desired_steps)
#print 'desiredFoot_steps', desiredFoot_steps, '\n'

#plannedFoot_steps    = reference_trajectories.manual_foot_placement(foot_step_0, Step_length, no_planned_steps)

desired_Z_ref = reference_trajectories.create_CoP_trajectory(no_desired_steps, desiredFoot_steps, \
                                                    desired_walking_time, no_steps_per_T)
#planned_Z_ref = reference_trajectories.create_CoP_trajectory(no_planned_steps, plannedFoot_steps, \
#                                                planned_walking_time, no_steps_per_T)
planned_Z_ref = np.zeros((planned_walking_time, 2))
planned_Z_ref[0:desired_walking_time,:] =  desired_Z_ref

# plan the last 2 steps in your CoP plan to step in place
planned_Z_ref[desired_walking_time:desired_walking_time+N/2, 0] = desired_Z_ref[desired_walking_time-1,0]
planned_Z_ref[desired_walking_time:desired_walking_time+N/2, 1] = -desired_Z_ref[desired_walking_time-1,1]

planned_Z_ref[desired_walking_time+N/2:planned_walking_time, 0] = desired_Z_ref[desired_walking_time-1,0]
planned_Z_ref[desired_walking_time+N/2:planned_walking_time, 1] = desired_Z_ref[desired_walking_time-1,1]
#print 'desired_Z_ref', desired_Z_ref, '\n'
#print 'planned_Z_ref', planned_Z_ref, '\n'

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

T_k = 0.0
time_k = np.zeros((N))
[P_ps, P_vs, P_as, P_zs, P_pu, P_vu, P_au, P_zu] = recursive_matrices.compute_recursive_matrices(N, T, h, g)

# create a dictionary for data logging of horizons at every sampling time T
horizon_data = {}

for i in range(desired_walking_time):
    #print 'Z_ref_k = ', Z_ref_k
    horizon_data[i] = {}
    horizon_data[i]['zmp_reference'] = Z_ref_k
    [Q, p_k] = cost_function.compute_objective_terms(alpha, gamma, N, P_zs,\
                                                      P_zu, x_hat_k, y_hat_k,\
                                                      Z_ref_k)
    [A, b]   = constraints.add_ZMP_constraints(N, foot_length, foot_width,\
                                                   P_zs, P_zu, Z_ref_k, \
                                                   x_hat_k, y_hat_k)
    #Q        += Q_reg*np.identity(2*N)   #making sure that Q is +ve definite
    current_U = solve_qp(Q, -p_k, A.T, b)[0] + U_noise

    # evaluate your recursive dynamics over the current horizon:
    [X_k, Y_k, Z_x_k, Z_y_k] = motionModel.compute_recursive_dynamics(P_ps, P_vs, P_as, P_zs, P_pu, \
                                                            P_vu, P_au, P_zu, N, \
                                                            x_hat_k, y_hat_k, current_U)

    horizon_data[i]['X_k']   = X_k
    horizon_data[i]['Y_k']   = Y_k
    horizon_data[i]['Z_x_k'] = Z_x_k
    horizon_data[i]['Z_y_k'] = Z_y_k

    time_k               = np.arange(T_k, T_k + (N*T), T)
    if time_k.size != N:
        time_k =  time_k[0:N]
    horizon_data[i]['time_k'] = time_k

    # update the next state
    X_total[i,:]  = X_k[0,:]
    Y_total[i,:]  = Y_k[0,:]

    Z_x_total[i]  = Z_x_k[0]
    Z_y_total[i]  = Z_y_k[0]

    x_hat_k   = X_k[0,:]                    # update the next x state for next iteration
    y_hat_k   = Y_k[0,:]                    # update the next y state for next iteration
    Z_ref_k   = planned_Z_ref[i+1:i+N+1,:]  # update cop references for next iteration

    # update time step
    T_k = T_k + T
    print 'T_k = ', T_k
    print 'loop number = ', i
    #print horizon_data[i]

# ------------------------------------------------------------------------------
# debugging:
# ------------------------------------------------------------------------------
    #print horizon_data[i]
    #print 'x_hat_k = ', x_hat_k[0], '\n'
    #print 'x_hatdot_k = ', X_k[0,1], '\n'
    #print 'x_hatddot_k = ', X_k[0,2], '\n'

    #print 'y_hat_k = ', y_hat_k[0], '\n'
    #print 'y_hatdot_k = ', Y_k[0,1], '\n'
    #print 'y_hatddot_k = ', Y_k[0,2], '\n'

    #print 'Z_x_k = ', Z_x_k[0], '\n'
    #print 'Z_y_k = ', Z_y_k[0], '\n'

    #print 'Z_ref_k = ',  Z_ref_k

# visualize all your freaking horizons: "GOTCHA HORIZON 51, FIGURE 104"
#plot_utils.plot_horizons(desired_walking_time, N, desired_Z_ref, horizon_data, foot_length, foot_width)

# ------------------------------------------------------------------------------
# visualize your final trajectory:
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
