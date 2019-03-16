import numpy as np
# Description:
# -----------
# this function implements a hard-coded desired foot steps placement designed
# to be in the middle of the current stance foot, starting with the right foot

# Parameters:
# ----------
#  foot_step_0 : initial foot step location
#            [foot_step_x0, foot_step_y0].T (2x1 numpy.array)
#  no_steps    : number of walking foot steps  (scalar)

# Returns:
# -------
#  Foot_steps = [Foot_steps_x, Foot_steps_y].T (no_stepsx2 numpy.array)

def manual_foot_placement(foot_step_0, fixed_step_x, no_steps):
    Foot_steps   = np.zeros((no_steps, 2))
    if no_steps == 1:
        Foot_steps[0,0] = foot_step_0[0]
        Foot_steps[0,1] = foot_step_0[1]

    elif no_steps == 2:
        Foot_steps[0,:] = foot_step_0
        Foot_steps[1,:] = -foot_step_0
    else:
        for i in range(Foot_steps.shape[0]):
            if i == 0:
                Foot_steps[i,:] = foot_step_0
            elif i == 1:
                Foot_steps[i,:] = -foot_step_0
            else:
                Foot_steps[i,0] = Foot_steps[i-2,0] + fixed_step_x
                Foot_steps[i,1] = Foot_steps[i-2,1] 
    return Foot_steps


def create_CoP_trajectory(no_steps, Foot_steps, walking_time, no_steps_per_T):
    Z_ref  = np.zeros((walking_time,2))
    j = 0
    if Foot_steps.shape[0] == 1:
        Z_ref[j:j+no_steps_per_T, :] = Foot_steps[0,:]
    else:
        for i in range (Foot_steps.shape[0]):
             Z_ref[j:j+no_steps_per_T, :] = Foot_steps[i,:]
             j = j + no_steps_per_T
    return Z_ref
