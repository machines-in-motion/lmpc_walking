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

def manual_foot_placement(foot_step_0, no_steps):
    first_stupid_step_left  = 0.2
    first_stupid_step_right = 0.45
    fixed_step_x = 0.525
    Foot_steps        = np.zeros((no_steps, 2))
    Foot_steps[0,:]   = foot_step_0
    Foot_steps[1,:]   = -foot_step_0
    for i in range(Foot_steps.shape[0]):
        if i == Foot_steps.shape[0] - 2:
            break
        if i == 0:
            Foot_steps[i+2,0] =  Foot_steps[i,0] + first_stupid_step_left  #x-component
            Foot_steps[i+2,1] =  Foot_steps[i,1]                           #y-component
        elif i == 1:
            Foot_steps[i+2,0] =  Foot_steps[i,0] + first_stupid_step_right #x-component
            Foot_steps[i+2,1] =  Foot_steps[i,1]                           #y-component
        else:
            Foot_steps[i+2,0] =  Foot_steps[i,0] + fixed_step_x            #x-component
            Foot_steps[i+2,1] =  Foot_steps[i,1]                           #y-component
    Foot_steps[no_steps-1,0]  = Foot_steps[no_steps-2,0]
    Foot_steps[no_steps-1,1]  = Foot_steps[no_steps-2,1]
    return Foot_steps

def create_CoP_trajectory(Foot_steps, walking_time, no_steps_per_T):
    Z_ref  = np.zeros((walking_time,2))
    #print Z_ref.shape
    j = 0
    for i in range (Foot_steps.shape[0]):
         Z_ref[j:j+no_steps_per_T, :] = Foot_steps[i,:]
         j = j + no_steps_per_T
    return Z_ref
