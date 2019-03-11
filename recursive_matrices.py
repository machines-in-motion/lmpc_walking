import numpy as np
from scipy.linalg import toeplitz
# Description:
# -----------
# this functon computes the recursive matrices of the preview system offline

# Parameters:
# ----------
# N: number of sampling time intervals
# T: sampling time
# h: fixed height of the CoM assuming walking on a flat terrain
# g: norm of the gravity acceleration vector

# Returns:
# -------
# P_ps, P_vs, P_zs (Nx3 matrices)
# P_pu, P_vu, P_zu (NxN matrices)

def compute_recursive_matrices(N, T, h, g):
    P_ps = np.zeros((N,3))
    P_vs = np.zeros((N,3))
    P_zs = np.zeros((N,3))

    P_pu = np.zeros((N,N))
    P_vu = np.zeros((N,N))
    P_zu = np.zeros((N,N))

    temp_pu = np.zeros((N))
    temp_vu = np.zeros((N))
    temp_zu = np.zeros((N))

    for i in range(N):
        P_ps[i, 0:3] = np.array([1.0, (i+1.0)*T, 0.5*((i+1.0)*T)**2.0])
        P_vs[i, 0:3] = np.array([0.0,      1.0 , (i+1.0)*T])
        P_zs[i, 0:3] = np.array([1.0, (i+1.0)*T, 0.5*(((i+1.0)*T)**2.0) - h/g])

        if i == 0:
            temp_pu[i] = np.array([(T**3.0)/6.0])
            temp_vu[i] = np.array([(T**2.0)*0.5])
            temp_zu[i] = np.array([((T**3.0)/6.0) -T*h/g])
        else:
            temp_pu[i] = np.array([(1.0 + (3.0*(i+1.0)) + 3.0*((i+1.0)**2.0))*(T**3.0)/6.0])
            temp_vu[i] = np.array([(1.0 + (2.0*(i+1.0))) *(T**2.0)*0.5])
            temp_zu[i] = np.array([(1.0 + (3.0*(i+1.0)) + 3.0*((i+1.0)**2.0))*((T**3.0)/6.0) -T*h/g])

        #print temp_pu
        #print temp_vu
        #print temp_zu
        P_pu = toeplitz(temp_pu) * np.tri(N,N)
        P_vu = toeplitz(temp_vu) * np.tri(N,N)
        P_zu = toeplitz(temp_zu) * np.tri(N,N)
    ## Always check your matrices because karma is a fucking bitch !
    #print P_ps
    #print P_zs.shape
    #print P_vs

    #print P_pu
    #print P_vu
    #print P_zu

    return P_ps, P_vs, P_zs, P_pu, P_vu, P_zu
