import numpy as np
from gym_ballhoop.envs import params, transition

# state is a 9-d vector
# it goes th Dth psi Dpsi r Dr phi Dphi mode

def update_outer(state, t_i):
    """
    Updates the environment's state when it is rolling
    on the outer hoop

    Parameters
    ----------
    t_i : float
        Un-normalized torque applied to the hoop

    Returns
    -------
    np.ndarray
        The state of the environment after applying the torque
    """
    assert state[8] == 1

    th = state[0]
    Dth = state[1]
    psi = state[2]
    Dpsi = state[3]
    r = state[4]
    Dr = state[5]
    phi = state[6]
    Dphi = state[7]
    mode = state[8]
    #time = state[9]

    th_dot = Dth
    Dth_dot = params.ath1 * Dth + params.ath2 * np.sin(psi) + params.ath3 * Dpsi + params.bth * t_i
    psi_dot = Dpsi
    Dpsi_dot = params.apsi1 * Dth + params.apsi2 * np.sin(psi) + params.apsi3 * Dpsi + params.bpsi * t_i

    th_out = th + params.h * th_dot
    Dth_out = Dth + params.h * Dth_dot
    psi_out = psi + params.h * psi_dot
    Dpsi_out = Dpsi + params.h * Dpsi_dot
    r_out = r
    Dr_out = 0
    phi_out = (th_out - psi_out) * (params.Ro / params.Rb)
    Dphi_out = (Dth_out - Dpsi_out) * (params.Ro / params.Rb)
    mode_out = 1

    return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out], dtype=np.float64)

def update_free(state, t_i):
    """
    Updates the environment's state when it is in
    free fall

    Parameters
    ----------
    t_i : float
        Un-normalized torque applied to the hoop

    Returns
    -------
    np.ndarray
        The state of the environment after applying the torque
    """
    assert state[8] == 2

    th = state[0]
    Dth = state[1]
    psi = state[2]
    Dpsi = state[3]
    r = state[4]
    Dr = state[5]
    phi = state[6]
    Dphi = state[7]
    mode = state[8]
    #time = state[9]

    th_dot = Dth
    Dth_dot = params.ath1 * Dth + params.bth * t_i
    phi_dot = Dphi
    Dphi_dot = 0
    r_dot = Dr
    Dr_dot = r * Dpsi**2 + params.g * np.cos(psi)
    psi_dot = Dpsi
    Dpsi_dot = -(params.g * np.sin(psi) + 2 * Dpsi * Dr) / r
    
    th_out = th + params.h * th_dot
    Dth_out = Dth + params.h * Dth_dot
    phi_out = phi + params.h * phi_dot
    Dphi_out = Dphi + params.h * Dphi_dot
    r_out = r + params.h * r_dot
    Dr_out = Dr + params.h * Dr_dot
    psi_out = psi + params.h * psi_dot
    Dpsi_out = Dpsi + params.h * Dpsi_dot
    mode_out = 2

    return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out], dtype=np.float64)

def update_inner(state, t_i):
    """
    Updates the environment's state when it is rolling
    on the inner hoop

    Parameters
    ----------
    t_i : float
        Un-normalized torque applied to the hoop

    Returns
    -------
    np.ndarray
        The state of the environment after applying the torque
    """
    assert state[8] == 3

    th = state[0]
    Dth = state[1]
    psi = state[2]
    Dpsi = state[3]
    r = state[4]
    Dr = state[5]
    phi = state[6]
    Dphi = state[7]
    mode = state[8]
    #time = state[9]

    th_dot = Dth
    Dth_dot = params.ath1 * Dth + params.bth * t_i
    psi_dot = Dpsi
    Dpsi_dot = (1 / params.ath1) * (-params.b_bar * Dpsi - params.c_bar * np.sin(psi) - params.d_bar * Dth + params.e_bar * t_i)
    
    th_out = th + params.h * th_dot
    Dth_out = Dth + params.h * Dth_dot
    psi_out = psi + params.h * psi_dot
    Dpsi_out = Dpsi + params.h * Dpsi_dot
    r_out = r
    Dr_out = 0
    phi_out = -(th_out - psi_out) * (params.Rui / params.Rb) # FIND THE CORRECT Ri
    Dphi_out = -(Dth_out - Dpsi_out) * (params.Rui / params.Rb)
    mode_out = 3

    return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out], dtype=np.float64)

def update_all(state, t_i, double_hoop=False,debug=False):
    """
    Updates the environment's state no matter what and
    handles transitions

    Parameters
    ----------
    t_i : float
        Un-normalized torque applied to the hoop

    double_hoop : bool
        Whether we want the inner hoop

    debug : bool
        Whether to print auxilliary information

    Returns
    -------
    state : np.ndarray
        The state of the environment after applying the torque
    """
    # check to see if the state should transition
    if state[8] == 1:
        state = transition.outer_to_free(state, debug=debug)
        
    elif state[8] == 2:
        state = transition.free_to_outer(state, debug=debug)
        
        if double_hoop:
            try:
                state = transition.free_to_inner(state)
            except AssertionError:
                pass
        
    elif state[8] == 3:
        state = transition.inner_to_free(state)
     
    # update the state based on the meta-state
    if state[8] == 1:
        state = update_outer(state, t_i)
        
    if state[8] == 2:
        state = update_free(state, t_i)
        
    if double_hoop and state[8] == 3:    
        state = update_inner(state, t_i)
        
    return state
