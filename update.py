import params
import numpy as np

# state is a 9-d vector
# it goes th Dth psi Dpsi r Dr phi Dphi mode

def update_outer(state, t_i):
    '''
    Update the state of the environment given a torque t_i
    '''
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

    return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out])