import params
import numpy as np

# state is a 9-d vector
# it goes th Dth psi Dpsi r Dr phi Dphi mode

def outer_to_free(state, eps=-0.25, debug=False):

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

    gamma = -params.g * np.cos(psi) - (params.Ro - params.Rb) * (Dpsi ** 2)

    if debug:
        print(f'outer_to_free_gamma: {gamma}')

    if gamma > eps:
        
        print('transitioning OUTER to FREE')
        th_out = th
        Dth_out = Dth
        psi_out = psi
        Dpsi_out = Dpsi
        r_out = params.Ro - params.Rb
        Dr_out = 0.
        phi_out = (th - psi) * (params.Ro / params.Rb)
        Dphi_out = (params.Ro + params.Rb) / params.Rb * Dth - params.Ro / params.Rb * Dpsi
        mode_out = 2

        return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out])

    return state
        
def free_to_outer(state, eps=1e-5, debug=False):

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

    gamma = r - params.Ro + params.Rb

    if debug:
        print(f'free_to_outer_gamma: {gamma}')

    if gamma > eps:
        print('transitioned from FREE to OUTER')
        th_out = th
        Dth_out = Dth
        psi_out = psi
        Dpsi_out = Dth - (params.Rb / params.Ro) * Dphi + Dpsi
        r_out = params.Ro - params.Rb
        Dr_out = 0
        phi_out = phi
        Dphi_out = Dphi
        mode_out = 1

        return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out])

    return state

def inner_to_free(state, eps=0.):

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

    if params.g * np.cos(psi) + (params.Rui + params.Rb) * Dpsi**2 > eps:

        th_out = th
        Dth_out = Dth
        psi_out = psi
        Dpsi_out = Dpsi
        r_out = params.Rui + params.Rb
        Dr_out = 0
        phi_out = -(th - psi) * (params.Rui / params.Rb)
        Dphi_out = -((params.Rui - params.Rb) / params.Rb * Dth - params.Rui / params.Rb * Dpsi)
        mode_out = 2

        return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out])

    return state

def free_to_inner(state, eps=0.):

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

    if params.Rui + params.Rb - r > eps:
        
        th_out = th
        Dth_out = Dth
        psi_out = psi
        Dpsi_out = Dth + params.Rb / params.Rui * Dphi + Dpsi
        r_out = params.Rui + params.Rb
        Dr_out = 0
        phi_out = phi
        Dphi_out = Dphi
        mode_out = 3

        return np.asarray([th_out, Dth_out, psi_out, Dpsi_out, r_out, Dr_out, phi_out, Dphi_out, mode_out])

    return state