# PORTED FROM https://github.com/aa4cc/flying-ball-in-hoop/blob/master/m/params_init.m

import numpy as np

# h is the timestep in seconds
h = 0.002

# Physical Parameters
R_oring = 1.5e-3 # diameter of the outer ring
d_orings = 14.7e-3 # distance between the orings

Rb_real = 0.0246 / 2.0 # radius of the ball
h_new = np.sqrt((Rb_real + R_oring)**2 - (d_orings / 2.0)**2) # perpendicular distance from the center of the ball to the line going through the centers of the orings
d_rotAxis2oring_outer = 203.828 / 2.0 * 1e-3 # distance from the center of the hoop (rotation axis) to the center of the outer oring;
d_rotAxis2oring_Uinner = 89.0 / 2.0 * 1e-3 # distance from the center of the hoop (rotation axis) to the center of the outer oring in the inner part of the ushape
d_rotAxis2oring_Uouter = 111.0 / 2.0 * 1e-3 # distance from the center of the hoop (rotation axis) to the center of the outer oring in the outer part of the ushape

Rb = h_new * Rb_real / (Rb_real + R_oring) # efective radius of the ball
Ro = d_rotAxis2oring_outer - (h_new - Rb) # rolling radius of the outer hoop
Rui = d_rotAxis2oring_Uinner - (h_new - Rb) # rolling radius of the ball in the inner ushape
Ruo = d_rotAxis2oring_Uouter + (h_new - Rb) # rolling radius of the ball in the outer ushape

m = 0.0608 # mass of the metal ball
Ib = 2/5 * m * Rb_real**2 # moment of inertia of the ball
b = 1.394e-6 # some friction coefficient??
g = 9.81

ath1 =  -0.4800
ath2 = -5.1195
ath3 =   0.0677
bth = 586.3695
apsi1 =   -0.0034
apsi2 = -73.6336
apsi3 = -0.3351
bpsi = 210.12

a_bar = Ib * (Ro ** 2)/(Rb ** 2) + m * (Ro - Rb) **2
b_bar = b * (Ro / Rb) ** 2
c_bar = 0.0298 # this was apparently better
d_bar = -b_bar
e_bar = Ib * (Ro / Rb) * (Ro / (Rb + 1))
