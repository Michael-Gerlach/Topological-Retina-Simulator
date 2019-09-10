import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import importlib
import time

import Parameters as P
import standard_template as sttp

# This file calculates the response of the different sensor types to the predefined light stimuli

importlib.reload(P)

plt.close('all')
starttime = time.time()

# Parameters for integration: ------------------------------------------------------------------------------------------
low_limit = 250
high_limit = 700
fitspace = np.linspace(low_limit, high_limit, 1000)


# cones have 10 fold more release sites than rods (Wang, 2014)

rod_factor = P.rod_factor
S_cone_factor = P.S_cone_factor
M_cone_factor = P.M_cone_factor

# Choose between: 'TFT', 'LED'
Signal_source = P.Signal_source

num_rods = P.num_rods
num_S_cones = P.num_S_cones
num_M_cones = P.num_M_cones

P.num_sensors = num_rods + num_S_cones + num_M_cones

# TFT-Monitor emission spectra von Michael.G abgepaust aus LG24GM77 (=Modellname)-Grafik, die Stefan geschickt hatte:
# loading the csv data -------------------------------------------------------------------------------------------------
x_b, y_b = np.loadtxt('LED Display/blue.csv', unpack=True, delimiter=',', skiprows=1)
x_g, y_g = np.loadtxt('LED Display/green.csv', unpack=True, delimiter=',', skiprows=1)
x_r, y_r = np.loadtxt('LED Display/red.csv', unpack=True, delimiter=',', skiprows=1)

def R_TFT_emiss(lambda1):
    return np.interp(lambda1, x_r, y_r)
def G_TFT_emiss(lambda1):
    return np.interp(lambda1, x_g, y_g)
def B_TFT_emiss(lambda1):
    return np.interp(lambda1, x_b, y_b)


# Approximated LED-Emission spectra: -----------------------------------------------------------------------------------
# Standard deviation s:
s = 15

# LED Emission Peaks m1, m2, m3:
m1 = P.m1
m2 = P.m2
m3 = P.m3


# normalized normal distribution:
def normald(v, m, s):
    f = 1 / (s * ((2 * np.pi) ** 0.5)) * np.e ** - (((v - m) ** 2) / (2 * s ** 2))
    return f


# Specify Emission Signal: ---------------------------------------------------------------------------------------------
if Signal_source == 'TFT':
    def emission_1(x):
        emission_1 = R_TFT_emiss(x)
        return emission_1

    def emission_2(x):
        emission_2 = G_TFT_emiss(x)
        return emission_2

    def emission_3(x):
        emission_3 = B_TFT_emiss(x)
        return emission_3

    def emission_4(x):
        emission_4 = R_TFT_emiss(x) + G_TFT_emiss(x) + B_TFT_emiss(x)
        return emission_4

if Signal_source == 'LED':
    def emission_1(x):
        emission_1 = normald(x, m1, s)
        return emission_1

    def emission_2(x):
        emission_2 = normald(x, m2, s)
        return emission_2

    def emission_3(x):
        emission_3 = normald(x, m3, s)
        return emission_3

    def emission_4(x):
        emission_4 = normald(x, m1, s) + normald(x, m2, s) + normald(x, m3, s)
        return emission_4


# Mouse sens-curves according to "Govardovskii (2000)"-template: -------------------------------------------------------
def rod_sens(v):
    rod_sens = sttp.A1(v, 498)
    return rod_sens


def S_cone_sens(v):
    S_cone_sens = sttp.A1(v, 365)
    return S_cone_sens


def M_cone_sens(v):
    M_cone_sens = sttp.A1(v, 512)
    return M_cone_sens


# Sensor Responses: ----------------------------------------------------------------------------------------------------
rod_sens_1 = integrate.quad(lambda x: rod_sens(x) * emission_1(x), low_limit, high_limit, limit=1000)[0]
rod_sens_2 = integrate.quad(lambda x: rod_sens(x) * emission_2(x), low_limit, high_limit, limit=1000)[0]
rod_sens_3 = integrate.quad(lambda x: rod_sens(x) * emission_3(x), low_limit, high_limit, limit=1000)[0]

S_cone_sens_1 = integrate.quad(lambda x: S_cone_sens(x) * emission_1(x), low_limit, high_limit, limit=1000)[0]
S_cone_sens_2 = integrate.quad(lambda x: S_cone_sens(x) * emission_2(x), low_limit, high_limit, limit=1000)[0]
S_cone_sens_3 = integrate.quad(lambda x: S_cone_sens(x) * emission_3(x), low_limit, high_limit, limit=1000)[0]

M_cone_sens_1 = integrate.quad(lambda x: M_cone_sens(x) * emission_1(x), low_limit, high_limit, limit=1000)[0]
M_cone_sens_2 = integrate.quad(lambda x: M_cone_sens(x) * emission_2(x), low_limit, high_limit, limit=1000)[0]
M_cone_sens_3 = integrate.quad(lambda x: M_cone_sens(x) * emission_3(x), low_limit, high_limit, limit=1000)[0]


# ----------------------------------------------------------------------------------------------------------------------
# Defining Signal amplitude:
# Mode dicts:...
'''
# Curve 1 and 3 are in phase to each other with curve 2 180 degree offset
mode_dict = dict([['a', 'S and M Cone silencing, trichromatic'],
                  ['b', 'Rod silencing, dichromatic Channel 1 & 2'],
                  ['c', 'Rod silencing, dichromatic Channel 2 & 3'],
                  # todo: Mode d is not working?
                  ['d', 'Isoluminant, constant global response, same intensity for channel 1 & 2'],
                  ['e', 'Same intensity on all channels']
                  ])
mode = 'b'
print('Stimulus mode: ' + mode_dict[mode])

# Calculate Amplitudes for flickering stimuli or choose 'Same intensity on all channels':
if mode == 'a':
    ampl2 = 1/max(emission_2(fitspace))
    a = np.array([[S_cone_sens_1, S_cone_sens_3], [M_cone_sens_1, M_cone_sens_3]])
    b = np.array([ampl2 * S_cone_sens_2, ampl2 * M_cone_sens_2])
    ampl1, ampl3 = np.linalg.solve(a, b)

if mode == 'b':
    ampl2 = 1/max(emission_2(fitspace))
    ampl1 = ampl2 * rod_sens_2 / rod_sens_1
    ampl3 = 0

if mode == 'c':
    ampl2 = 1/max(emission_2(fitspace))
    ampl1 = 0
    ampl3 = ampl2 * rod_sens_2 / rod_sens_3

if mode == 'd':
    ampl2 = 1 / max(emission_2(fitspace))
    a = np.array([[S_cone_sens_1 * num_S_cones * S_cone_factor +
                   M_cone_sens_1 * num_M_cones * M_cone_factor +
                   rod_sens_1 * num_rods * rod_factor,
                   S_cone_sens_3 * num_S_cones * S_cone_factor +
                   M_cone_sens_3 * num_M_cones * M_cone_factor +
                   rod_sens_3 * num_rods * rod_factor,
                   -(S_cone_sens_2 * num_S_cones * S_cone_factor +
                     M_cone_sens_2 * num_M_cones * M_cone_factor +
                     rod_sens_2 * num_rods * rod_factor)],
                  [0, 0, 1],
                  [1, -1, 0]])

    b = np.array([0, ampl2, 0])

    Ampl1, Ampl3, Ampl2 = np.linalg.solve(a, b)
    ampl1 = ampl2 * Ampl1/Ampl2
    ampl3 = ampl2 * Ampl3/Ampl2


if mode == 'e':
    ampl1 = 1 / max(emission_1(fitspace))
    ampl2 = 1 / max(emission_2(fitspace))
    ampl3 = 1 / max(emission_3(fitspace))


def channel_1(x):
    out = emission_1(x)*ampl1
    return out

def channel_2(x):
    out = emission_2(x)*ampl2
    return out

def channel_3(x):
    out = emission_3(x)*ampl3
    return out
'''

# Fixing the Amplitudes:
#'''
ampl1 = 1 / max(emission_1(fitspace))
ampl2 = 1 / max(emission_2(fitspace))
ampl3 = 1 / max(emission_3(fitspace))
#'''

# Sensor Output: -------------------------------------------------------------------------------------------------------
# response of each sensor to the calculated stimuli:
rod_res_1 = rod_sens_1 * ampl1
rod_res_2 = rod_sens_2 * ampl2
rod_res_3 = rod_sens_3 * ampl3

S_cone_res_1 = S_cone_sens_1 * ampl1
S_cone_res_2 = S_cone_sens_2 * ampl2
S_cone_res_3 = S_cone_sens_3 * ampl3

M_cone_res_1 = M_cone_sens_1 * ampl1
M_cone_res_2 = M_cone_sens_2 * ampl2
M_cone_res_3 = M_cone_sens_3 * ampl3

rod_dict=dict([
    ['Color1', rod_res_1],
    ['Color2', rod_res_2],
    ['Color3', rod_res_3],
    ['White', rod_res_1 + rod_res_2 + rod_res_3]
])

S_cone_dict=dict([
    ['Color1', S_cone_res_1],
    ['Color2', S_cone_res_2],
    ['Color3', S_cone_res_3],
    ['White', S_cone_res_1 + S_cone_res_2 + S_cone_res_3]
])

M_cone_dict=dict([
    ['Color1', M_cone_res_1],
    ['Color2', M_cone_res_2],
    ['Color3', M_cone_res_3],
    ['White', M_cone_res_1 + M_cone_res_2 + M_cone_res_3]
])

np.savez('./Retina/sensor_responses.npz',
         # saves as .npz -file.
         rod=rod_dict,
         S_cone=S_cone_dict,
         M_cone=M_cone_dict
         )



# total output of all sensors into the network:
rod_out_1 = rod_res_1 * num_rods * rod_factor
rod_out_2 = rod_res_2 * num_rods * rod_factor
rod_out_3 = rod_res_3 * num_rods * rod_factor

S_cone_out_1 = S_cone_res_1 * num_S_cones * S_cone_factor
S_cone_out_2 = S_cone_res_2 * num_S_cones * S_cone_factor
S_cone_out_3 = S_cone_res_3 * num_S_cones * S_cone_factor

M_cone_out_1 = M_cone_res_1 * num_M_cones * M_cone_factor
M_cone_out_2 = M_cone_res_2 * num_M_cones * M_cone_factor
M_cone_out_3 = M_cone_res_3 * num_M_cones * M_cone_factor


# Print results of calculations: ---------------------------------------------------------------------------------------

print('Rod output due to stimulation by curve 1: ' + str(rod_res_1))
print('Rod output due to stimulation by curve 2: ' + str(rod_res_2))
print('Rod output due to stimulation by curve 3: ' + str(rod_res_3))

print('S cone output due to stimulation by curve 1: ' + str(S_cone_res_1))
print('S cone output due to stimulation by curve 2: ' + str(S_cone_res_2))
print('S cone output due to stimulation by curve 3: ' + str(S_cone_res_3))

print('M cone output due to stimulation by curve 1: ' + str(M_cone_res_1))
print('M cone output due to stimulation by curve 2: ' + str(M_cone_res_2))
print('M cone output due to stimulation by curve 3: ' + str(M_cone_res_3))

print('Total output for curve 1 and 3: ' + str(S_cone_out_1 + S_cone_out_3 +
                                               M_cone_out_1 + M_cone_out_3 +
                                               rod_out_1 + rod_out_3))
print('Total output for curve 2: ' + str(S_cone_out_2 + M_cone_out_2 + rod_out_2))


'''
# Plot emission and sensitivity curves: --------------------------------------------------------------------------------
plt.close('all')
plt.figure('Emission and Sensitivity Spectra')
plt.title(mode_dict[mode])
plt.fill_between(fitspace, 0, channel_1(fitspace), color='r', alpha=0.5, label='Curve 1')
plt.fill_between(fitspace, 0, channel_2(fitspace), color='g', alpha=0.5, label='Curve 2')
plt.fill_between(fitspace, 0, channel_3(fitspace), color='b', alpha=0.5, label='Curve 3')
plt.xlabel('Wavelength [nm]')
plt.ylabel('relative spectral sensitivity')


#plt.figure('Sensitivity')
plt.plot(fitspace,rod_sens(fitspace), color='k', label='rods')
plt.plot(fitspace,S_cone_sens(fitspace), color='#b412ed', label='S-cones')
plt.plot(fitspace,M_cone_sens(fitspace), color='#019309', label='M-cones')
plt.xlabel('Wavelength [nm]')
plt.ylabel('relative spectral sensitivity')

plt.legend(loc='upper right')
plt.show()
'''

stoptime = time.time()
print("\n \nSimulation took " + str(stoptime-starttime)+" seconds")