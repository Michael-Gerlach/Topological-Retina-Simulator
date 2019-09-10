import numpy as np

# Retinal Topology: ----------------------------------------------------------------------------------------------------
num_rods = 100000
num_S_cones = int(0.015*num_rods)
num_M_cones = int(0.015*num_rods)
num_sensors = num_rods + num_S_cones + num_M_cones

num_RGCs = 45000
num_horizontal_cells = 18000

#sensors_per_RF = 100
diam_retina = 5.
rad_VF = diam_retina/2  # radius of visual field [mm]

# number of sensors the RGCs are connected to directly:
# for 100k: 20
conn_in_inner_RF = int(num_rods/5000)

# number of sensors the RGCs are connected to via horizontal cells:
# for 100k: 10
conn_in_outer_RF = int(0.5*conn_in_inner_RF)

# number of horizontal cells that connect to each RGC:
horizontal_inputs = 10

# delay of horizontal input to RGC:
horizontal_delay = 1


# Canvas Parameters: ---------------------------------------------------------------------------------------------------
resolution = 5000


# Experimental Setup: --------------------------------------------------------------------------------------------------
# Choose between: 'TFT', 'LED'
Signal_source = 'LED'

# LED Emission Peaks m1, m2, m3:
m1 = 350
m2 = 450
m3 = 550

# Signal Parameters: ---------------------------------------------------------------------------------------------------
Signal_dict = dict([
    ('LED1', (61, 0, 61)),
    ('LED2', (0, 46, 255)),
    ('LED3', (163, 255, 0)),
    ('TFT1', (0, 0, 255)),
    ('TFT2', (0, 255, 0)),
    ('TFT3', (255, 0, 0))
])

# Network Specifications: ----------------------------------------------------------------------------------------------
# Factors to account for the 10-fold higher number of release sites in cones compared to rods and the absolute light
# gathering capabilities:
rod_factor = 1.
S_cone_factor = 10.
M_cone_factor = 10.

J_e = 100.  # [mV]
g = 1.  # inhibition scaling factor for syn strength
J_i = -g*J_e
J_RGC = -10.
syn_delay = 0.1  # [ms]

#model parameters for 'iaf_psc_delta':
RGC_model = 'iaf_psc_alpha'
horizontal_model = 'iaf_psc_alpha'

RGC_params = {
    'E_L': -70.0,  # -70 mV is default. Resting membrane potential in mV.
    'V_th': -55.0,  # -55mV is default. Spike threshold in mV.
    'V_reset': -70.0,  # -70.0 is default. Reset potential of the membrane in mV
    'tau_m': 10.0,  # 10 ms is default. Membrane time constant in ms
    't_ref': 2.0,  # 2ms is default. Duration of refractory period in ms.
    'I_e': 0.0,  # Constant input current in pA
    'V_min': -100.0  # V_min double - Absolute lower value for the membrane potential in mV
                }

horizontal_params = {
    'E_L': -70.0,  # -70 mV is default. Resting membrane potential in mV.
    'V_th': -55.0,  # -55mV is default. Spike threshold in mV.
    'V_reset': -70.0,  # -70.0 is default. Reset potential of the membrane in mV
    'tau_m': 10.0,  # 10 ms is default. Membrane time constant in ms
    't_ref': 2.0,  # 2ms is default. Duration of refractory period in ms.
    'I_e': 0.0,  # Constant input current in pA
    'V_min': -100.0  # V_min double - Absolute lower value for the membrane potential in mV
                }

# Connection dictionary:
conn_dict = {'rule': 'one_to_one'}

rod_delay = syn_delay
S_cone_delay = syn_delay
M_cone_delay = syn_delay

# between horizontal cells and RGCs:
horizontal_RGC_synapse_dict_inh={
    'weight': J_RGC,
    'delay': horizontal_delay #ms
                }

# Simulation parameters: -----------------------------------------------------------------------------------------------
wuptime = 50.0
simtime = 100.0
