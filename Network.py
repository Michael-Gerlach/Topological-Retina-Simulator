import nest
import nest.topology as tp

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib
from tqdm import tqdm

import Parameters as P
import Input_Signal as IPS

importlib.reload(P)
importlib.reload(IPS)

starttime = time.time()
plt.close('all')

# Define what to iterate over:
Folder = 'Network/Color/'

# Define what to iterate over:
subs = next(os.walk(Folder))[1]

subs = ['3_0']

# define the horizontal weight factor for low = 1 or high = 20 weight simulation
weight_factor = 1

for sub in subs:
    # Set working folder: ----------------------------------------------------------------------------------------------

    subfolder = Folder+str(sub)+'/'


    '''
    # Scale weights of connections to RGCs, to keep RGC input constant:
    # Deactivated until now
    RGC_connection_factor = 20/sub
    RGC_horizontal_factor = 10/sub
    '''

    print('Running Computation for subfolder: ' + subfolder)

    # Set up the Kernel and load dependencies: -------------------------------------------------------------------------
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 4,
                          'overwrite_files': True,
                          # Whether to print progress information during the simulation
                          'print_time': False,
                          # Seed for global random number generator used synchronously by all virtual processes to create,
                          # e.g., fixed fan-out connections (write only).  ## Zero seems to be default!
                          'grng_seed': 0
                          # 'data_path':'./plot/fresh_out/data'
                          })

    # Set seed for numpy random:
    np.random.seed = 0

    # Load sensor and RGC positions:
    Retinafile = np.load(subfolder+'positions.npz')

    sensor_pos = Retinafile['sensor_pos']
    sensor_type = Retinafile['sensor_type']
    RF_pos = Retinafile['RGC_pos']

    sensor_to_RGC = Retinafile['sensor_to_RGC']
    sensor_to_horizontal = Retinafile['sensor_to_horizontal']
    horizontal_to_RGC = Retinafile['horizontal_to_RGC']

    rod_pos = Retinafile['rod_pos']
    S_cone_pos = Retinafile['S_cone_pos']
    M_cone_pos = Retinafile['M_cone_pos']
    RGC_pos = Retinafile['RGC_pos']
    horizontal_pos = Retinafile['horizontal_pos']

    max_dist_inner_RF = Retinafile['max_dist_inner_RF']
    max_dist_outer_RF = Retinafile['max_dist_outer_RF']
    max_dist_horizontal_RF = Retinafile['max_dist_horizontal_RF']
    max_dist_horizontal_input = Retinafile['max_dist_horizontal_input']
    conn_in_inner_RF = Retinafile['conn_in_inner_RF']
    conn_in_outer_RF = Retinafile['conn_in_outer_RF']
    horizontal_inputs = Retinafile['horizontal_inputs']

    rod_pos = np.ndarray.tolist(rod_pos[0])
    S_cone_pos = np.ndarray.tolist(S_cone_pos[0])
    M_cone_pos = np.ndarray.tolist(M_cone_pos[0])
    RGC_pos = np.ndarray.tolist(RGC_pos)

    diam_retina = P.diam_retina

    # Calculate the Receptor weight based on number of rods
    scaling_factor = 20/conn_in_inner_RF

    rod_factor = P.rod_factor * P.J_e * scaling_factor
    S_cone_factor = P.S_cone_factor * P.J_e * scaling_factor
    M_cone_factor = P.M_cone_factor * P.J_e * scaling_factor

    # Define sensor positions and type (cone type; exc/inh): -----------------------------------------------------------
    input_sensors = []  # for each neuron: indices of its input sensors
    input_sensors_exc = []
    input_sensors_inh = []


    inputs_to_RF = []
    for i in range(len(RF_pos)):
        inputs_to_RF.append([])
    for i in range(len(sensor_to_RGC)):
        inputs_to_RF[int(sensor_to_RGC[i][0])].append(sensor_to_RGC[i][1])

    # Convert the retina positions to pixels on the array and read out the signal --------------------------------------
    sensor_pos_array = 1000*(sensor_pos+2.5) # to convert from -2.5 to +2.5 mm to 0 to 5000 pixels
    sensor_pos_int = [] # make the position an int to later use it as index for the signal amplitude
    for i in range(len(sensor_pos_array)):
        sensor_pos_int.append([int(sensor_pos_array[i][0]), int(sensor_pos_array[i][1])])


    sensor_responses = np.load('Retina/sensor_responses.npz', allow_pickle=True)

    # '''
    # set up layers and create nodes: ----------------------------------------------------------------------------------
    sensors = nest. Create('dc_generator', int(len(sensor_pos)), params={'amplitude': 0.0})
    RGCs = nest.Create(P.RGC_model, int(len(RGC_pos)), params=P.RGC_params)
    horizontals = nest.Create(P.horizontal_model, int(len(horizontal_pos)), params=P.horizontal_params)
    spikedetector = nest.Create('spike_detector',
                                params={'withgid': True,
                                        'withtime': True,
                                        'to_memory': True,
                                        'to_file': False,
                                        'label': 'spike_det',
                                        'start': 0.}
                                )

    RGC_multimeters = nest.Create('multimeter', 10, params={"withtime": True, "record_from": ["V_m"]})
    horizontal_multimeters = nest.Create('multimeter', 10, params={"withtime": True, "record_from": ["V_m"]})
    horizontal_spikedetector = nest.Create('spike_detector',
                                           params={'withgid': True,
                                                   'withtime': True,
                                                   'to_memory': True,
                                                   'to_file': False,
                                                   'label': 'spike_det',
                                                   'start': 0.}
                                           )

    # Connect network: -------------------------------------------------------------------------------------------------
    nest.Connect(RGCs, spikedetector)
    nest.Connect(RGC_multimeters, RGCs[0:10], 'one_to_one')
    nest.Connect(horizontal_multimeters, horizontals[0:10], 'one_to_one')

    # Build Connection from scratch:

    # Connect sensors to RGCs:
    # Create Lists with elements and connection parameters:
    sensor = [0]*len(sensor_to_RGC)
    RGC = [0]*len(sensor_to_RGC)

    weight = [0]*len(sensor_to_RGC)
    delay = [0]*len(sensor_to_RGC)
    print('Connecting sensors to RGCs:')
    for i in tqdm(range(len(sensor_to_RGC))):
        sensor[i] = sensors[int(sensor_to_RGC[i][1])]
        RGC[i] = RGCs[int(sensor_to_RGC[i][0])]

        if sensor_type[int(sensor_to_RGC[i][1])] == 'rod':
            weight[i] = rod_factor #* RGC_connection_factor
            delay[i] = P.rod_delay

        if sensor_type[int(sensor_to_RGC[i][1])] == 'S_cone':
            weight[i] = S_cone_factor #* RGC_connection_factor
            delay[i] = P.S_cone_delay

        if sensor_type[int(sensor_to_RGC[i][1])] == 'M_cone':
            weight[i] = M_cone_factor #* RGC_connection_factor
            delay[i] = P.M_cone_delay

    #Connect the Network:
    nest.Connect(sensor,
                 RGC,
                 P.conn_dict,
                 syn_spec={'weight': weight, 'delay': delay}
                 )

    # Connect sensors to horizontal cells:
    # Create Lists with elements and connection parameters:
    print('Connecting sensors to horizontal cells:')
    sensor = [0]*len(sensor_to_horizontal)
    horizontal = [0]*len(sensor_to_horizontal)
    weight = [0]*len(sensor_to_horizontal)
    delay = [0]*len(sensor_to_horizontal)
    for i in tqdm(range(len(sensor_to_horizontal))):
        sensor[i] = sensors[int(sensor_to_horizontal[i][1])]
        horizontal[i] = horizontals[int(sensor_to_horizontal[i][0])]

        if sensor_type[int(sensor_to_horizontal[i][1])] == 'rod':
            weight[i] = rod_factor
            delay[i] = P.rod_delay

        if sensor_type[int(sensor_to_horizontal[i][1])] == 'S_cone':
            weight[i] = S_cone_factor
            delay[i] = P.S_cone_delay

        if sensor_type[int(sensor_to_horizontal[i][1])] == 'M_cone':
            weight[i] = M_cone_factor
            delay[i] = P.M_cone_delay

    #Connect the Network:
    nest.Connect(sensor,
                 horizontal,
                 P.conn_dict,
                 syn_spec={'weight': weight, 'delay': delay}
                 )

    # Connect horizontal cells to RGCs:
    # Create Lists with elements and connection parameters:
    print('Connecting horizontal cells to RGCs:')
    horizontal = [0]*len(horizontal_to_RGC)
    RGC = [0]*len(horizontal_to_RGC)
    for i in tqdm(range(len(horizontal_to_RGC))):
        horizontal[i] = horizontals[int(horizontal_to_RGC[i][1])]
        RGC[i] = RGCs[int(horizontal_to_RGC[i][0])]

    #Connect the Network:
    nest.Connect(horizontal,
                 RGC,
                 P.conn_dict,
                 {'weight': P.horizontal_RGC_synapse_dict_inh['weight']*weight_factor,
                  'delay': P.horizontal_RGC_synapse_dict_inh['delay']})

    print('Network build took: ' + str(time.time()-starttime))

    # ------------------------------------------------------------------------------------------------------------------
    # Simulate Network:
    # ------------------------------------------------------------------------------------------------------------------

    # Define grid to loop over:
    grids = [1, 5, 10, 15, 20, 25, 30, 35, 50, 70, 150, 300]
    # grid = [50]
    results = dict([])
    phase = dict([])
    antiphase = dict([])

    # Loop over grid:
    for g in grids:
        print('-'*100)
        print('Subfolder: '+subfolder)
        print('Running Computation for grid '+str(g))
        print('-'*100)

        # Load Intensity-Dictionaries:
        print('Loading the input signal...')
        dummytime = time.time()
        Int_dict, Int_dict_anti = IPS.Input_Signal(g)

        Int_C1 = Int_dict['Color1']
        Int_C2 = Int_dict['Color2']
        Int_C3 = Int_dict['Color3']
        Int_CW = Int_dict['White']

        Int_C1_anti = Int_dict_anti['Color1']
        Int_C2_anti = Int_dict_anti['Color2']
        Int_C3_anti = Int_dict_anti['Color3']
        Int_CW_anti = Int_dict_anti['White']

        # --------------------------------------------------------------------------------------------------------------
        # Simulate Network for Phase:
        # --------------------------------------------------------------------------------------------------------------

        # Set sensor responses: ----------------------------------------------------------------------------------------
        responses = []
        print('Calculating sensor response')
        for i in tqdm(range(len(sensor_type))):
            x = sensor_pos_int[i][0]
            y = sensor_pos_int[i][1]
            r_1 = sensor_responses[sensor_type[i]][()]['Color1'] * Int_C1[x][y]
            r_2 = sensor_responses[sensor_type[i]][()]['Color2'] * Int_C2[x][y]
            r_3 = sensor_responses[sensor_type[i]][()]['Color3'] * Int_C3[x][y]
            r_W = sensor_responses[sensor_type[i]][()]['White'] * Int_CW[x][y]

            response = r_1 + r_2 + r_3 + r_W
            responses.append(response)

        # Simulate Network for wuptime:
        nest.Simulate(P.wuptime)

        # Readout Spikedetector before phase: --------------------------------------------------------------------------
        RGC_senders_before_phase = list(nest.GetStatus(spikedetector)[0]['events']['senders'])
        RGC_times_before_phase = list(nest.GetStatus(spikedetector)[0]['events']['times'])

        # Set voltage according to sensor response: --------------------------------------------------------------------

        print('Defining sensor voltage')
        rates = (scaling_factor/max(responses))*np.array(responses)
        amplitudes = []
        for rate in rates:
            amplitude = dict([['amplitude', rate]])
            amplitudes.append(amplitude)

        nest.SetStatus(sensors, amplitudes)

        # Simulate Network: --------------------------------------------------------------------------------------------
        simstart = time.time()

        nest.Simulate(P.simtime)

        simstop = time.time()
        print('Simulation took: ' + str(simstop-simstart))

        # Readout Spikedetector after phase: ---------------------------------------------------------------------------
        RGC_senders_after_phase = list(nest.GetStatus(spikedetector)[0]['events']['senders'])
        RGC_times_after_phase = list(nest.GetStatus(spikedetector)[0]['events']['times'])

        # --------------------------------------------------------------------------------------------------------------
        # Simulate Network for Antiphase:
        # --------------------------------------------------------------------------------------------------------------

        # Set sensor responses: ----------------------------------------------------------------------------------------
        responses_anti = []
        print('Calculating sensor response')
        for i in tqdm(range(len(sensor_type))):
            x = sensor_pos_int[i][0]
            y = sensor_pos_int[i][1]
            r_1 = sensor_responses[sensor_type[i]][()]['Color1'] * Int_C1_anti[x][y]
            r_2 = sensor_responses[sensor_type[i]][()]['Color2'] * Int_C2_anti[x][y]
            r_3 = sensor_responses[sensor_type[i]][()]['Color3'] * Int_C3_anti[x][y]
            r_W = sensor_responses[sensor_type[i]][()]['White'] * Int_CW_anti[x][y]

            response_anti = r_1 + r_2 + r_3 + r_W
            responses_anti.append(response_anti)

        # Simulate Network for wuptime
        nest.Simulate(P.wuptime)

        # Readout Spikedetector before antiphase: ----------------------------------------------------------------------
        RGC_senders_before_anti = list(nest.GetStatus(spikedetector)[0]['events']['senders'])
        RGC_times_before_anti = list(nest.GetStatus(spikedetector)[0]['events']['times'])

        # Set voltage according to sensor response: --------------------------------------------------------------------
        print('Defining sensor voltage for Antiphase')
        rates_anti = (scaling_factor/max(responses_anti))*np.array(responses_anti)
        amplitudes_anti = []
        for rate in rates_anti:
            amplitude_anti = dict([['amplitude', rate]])
            amplitudes_anti.append(amplitude_anti)

        nest.SetStatus(sensors, amplitudes_anti)

        # Simulate Network: --------------------------------------------------------------------------------------------
        simstart = time.time()

        nest.Simulate(P.simtime)

        simstop = time.time()
        print('Simulation took: ' + str(simstop-simstart))

        # Readout Spikedetector after antiphase: -----------------------------------------------------------------------
        print('Reading out spikedetector...')
        RGC_senders_anti = list(nest.GetStatus(spikedetector)[0]['events']['senders'])
        RGC_times_anti = list(nest.GetStatus(spikedetector)[0]['events']['times'])

        horizontal_spikes = list(nest.GetStatus(horizontal_spikedetector)[0]['events']['senders'])

        # Calculate Flicker Response Difference: -----------------------------------------------------------------------
        dummytime = time.time()
        print('Calculating flicker response...')

        RGC_number_before_phase, RGC_count_before_phase = np.unique(RGC_senders_before_phase, return_counts=True)

        RGC_number_after_phase, RGC_count_after_phase = np.unique(RGC_senders_after_phase, return_counts=True)

        RGC_number_before_anti, RGC_count_before_anti = np.unique(RGC_senders_before_anti, return_counts=True)

        RGC_number_after_anti, RGC_count_after_anti = np.unique(RGC_senders_anti, return_counts=True)

        RGC_count_list_before_phase = []
        for i in RGCs:
            count = RGC_count_before_phase[np.where(RGC_number_before_phase == i)]
            if count.size == 0:
                count = np.array([0])
            RGC_count_list_before_phase.append(count[0])

        RGC_count_list_after_phase = []
        for i in RGCs:
            count = RGC_count_after_phase[np.where(RGC_number_after_phase == i)]
            if count.size == 0:
                count = np.array([0])
            RGC_count_list_after_phase.append(count[0])

        RGC_count_list_before_anti = []
        for i in RGCs:
            count = RGC_count_before_anti[np.where(RGC_number_before_anti == i)]
            if count.size == 0:
                count = np.array([0])
            RGC_count_list_before_anti.append(count[0])

        RGC_count_list_after_anti = []
        for i in RGCs:
            count = RGC_count_after_anti[np.where(RGC_number_after_anti == i)]
            if count.size == 0:
                count = np.array([0])
            RGC_count_list_after_anti.append(count[0])

        RGC_count_array = np.array(RGC_count_list_after_phase)-np.array(RGC_count_list_before_phase)
        RGC_count_anti_array = np.array(RGC_count_list_after_anti)-np.array(RGC_count_list_before_anti)

        RGC_Amplitude = np.absolute(RGC_count_anti_array-RGC_count_array)


        RGC_positions = np.array(RGC_pos)

        Amplitude_dict = dict([('grid ' + str(g), RGC_Amplitude)])
        phase_dict = dict([('grid ' + str(g), RGC_count_array)])
        antiphase_dict = dict([('grid ' + str(g), RGC_count_anti_array)])

        results.update(Amplitude_dict)
        phase.update(phase_dict)
        antiphase.update(antiphase_dict)

        # Plot response amplitude
    '''
        plt.figure('amplitude between phases for grid constant '+str(g), figsize=(12, 8))
        plt.scatter(RGC_positions[:, 0], RGC_positions[:, 1], s=20, c=RGC_Amplitude)
        plt.colorbar()
        plt.axes().set_aspect('equal')
        plt.show()
    '''
    # Todo: Include a Dictionary with the Parameters that are important: Colors, Amplitudes, numbers of Network elements,
    #  Network parameters, type of Input Signal
    parameter_dict = dict([('C1', IPS.C1),
                           ('C2', IPS.C2),
                           ('A1', IPS.A1),
                           ('A2', IPS.A2),
                           ('image_type', IPS.image_type),
                           ('Signal_source', P.Signal_source),
                           ('num_rods', len(rod_pos)),
                           ('num_S_cones', len(S_cone_pos)),
                           ('num_M_cones', len(S_cone_pos)),
                           ('rod_factor', rod_factor),
                           ('S_cone_factor', S_cone_factor),
                           ('M_cone_factor', M_cone_factor),
                           ('RGC_params', P.RGC_params),
                           ('horizontal_params', P.horizontal_params),
                           ('horizontal_delay', P.horizontal_delay),
                           ('max_dist_inner_RF', max_dist_inner_RF),
                           ('max_dist_outer_RF', max_dist_outer_RF),
                           ('max_dist_horizontal_RF', max_dist_horizontal_RF),
                           ('max_dist_horizontal_input', max_dist_horizontal_input),
                           ('conn_in_inner_RF', conn_in_inner_RF),
                           ('conn_in_outer_RF', conn_in_outer_RF),
                           ('horizontal_inputs', horizontal_inputs)
    ])

    # Save file
    if not os.path.exists('./'+subfolder):
        os.makedirs('./'+subfolder)

    # Save Results of Computation:
    np.savez_compressed(subfolder + 'results',
                        # saves as .npz -file.
                        grids=grids,
                        RGC_pos=RGC_pos,
                        results=results,
                        phase=phase,
                        antiphase=antiphase,
                        parameter_dict=parameter_dict)
    # Read results with
    '''
    resultfile = np.load('Network/results.npz', allow_pickle=True)
    resultfile['results'][()]['grid 50']
    '''

    # Histogram: -------------------------------------------------------------------------------------------------------
    '''
    plt.figure('Histogram of Response amplitude', figsize=(12, 6))
    plt.hist(RGC_Amplitude,10)
    plt.show()
    '''

    '''
    # Plot multimeters and spike detectors: ----------------------------------------------------------------------------
    # Multimeters:
    plt.figure('Horizontal-cell multimeters')
    for i in range(len(horizontal_multimeters)):
        dmm = nest.GetStatus([horizontal_multimeters[i]])[0]
        Vms = dmm["events"]["V_m"]
        ts = dmm["events"]["times"]
        plt.plot(ts, Vms, label='Multimeter number: '+str(i))
    plt.legend()
    plt.show()
    
    plt.figure('RGC multimeters')
    for i in range(len(RGC_multimeters)):
        dmm = nest.GetStatus([RGC_multimeters[i]])[0]
        Vms = dmm["events"]["V_m"]
        ts = dmm["events"]["times"]
        plt.plot(ts, Vms, label='Multimeter number: '+str(i))
    plt.legend()
    plt.show()
    '''

    '''
    # Spikedetectors
    plt.figure('sensor spikedetector')
    plt.show()
    
    plt.figure('RGC spikedetector')
    plt.scatter(RGC_senders, RGC_times)
    plt.show()
    '''

    # Make 2d histogram of RGC activity: -------------------------------------------------------------------------------
    ''' 
    plt.figure('Count for first phase', figsize=(12, 6))
    plt.scatter(RGC_positions[:, 0], RGC_positions[:, 1], s=20, c=RGC_count_array)
    plt.axes().set_aspect('equal')
    plt.colorbar()
    plt.show()
    
    plt.figure('Count for second phase', figsize=(12, 6))
    plt.scatter(RGC_positions[:, 0], RGC_positions[:, 1], s=20, c=RGC_count_anti_array)
    plt.axes().set_aspect('equal')
    plt.colorbar()
    plt.show()
    '''

print('')
stoptime = time.time()
print("Time passed: " + str(stoptime-starttime)+" seconds")
print('\a'*5)
