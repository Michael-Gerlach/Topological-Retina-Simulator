import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import importlib
import time
import nest
import sys
import os
from tqdm import tqdm

import Parameters as P
import Retina as R

# This file is calculates the allowed connections that should be made to build the Network,

importlib.reload(P)
importlib.reload(R)


nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": 4})
plt.close('all')
starttime = time.time()


# Define what to iterate over:
subs = [1,3,7]

# Load Parameters:
conn_in_inner_RF = P.conn_in_inner_RF
conn_in_outer_RF = P.conn_in_outer_RF
horizontal_inputs = P.horizontal_inputs

for sub in subs:
    # Set working folder: ----------------------------------------------------------------------------------------------
    subfolder = 'Network/inner_RF/low_horizontal_weight/'+str(sub)+'/'

    conn_in_inner_RF = sub


    print('')
    # Retinal Topology: ------------------------------------------------------------------------------------------------
    # compute the sensor positions from 'Retina.py':
    sensor_x, sensor_y, sensor_type, rod_pos, S_cone_pos, M_cone_pos = R.sensors(P.num_rods,
                                                                                 P.num_S_cones,
                                                                                 P.num_M_cones)

    # Convert sensor x and y into a single array:
    sensor_pos = []
    for i in range(len(sensor_x)):
        sensor_pos.append([sensor_x[i], sensor_y[i]])
    sensor_pos = np.array(sensor_pos)
    sensor_pos = sensor_pos.astype(np.float)

    # compute the RGC and horizontal positions from 'Retina.py':
    RGC_pos = R.RGCs(P.num_RGCs)
    RGC_pos = np.array(RGC_pos)

    horizontal_pos = R.horizontal(P.num_horizontal_cells)
    horizontal_pos = np.array(horizontal_pos)

    # Compute Connections: ---------------------------------------------------------------------------------------------
    print('\n'+50*'-'+'\nstart after:'+str(time.time()-starttime))

    list_of_sensors = np.arange(0, len(sensor_pos))

    # compute connections of RGCs to sensors:
    print('Computing sensor to RGC connections:')
    sensor_to_RGC = np.zeros([1, 3])
    max_dist_inner_RF = []
    for i in tqdm(range(P.num_RGCs)):
        list_of_i = np.full(len(sensor_pos), i, dtype=int)

        distance = np.sqrt(np.sum((RGC_pos[i] - sensor_pos) ** 2, axis=1))
        distance_array = np.dstack((list_of_i, list_of_sensors, distance))[0]

        distance_array_sorted = distance_array[distance_array[:, 2].argsort()]
        distance_array_short = distance_array_sorted[0:conn_in_inner_RF]
        max_distance = max(distance_array_short[:, 2])

        sensor_to_RGC = np.append(sensor_to_RGC, distance_array_short, axis=0)
        max_dist_inner_RF.append(max_distance)

    # delete the unnecessary 1st line of array that results from np.zeros:
    sensor_to_RGC = np.delete(sensor_to_RGC, 0, 0)



    # compute connections of sensors to horizontal cells:
    print('Computing sensor to horizontal connections:')
    sensor_to_horizontal = np.zeros([1, 3])
    max_dist_horizontal_RF = []
    for i in tqdm(range(P.num_horizontal_cells)):
        list_of_i = np.full(len(sensor_pos), i, dtype=int)

        distance = np.sqrt(np.sum((horizontal_pos[i] - sensor_pos) ** 2, axis=1))
        distance_array = np.dstack((list_of_i, list_of_sensors, distance))[0]

        distance_array_sorted = distance_array[distance_array[:, 2].argsort()]
        distance_array_short = distance_array_sorted[0:conn_in_outer_RF]
        max_distance = max(distance_array_short[:, 2])

        sensor_to_horizontal = np.append(sensor_to_horizontal, distance_array_short, axis=0)
        max_dist_horizontal_RF.append(max_distance)

    # delete the unnecessary 1st line of array that results from np.zeros:
    sensor_to_horizontal = np.delete(sensor_to_horizontal, 0, 0)

    # compute connections of horizontal cells to RGCs:
    print('Computing horizontal cell to RGC connections:')
    horizontal_to_RGC = np.zeros([1, 3])
    max_dist_horizontal_input = []
    max_dist_outer_RF = []
    list_of_horizontal_cells = np.arange(0, len(horizontal_pos))
    for i in tqdm(range(P.num_RGCs)):
        list_of_i = np.full(len(horizontal_pos), i, dtype=int)

        distance = np.sqrt(np.sum((RGC_pos[i] - horizontal_pos) ** 2, axis=1))
        distance_array = np.dstack((list_of_i, list_of_horizontal_cells, distance))[0]

        distance_array_sorted = distance_array[distance_array[:, 2].argsort()]
        distance_array_short = distance_array_sorted[0:horizontal_inputs]
        max_distance = max(distance_array_short[:, 2])

        horizontal_to_RGC = np.append(horizontal_to_RGC, distance_array_short, axis=0)
        max_dist_horizontal_input.append(max_distance)

        # Calculate the outer receptive field:
        distance_to_outer_sensor = []
        for j in distance_array_short:
            connection_index = np.where(sensor_to_horizontal[:, 0] == j[1])
            connected_sensors = sensor_to_horizontal[connection_index, 1][0]
            distances = []
            for c in connected_sensors:
                distance = np.sqrt(np.sum((RGC_pos[i] - sensor_pos[int(c)]) ** 2))
                distances.append(distance)

            distance_to_outer_sensor.append(distances)
        max_dist_outer_RF.append(max(distance_to_outer_sensor))

    # delete the unnecessary 1st line of array that results from np.zeros:
    horizontal_to_RGC = np.delete(horizontal_to_RGC, 0, 0)

    # Save file
    if not os.path.exists('./'+subfolder):
        os.makedirs('./'+subfolder)

    np.savez('./'+subfolder+'positions',
             # saves as .npz -file.
             sensor_x=sensor_x,
             sensor_y=sensor_y,
             sensor_type=sensor_type,
             sensor_pos=sensor_pos,
             rod_pos=rod_pos,
             S_cone_pos=S_cone_pos,
             M_cone_pos=M_cone_pos,
             RGC_pos=RGC_pos,
             horizontal_pos=horizontal_pos,
             sensor_to_RGC=sensor_to_RGC,
             sensor_to_horizontal=sensor_to_horizontal,
             horizontal_to_RGC=horizontal_to_RGC,
             max_dist_inner_RF=max_dist_inner_RF,
             max_dist_outer_RF=max_dist_outer_RF,
             max_dist_horizontal_RF=max_dist_horizontal_RF,
             max_dist_horizontal_input=max_dist_horizontal_input,
             conn_in_inner_RF=conn_in_inner_RF,
             conn_in_outer_RF=conn_in_outer_RF,
             horizontal_inputs=horizontal_inputs
             )



stoptime = time.time()
print("\n \nScript took " + str(stoptime-starttime)+" seconds")
print('\a')

'''
# Stop here: -----------------------------------------------------------------------------------------------------------
stoptime = time.time()
print("\n \nSimulation took " + str(stoptime-starttime)+" seconds")
print('\n' + 50*'-')
print('\a')
sys.exit('Exit...')
'''
