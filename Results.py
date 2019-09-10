import numpy as np
import matplotlib.pyplot as plt
import re
import os
import time

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# This file loads result files from Network.py and plots the RGC amplitudes and histograms:
# ----------------------------------------------------------------------------------------------------------------------
# Convert grid to cpd:
def cpd(g):
    return(5000/(180*g))

# Define what to iterate over:
Folder = 'Network/shape/'
subs = next(os.walk(Folder))[1]

#subs=[]

for sub in subs:
    # Set working folder: ----------------------------------------------------------------------------------------------
    subfolder = Folder+str(sub)+'/'


    print('')
    print('-'*50)
    print('')

    print('Running Computation for subfolder: ' + subfolder)

    plt.close('all')

    resultfile = np.load(subfolder+'results.npz', allow_pickle=True)
    RGC_pos = resultfile['RGC_pos']
    amplitude = resultfile['results'][()]
    phase = resultfile['phase'][()]
    antiphase = resultfile['antiphase'][()]
    parameter_dict = resultfile['parameter_dict'][()]
    result_keys = []
    result_keys_cpd = []
    cpd_grid_dict = dict([])
    grid_cpd_dict = dict([])
    result_dict = dict([])
    result_dict_cpd = dict([])
    for k in resultfile['grids']:
        key = 'grid '+str(k)
        key_cpd = 'cpd = '+str(np.round(cpd(k),3))
        result_keys.append(key)
        result_keys_cpd.append(key_cpd)
        dictionary = dict([(key, amplitude[key])])
        dictionary_cpd = dict([(key_cpd, amplitude[key])])
        result_dict.update(dictionary)
        result_dict_cpd.update(dictionary_cpd)
        cpd_grid_dict.update(dict([(key_cpd,key)]))
        grid_cpd_dict.update(dict([(key,key_cpd)]))

    # Print contents of result file: -----------------------------------------------------------------------------------
    print('Contents of result file:')
    for r in resultfile:
        print(r)

    print('')
    print('Contents of parameter_dict:')
    for p in parameter_dict:
        try:
            if len(parameter_dict[p]) > 10:
                print(p+' = '+str(parameter_dict[p][0:3])+'...')
        except:
            print(p + ' = ' + str(parameter_dict[p]))

    # Calculate receptive fields: --------------------------------------------------------------------------------------
    max_dist_inner_RF = parameter_dict['max_dist_inner_RF']
    max_dist_outer_RF = parameter_dict['max_dist_outer_RF']
    max_dist_horizontal_RF = parameter_dict['max_dist_horizontal_RF']
    max_dist_horizontal_input = parameter_dict['max_dist_horizontal_input']

    avg_dist_inner_RF = np.mean(max_dist_inner_RF)
    avg_dist_outer_RF = np.mean(max_dist_outer_RF)
    avg_dist_horizontal_RF = np.mean(max_dist_horizontal_RF)
    avg_dist_horizontal_input = np.mean(max_dist_horizontal_input)

    # Print Average
    print('')
    print('Average max radius...')
    print('of inner receptive field: ' + str(avg_dist_inner_RF)+' mm')
    print('of outer receptive field: ' + str(avg_dist_outer_RF)+' mm')
    print('of horizontal input: ' + str(avg_dist_horizontal_RF)+' mm')
    print('of horizontal cells connected to the RGCs: ' + str(avg_dist_horizontal_input)+' mm')

    # Calculate the mean amplitude for every stimulus: -----------------------------------------------------------------
    grid_values = []
    mean_amplitudes = []
    for k in result_keys:
        mean = np.mean(amplitude[k])
        mean_amplitudes.append(mean)
        grid_values.append(int((re.findall('\d+', k))[0]))

    cpd_values = cpd(np.array(grid_values))

    # Create directory
    if not os.path.exists('./'+subfolder+'results/'):
        os.makedirs('./'+subfolder+'results/')

    '''
    # Plot and save Response Amplitude Histogrem:
    for k in result_keys_cpd:
        figtitle = 'Histogram_for_'+cpd_grid_dict[k]
        plt.title('Histogram for '+k)
        plt.figure(figtitle, figsize=(6, 6))
        plt.hist(result_dict_cpd[k], 15)
        plt.xlabel('Number of Spikes')
        plt.ylabel('Number of RGCs')
        plt.savefig(subfolder + 'results/' +figtitle, dpi=300)
        plt.show()
    
    # Plot and save RGC amplitude
    for k in result_keys_cpd:
        figtitle = 'Amplitude_for_'+cpd_grid_dict[k]
        plt.title('Activity Map for '+k)
        plt.figure(figtitle, figsize=(6, 6))
        plt.scatter(RGC_pos[:, 0], RGC_pos[:, 1], s=2, c=result_dict_cpd[k])
        plt.colorbar()
        plt.axes().set_aspect('equal')
        plt.savefig(subfolder + 'results/' +figtitle, dpi=300)
        plt.show()
    '''
    
    # Plot and save mean amplitude for every stimulus:
    figtitle = 'Mean_amplitudes_grid'
    plt.figure(figtitle, figsize=(6, 6))
    plt.title('Mean Amplitudes for Different Grid Values')
    plt.scatter(grid_values, mean_amplitudes)
    plt.xlabel('Grid Value')
    plt.ylabel('Mean Amplitude')
    plt.savefig(subfolder + 'results/' + figtitle, dpi=300)
    plt.show()

    figtitle = 'Mean_amplitudes_cpd'
    plt.figure(figtitle, figsize=(6, 6))
    plt.title('Mean Amplitudes for Different cpd Values')
    plt.xlabel('cpd Value')
    plt.ylabel('Mean Amplitude')
    plt.scatter(cpd_values, mean_amplitudes)
    plt.savefig(subfolder + 'results/' + figtitle, dpi=300)
    plt.show()
    #'''


    # Plot and save first phase response:
    '''
    for k in result_keys:
        figtitle = 'Response_to_ '+k
        plt.figure(figtitle, figsize=(6, 6))
        plt.scatter(RGC_pos[:, 0], RGC_pos[:, 1], s=2, c=phase[k])
        plt.colorbar()
        plt.axes().set_aspect('equal')
        plt.savefig(subfolder + 'results/' + figtitle, dpi=300)
        plt.show()
    '''



print('\a')