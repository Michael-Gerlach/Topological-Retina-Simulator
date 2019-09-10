import numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy.optimize import curve_fit
from scipy import stats
import time

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# This file loads result files from subfolders and performs analisations:
# ----------------------------------------------------------------------------------------------------------------------

# Convert grid to cpd:
def cpd(g):
    return(5000/(180*g))


# Define what to iterate over:
Folder = 'Network/shape/'
iteration_parameter = 'Shape of the Signal'
subs = next(os.walk(Folder))[1]

# Define dictionaries: -------------------------------------------------------------------------------------------------
amplitude_dict = dict([])

'''
# dict to name the keys in horizontal_weights
weight_factor_dict = dict([
    ('one_hundredth', 0.01),
    ('one_twentieth', 0.05),
    ('one_tenth', 0.1),
    ('one_fifth', 0.2),
    ('one', 1),
    ('five', 5),
    ('ten', 10),
    ('fifteen', 15),
    ('twenty', 20),
    ('twentyfive', 25),
    ('fifty', 50),
    ('hundred', 100)])
#'''

for sub in subs:
    # Set working folder: ----------------------------------------------------------------------------------------------
    subfolder = Folder+str(sub)+'/'

    fig_size = (8, 6)

    print('')
    print('Reading out results for subfolder: ' + subfolder)
    print('-'*50)
    print('')
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
    grid_values = []
    for k in resultfile['grids']:
        grid_values.append(k)
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
    cpd_values = cpd(np.array(grid_values))

    # Calculate the mean amplitude for every stimulus: -----------------------------------------------------------------
    mean_amplitudes = []
    for k in result_keys:
        mean = np.mean(amplitude[k])
        mean_amplitudes.append(mean)

    amplitude_sub_array = np.array([grid_values, mean_amplitudes])

    #sub_factor = weight_factor_dict[sub]
    #sub_dict = dict([(sub_factor, amplitude_sub_array)])
    #sub_dict = dict([(int(sub), amplitude_sub_array)])
    sub_dict = dict([(sub, amplitude_sub_array)])

    amplitude_dict.update(sub_dict)

# Sort Amplitude Dictionary:
amplitude_dict_sorted = dict([])
for key, value in sorted(amplitude_dict.items(), key=lambda item: item[0]):
    amplitude_dict_sorted.update(dict([(key,value)]))

# Define the fit to be used: -------------------------------------------------------------------------------------------
# tanh:
def regfunc(x, a, b, c):
    return a+(1-a)*((1+np.tanh(b*x-c))/2)

'''
figtitle = 'Response Amplitudes Compared for different inner Receptive field sizes'
plt.figure('Mean Response Amplitudes', figsize=(12, 8))
for k in amplitude_dict:
    grid = amplitude_dict[k][0]
    response = amplitude_dict[k][1]
    plt.scatter(grid, response, label=k)
plt.title(figtitle)
plt.legend()
plt.show()
'''

resolution_dict = dict([])

figtitle = 'Normalized Response Amplitudes'
plt.figure(figtitle, figsize=fig_size)
ax = plt.gca()
for k in amplitude_dict_sorted:
    print('')
    print('Plotting results for subfolder: '+subfolder)
    # load values:
    grid = amplitude_dict_sorted[k][0]
    response = amplitude_dict_sorted[k][1]
    response_norm = response / max(response)
    # Deflag if unnormalized response is desired:
    #response_norm = response
    x = np.linspace(min(grid), max(grid), 1000)

    # calculate regression:
#    '''
    fitparams = curve_fit(regfunc, grid, response_norm, p0=[min(response_norm), 0, 0])[0]

    # calculate steepest point of curve:
    fitted_curve = regfunc(x, *fitparams)
    gradient = np.gradient(fitted_curve)
    x_idx = np.where(gradient == max(gradient))
    x_pos = x[x_idx]
    y_pos = fitted_curve[x_idx]

    # read out value from dictionary key:
    # if string:
    #iteration_value = int((re.findall('\d+', k))[0])
    #resolution_dict.update(dict([(iteration_value, x_pos[0])]))

    # if int:
    resolution_dict.update(dict([(k, x_pos[0])]))
#    '''

    # plot results:
    # fix colors for every iteration value:
    color = next(ax._get_lines.prop_cycler)['color']

    # connected scatterplot if no tanh fit is possible:
#    connected = plt.plot(grid, response_norm, marker='o', label=iteration_parameter+': '+str(k), color=color)
#    '''

    # scatterplot of the mean amplitudes:
    scatter = plt.scatter(grid, response_norm,
                          label=
                          iteration_parameter+': '+str(k)+'\nMax gradient at: '+str(np.round(x_pos[0], decimals=1)),
                          #s=20,
                          color=color)

    # fitted tanh curve.
    plot = plt.plot(x, fitted_curve, color=color)

    # Vertical line indicating position of highest gradient:
    try:
        vline = plt.axvline(x_pos, 0, 1, color=color)
    except ValueError:
        pass
#    '''

plt.title(figtitle)
plt.legend(loc='lower right')
plt.xlabel('Grid Constant')
plt.ylabel('Normalized Response')
plt.savefig(Folder + 'response_amplitude', dpi=300, bbox_inches="tight")
plt.show()

resolution_dict = dict(sorted(resolution_dict.items()))
resolution_keys = []
resolution_values = []
resolution_values_cpd = []
for i in resolution_dict:
    resolution_keys.append(int(i))
    resolution_values.append(resolution_dict[i])
    resolution_values_cpd.append(cpd(resolution_dict[i]))


# Plot of the max gradient in dependency of the iteration value:
#'''
x = np.linspace(min(resolution_keys), max(resolution_keys), 1000)

m, b, r_value, p_value, std_err = stats.linregress(resolution_keys, resolution_values)
plottitle = 'Max Gradient over ' + iteration_parameter + ' for Grid Value'
plt.figure(plottitle, figsize=fig_size)
plt.title(plottitle)
plt.scatter(resolution_keys, resolution_values, label='Locations of steepest gradient')
plt.plot(x, m * x + b, label='linear fit \ngradient=' + str(np.round(m, 4)) + '±' + str(np.round(std_err, 4)))
plt.xlabel(iteration_parameter)
plt.ylabel('Max Gradient of Response Amplitude [Grid Value]')
plt.legend()
plt.savefig(Folder + 'max_grad_grid', dpi=300, bbox_inches="tight")
plt.show()

m, b, r_value, p_value, std_err = stats.linregress(resolution_keys, resolution_values_cpd)
plottitle = 'Max Gradient over ' + iteration_parameter+' for cpd Value'
plt.figure(plottitle, figsize=fig_size)
plt.title(plottitle)
plt.scatter(resolution_keys, resolution_values_cpd, label='Locations of steepest gradient')
#plt.plot(x, m * x + b, label='linear fit \ngradient=' + str(np.round(m, 4)) + '±' + str(np.round(std_err, 4)))
plt.xlabel(iteration_parameter)
plt.ylabel('Max Gradient of Response Amplitude [cpd]')
plt.legend()
plt.savefig(Folder + 'max_grad_cpd', dpi=300, bbox_inches="tight")
plt.show()
#'''

