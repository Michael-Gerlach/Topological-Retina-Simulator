import numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy.optimize import curve_fit
from scipy import stats
import time

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# This file calculates the density of the RGCs and their influence on visual acuity
# ----------------------------------------------------------------------------------------------------------------------

# Convert grid to cpd:
def cpd(g):
    return(5000/(180*g))

# Define what to iterate over:
Folder = 'Network/inner_RF/low_horizontal_weight/'
subs = next(os.walk(Folder))[1]

subs = [20]

for sub in subs:
    # Set working folder: ----------------------------------------------------------------------------------------------
    subfolder = Folder+str(sub)+'/'

    fig_size = (8, 6)


    print('')
    print('-'*50)
    print('')

    print('Running Computation for subfolder: ' + subfolder)
    print('\n'*5)

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
    grid = []
    for k in resultfile['grids']:
        grid.append(k)
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

    x = np.linspace(0, 250, 1000)

    # Compare RGC density to the amplitude
    # Define bins:
    xbin = np.linspace(-2.5, 2.5, 21)
    ybin = xbin

    # Calculate and plot RGC density:
    RGC_density, xedges, yedges = np.histogram2d(RGC_pos[:, 0], RGC_pos[:, 1], bins=(xbin, ybin))
    RGC_density = RGC_density.transpose()[::-1]

    plt.figure('RGC_density')
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, RGC_density)
    plt.axes().set_aspect('equal')
    plt.colorbar()
    plt.show()

    # Define function for tanh fit:
    def regfunc(x, a, b, c):
        return a + (1 - a) * ((1 + np.tanh(b * x - c)) / 2)

    # Define funciton for linear regression:
    def linreg(x, m, b):
        return m*x+b

    RGC_density_flat = RGC_density.flatten('C')

    # Calclulate Amplitude density:
    fig1, ax1 = plt.subplots(figsize=fig_size)
    ax1.set_title('Amplitude density over RGC Density')
    fig2, ax2 = plt.subplots(figsize=fig_size)
    ax2.set_title('Average Amplitude per RGC over RGC Density')
    RGC_amplitude_densities = dict([])
    ax = plt.gca()

    # iterate over the grid:
    amplitude_density_dict = dict([])
    for g in amplitude:
        RGC_amplitudes = amplitude[g]

        Amplitude_position_product = []
        for i in range(len(RGC_pos)):
            appendix = [list(RGC_pos[i])] * RGC_amplitudes[i]
            Amplitude_position_product.extend(appendix)

        Amplitude_pos = np.array(Amplitude_position_product)

        Amplitude_density, xedges, yedges = np.histogram2d(Amplitude_pos[:, 0], Amplitude_pos[:, 1], bins=(xbin, ybin))
        Amplitude_density = Amplitude_density.transpose()[::-1]
        Amplitude_density_flat = Amplitude_density.flatten('C')

        amplitude_dict = dict([(g, Amplitude_density)])
        amplitude_density_dict.update(amplitude_dict)

        # Plot the Amplitude density:
        '''
        plt.figure('Amplitude density for: '+g)
        X, Y = np.meshgrid(xedges, yedges)
        plt.pcolormesh(X, Y, Amplitude_density)
        plt.axes().set_aspect('equal')
        plt.colorbar()
        plt.show()
        '''
        # Plot the Amplitude density over RGC density:
        #plt.figure('Amplitude density over RGC density')
        color = next(ax._get_lines.prop_cycler)['color']
        # linear fit:

        m, b, r_value, p_value, std_err = stats.linregress(RGC_density_flat, Amplitude_density_flat)
        ax1.scatter(RGC_density_flat, Amplitude_density_flat, alpha=0.5, s=2, color=color,
                    label=g + '\n'+'gradient='+str(np.round(m, 4))+'±'+str(np.round(std_err, 4)))
        ax1.plot(x, m*x+b, color=color)

        # Plot the individual RGC Amplitude over RGC density:
        #plt.figure('Average Amplitude per RGC over RGC Density')
        individual_Amplitude = []
        for i in range(len(RGC_density_flat)):
            individual_Amplitude.append(Amplitude_density_flat[i]/RGC_density_flat[i])
        individual_Amplitude = [x for x in individual_Amplitude if str(x) != 'nan']
        RGC_density_flat_nozero = [x for x in RGC_density_flat if x != 0]
        # linear fit:
        m, b, r_value, p_value, std_err = stats.linregress(RGC_density_flat_nozero, individual_Amplitude)
        ax2.scatter(RGC_density_flat_nozero, individual_Amplitude, alpha=0.5, s=2, color=color,
                    label=g + '\n'+'gradient='+str(np.round(m, 4))+'±'+str(np.round(std_err, 4)))
        ax2.plot(x, m*x+b, color=color)


    # Create directory to save plots in
    if not os.path.exists('./'+subfolder+'density/'):
        os.makedirs('./'+subfolder+'density/')

    ax1.legend(markerscale=6, bbox_to_anchor=(1.04,1), borderaxespad=0)
    ax1.set_xlabel('RGC density')
    ax1.set_ylabel('Amplitude density')
    #plt.subplots_adjust(right=0.7)
    ax2.legend(markerscale=6, bbox_to_anchor=(1.04,1), borderaxespad=0)
    ax2.set_xlabel('RGC density')
    ax2.set_ylabel('Average RGC Amplitude')
    #plt.subplots_adjust(right=0.7)
    fig1.savefig(subfolder + 'density/Amplitude_density', dpi=300, bbox_inches="tight")
    fig2.savefig(subfolder + 'density/Average_Amplitude_density', dpi=300, bbox_inches="tight")

    plt.show()

    RGC_density_dict = dict([])

    amplitude_density_list = []
    for k in range(len(RGC_density_flat)):
        amplitude_density_list.append([])

    for k in amplitude_density_dict:
        amplitude_density_flat = amplitude_density_dict[k].flatten('C')
        for v in range(len(amplitude_density_flat)):
            amplitude_density_list[v].append(amplitude_density_flat[v])

    for k in range(len(RGC_density_flat)):
        RGC_density_dict.update(dict([[RGC_density_flat[k], amplitude_density_list[k]]]))

    # Compute position of steepest gradient and plot over RGC density:
    max_gradient = np.array([[0,0]])
    for k in RGC_density_dict:
        try:
            Amplitudes = RGC_density_dict[k]

            Amplitudes_norm = Amplitudes / max(Amplitudes)

            # calculate regression:
            fitparams = curve_fit(regfunc, grid, Amplitudes_norm, p0=[min(Amplitudes_norm), 0, 0])[0]

            # calculate steepest point of curve:
            fitted_curve = regfunc(x, *fitparams)
            gradient = np.gradient(fitted_curve)
            x_idx = np.where(gradient == max(gradient))
            x_pos = x[x_idx]
            max_gradient = np.append(max_gradient, [[k, x_pos[0]]], axis=0)

        except:
            pass
    max_gradient = max_gradient[1:]

    m, b, r_value, p_value, std_err = stats.linregress(max_gradient[:,0], max_gradient[:,1])
    plottitle = 'Max Gradient over RGC Density'
    plt.figure(plottitle, figsize=fig_size)
    plt.title(plottitle)
    plt.plot(x, m*x+b,label='gradient='+str(np.round(m, 4))+'±'+str(np.round(std_err, 4)))
    plt.scatter(max_gradient[:, 0], max_gradient[:, 1], label='Maximum Gradient for individual Bins', alpha=0.5)
    plt.xlabel('RGC density')
    plt.ylabel('Max Gradient of Response Amplitude')
    #plt.subplots_adjust(right=0.7)
    plt.legend()
    plt.savefig(subfolder + 'density/maxgrad', dpi=300)
    plt.show()
