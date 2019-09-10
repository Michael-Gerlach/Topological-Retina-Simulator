import nest
import nest.topology as tp
import numpy as np
import time
import matplotlib.pyplot as plt
import importlib
import sys
from tqdm import tqdm

import Parameters as P

importlib.reload(P)

nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": 4})
#plt.close('all')
starttime = time.time()

#np.random.seed(0)

# Define a progress bar: -----------------------------------------------------------------------------------------------
def progressBar(value, endvalue, title, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r'+title+': [{0}] {1}%'.format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


# define cells on retina: ----------------------------------------------------------------------------------------------
# Todo: Add possibility to limit distribution to a definable portion of the area!


# define the gradient and density of the distribution:
diam_retina = P.diam_retina

fitspace = np.linspace(0,0.5*diam_retina,1000)
doublefitspace = np.linspace(-0.5*diam_retina,0.5*diam_retina,2000)

a1, a2, a3, a4, t1, t2, t3, t4, c = 3.06938058e+05, \
                                    -1.63127887e+05, \
                                    -2.36295506e+03, \
                                    -1.45336859e+05, \
                                    -1.02041227e+00, \
                                    -1.08324504e+00,  \
                                    1.18904776e-01, \
                                    -9.43441960e-01, \
                                    3.88936115e+03

# Definition of the PDFs of the different cell types:
def rodfit(x):
    rodfit = a1*np.exp(x*t1)+a2*np.exp(x*t2)+a3*np.exp(x*t3)+a4*np.exp(x*t4)+c
    return rodfit

def S_cone_over(w):
    S_cone_over = 4-(w+1)**2
    return S_cone_over

def RGC_over(t):
    RGC_over = (-0.4 * t + 2)
    return RGC_over

# Function that returns a list of the sensor positions and types
def sensors(num_rods, num_S_cones, num_M_cones):
    rod_pos = []
    S_cone_pos = []
    M_cone_pos = []

    sensor_x = []
    sensor_y = []
    sensor_type = []

    # Rods:
    r = 0
    print('Computing rod positions')
    with tqdm(total=num_rods, initial=r) as pbar:
        while r < num_rods:
            pbar.update(r- pbar.n)
            rod_x = np.random.uniform(-0.5*diam_retina,0.5*diam_retina)
            rod_y = np.random.uniform(-0.5*diam_retina,0.5*diam_retina)
            rod_p = np.random.uniform(0,1.1*max(rodfit(fitspace)))

            if rod_p <= rodfit(np.sqrt(rod_x**2+rod_y**2)) and np.sqrt(rod_x**2+rod_y**2)<=0.5*diam_retina:
                sensor_x.append(rod_x)
                sensor_y.append(rod_y)
                sensor_type.append('rod')
                rod_pos.append([rod_x, rod_y])
                r = r+1

    # S-cones:
    s = 0
    print('Computing S-cone positions')
    with tqdm(total=num_S_cones, initial=s) as pbar:
        while s < num_S_cones:
            pbar.update(s- pbar.n)
            S_cone_x = np.random.uniform(-0.5*diam_retina,0.5*diam_retina)
            S_cone_y = np.random.uniform(-0.5*diam_retina,0.5*diam_retina)
            S_cone_p = np.random.uniform(0,1.1*max(rodfit(np.absolute(doublefitspace))*S_cone_over(doublefitspace)))

            if S_cone_p <= rodfit(np.sqrt(S_cone_x**2+S_cone_y**2))*0.5*(S_cone_over(S_cone_x)+S_cone_over(S_cone_y)) \
                    and np.sqrt(S_cone_x**2+S_cone_y**2) <= 0.5*diam_retina:
                sensor_x.append((S_cone_x))
                sensor_y.append(S_cone_y)
                sensor_type.append('S_cone')
                S_cone_pos.append([S_cone_x, S_cone_y])
                s = s+1

    # M-cones:
    m = 0
    print('Computing M-cone positions')
    with tqdm(total=num_M_cones, initial=m) as pbar:
        while m < num_M_cones:
            pbar.update(m - pbar.n)
            M_cone_x = np.random.uniform(-0.5*diam_retina,0.5*diam_retina)
            M_cone_y = np.random.uniform(-0.5*diam_retina,0.5*diam_retina)
            M_cone_p = np.random.uniform(0,1.1*max(rodfit(fitspace)))

            if M_cone_p <= rodfit(np.sqrt(M_cone_x**2+M_cone_y**2)) \
                    and np.sqrt(M_cone_x**2+M_cone_y**2) <= 0.5*diam_retina:
                sensor_x.append(M_cone_x)
                sensor_y.append(M_cone_y)
                sensor_type.append('M_cone')
                M_cone_pos.append([M_cone_x, M_cone_y])
                m = m+1
    return (sensor_x, sensor_y, sensor_type, rod_pos, S_cone_pos, M_cone_pos)


# function that returns a list of the RGC positions:
def RGCs(num_neurons):
    RGC_pos = []
    r = 0
    print('Computing RGC positions')
    with tqdm(total=num_neurons, initial=r) as pbar:
        while r < num_neurons:
            pbar.update(r - pbar.n)
            RGC_x = np.random.uniform(-0.5 * diam_retina, 0.5 * diam_retina)
            RGC_y = np.random.uniform(-0.5 * diam_retina, 0.5 * diam_retina)
            RGC_p = np.random.uniform(0, 1.1 * max(rodfit(fitspace)))
            RGC = [RGC_x, RGC_y]

            if RGC_p <= rodfit(np.sqrt(RGC_x ** 2 + RGC_y ** 2))*0.5*(RGC_over(RGC_x)) \
                    and np.sqrt(RGC_x ** 2 + RGC_y ** 2) <= 0.5*diam_retina:
                RGC_pos.append(RGC)
                r = r+1
    return RGC_pos


# function that returns a list of the horizontal cell positions:
def horizontal(num_horizontal_cells):
    horizontal_pos =[]
    h = 0
    print('Computing horizontal cell positions')
    with tqdm(total=num_horizontal_cells, initial=h) as pbar:
        while h < num_horizontal_cells:
            pbar.update(h - pbar.n)
            horizontal_x = np.random.uniform(-0.5 * diam_retina, 0.5 * diam_retina)
            horizontal_y = np.random.uniform(-0.5 * diam_retina, 0.5 * diam_retina)
            horizontal_p = np.random.uniform(0, 1.1 * max(rodfit(fitspace)))
            horizontal = [horizontal_x, horizontal_y]

            if horizontal_p <= rodfit(np.sqrt(horizontal_x ** 2 + horizontal_y ** 2)) \
                    and np.sqrt(horizontal_x ** 2 + horizontal_y ** 2) <= 0.5 * diam_retina:
                horizontal_pos.append(horizontal)
                h = h + 1
    return horizontal_pos



# Comment this out if you run it via Setup: ----------------------------------------------------------------------------
#'''
# Create Lists: --------------------------------------------------------------------------------------------------------
sensor_x, sensor_y, sensor_type, rod_pos, S_cone_pos, M_cone_pos = sensors(P.num_rods, P.num_S_cones, P.num_M_cones)

rod_pos = []
S_cone_pos = []
M_cone_pos = []
for i in np.arange(P.num_sensors):
    element = [sensor_x[i], sensor_y[i]]
    if i < P.num_rods:
        rod_pos.append(element)

    elif P.num_rods <= i < P.num_rods + P.num_S_cones:
        S_cone_pos.append(element)

    else:
        M_cone_pos.append(element)

RGC_pos = RGCs(P.num_RGCs)


# set up layer----------------------------------------------------------------------------------------------------------
retina_rods = tp.CreateLayer({'positions':rod_pos,
                              'elements':'iaf_psc_alpha',
                              'extent': [diam_retina, diam_retina]})

retina_S_cones = tp.CreateLayer({'positions': S_cone_pos,
                               'elements':'iaf_psc_alpha',
                               'extent': [diam_retina, diam_retina]})

retina_M_cones = tp.CreateLayer({'positions': M_cone_pos,
                               'elements':'iaf_psc_alpha',
                               'extent': [diam_retina, diam_retina]})

retina_RGCs = tp.CreateLayer({'positions': RGC_pos,
                               'elements':'iaf_psc_alpha',
                               'extent': [diam_retina, diam_retina]})


# Plot layer -----------------------------------------------------------------------------------------------------------
plt.close('all')
retinalplot = plt.figure(num='retina',figsize = (8, 8))
plt.title('Retina')
tp.PlotLayer(retina_rods, fig=retinalplot, nodecolor = 'k', nodesize = 1)
tp.PlotLayer(retina_S_cones, fig=retinalplot, nodecolor = '#b412ed', nodesize = 10)
tp.PlotLayer(retina_M_cones, fig=retinalplot, nodecolor = '#019309', nodesize = 10)
retinalplot.savefig('Retina/figures/retina.png', dpi = 100)
plt.show()

plt.figure('PDF')
plt.plot(fitspace, rodfit(fitspace))
plt.show()
#'''


#'''
# 2D histogram----------------------------------------------------------------------------------------------------------
# Rods:
rod_pos_x = [row[0] for row in rod_pos]
rod_pos_y = [row[1] for row in rod_pos]

rodhisto = plt.figure(figsize=(4,4))
plt.title('Rod density, n = '+str(P.num_rods))
plt.hist2d(rod_pos_x, rod_pos_y,bins=(20, 20),range=[[-0.5*diam_retina, 0.5*diam_retina],
                                                       [-0.5*diam_retina, 0.5*diam_retina]])
plt.colorbar()
plt.axes().set_aspect('equal')
rodhisto.savefig('Retina/figures/rod_histo.png')
plt.show()

# S-cones:
S_cone_pos_x=[row[0] for row in S_cone_pos]
S_cone_pos_y=[row[1] for row in S_cone_pos]

S_conehisto=plt.figure(figsize=(4,4))
plt.title('S-cone density, n = '+str(P.num_S_cones))
plt.hist2d(S_cone_pos_x, S_cone_pos_y, bins=(20, 20), range=[[-0.5*diam_retina, 0.5*diam_retina],
                                                           [-0.5*diam_retina, 0.5*diam_retina]])
plt.colorbar()
plt.axes().set_aspect('equal')
S_conehisto.savefig('Retina/figures/S_cone_histo.png')
plt.show()


# M-cones:
M_cone_pos_x=[row[0] for row in M_cone_pos]
M_cone_pos_y=[row[1] for row in M_cone_pos]

M_conehisto=plt.figure(figsize=(4,4))
plt.title('M-cone density, n = '+str(P.num_M_cones))
plt.hist2d(M_cone_pos_x, M_cone_pos_y, bins=(20, 20), range=[[-0.5*diam_retina, 0.5*diam_retina],
                                                           [-0.5*diam_retina, 0.5*diam_retina]])
plt.colorbar()
plt.axes().set_aspect('equal')
M_conehisto.savefig('Retina/figures/M_cone_histo.png')
plt.show()

# RGCs:
RGC_pos_x=[row[0] for row in RGC_pos]
RGC_pos_y=[row[1] for row in RGC_pos]

RGChisto=plt.figure(figsize=(4,4))
plt.title('RGC density, n = ' + str(P.num_RGCs))
plt.hist2d(RGC_pos_x, RGC_pos_y, bins=(20, 20), range=[[-0.5*diam_retina, 0.5*diam_retina],
                                                           [-0.5*diam_retina, 0.5*diam_retina]])
plt.colorbar()
plt.axes().set_aspect('equal')
RGChisto.savefig('Retina/figures/RGC_histo.png')
plt.show()
#'''


stoptime=time.time()
print('')
print("Simulation took " +str(stoptime-starttime)+" seconds")

