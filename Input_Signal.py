import Parameters as P

from PIL import Image, ImageDraw
import numpy as np
import importlib
import sys
import os
import time
import matplotlib.pyplot as plt

importlib.reload(P)
starttime = time.time()
# ----------------------------------------------------------------------------------------------------------------------
# In this file the input signal for the Network is created
# ----------------------------------------------------------------------------------------------------------------------

# Image dimensions: ----------------------------------------------------------------------------------------------------
x_axis_length = P.resolution
y_axis_length = P.resolution


x_coordinate = np.arange(0, x_axis_length)
y_coordinate = np.arange(0, y_axis_length)


# Signal Parameters: ---------------------------------------------------------------------------------------------------
if P.Signal_source == 'LED':
    Waves = dict([
        ('Color1', (61, 0, 61)),
        ('Color2', (0, 46, 255)),
        ('Color3', (163, 255, 0)),
        ('White', (254, 228, 255))
    ])

if P.Signal_source == 'TFT':
    Waves=dict([
        ('Color1', (0, 0, 255)),
        ('Color2', (0, 255, 0)),
        ('Color3', (255, 0, 0)),
        ('White', (255, 255, 255))
    ])


# Painting the Image: --------------------------------------------------------------------------------------------------
image_dict=dict([
    ['vgrid', 'vertical two colored grid with colors A & B and grid constant g '],
    ['vgrid_sin', 'vertical two colored grid with colors A & B and grid constant g and sinusoidal phases']
])

image_type = 'vgrid'

# Colors used, these colors must not be the same, for uniform colors, set g = 1000:
C1 = 'Color3'
C2 = 'Color2'

# relative amplitudes:
A1 = 1
A2 = 0

canvas = Image.new('RGB', (x_axis_length, y_axis_length), (0, 0, 0))
canvas_anti = Image.new('RGB', (x_axis_length, y_axis_length), (0, 0, 0))

def Input_Signal(g):
    Int_dict = dict([
        ('Color1', np.zeros((x_axis_length, y_axis_length))),
        ('Color2', np.zeros((x_axis_length, y_axis_length))),
        ('Color3', np.zeros((x_axis_length, y_axis_length))),
        ('White', np.zeros((x_axis_length, y_axis_length)))
    ])

    Int_dict_anti = dict([
        ('Color1', np.zeros((x_axis_length, y_axis_length))),
        ('Color2', np.zeros((x_axis_length, y_axis_length))),
        ('Color3', np.zeros((x_axis_length, y_axis_length))),
        ('White', np.zeros((x_axis_length, y_axis_length)))
    ])

    global canvas, canvas_anti
    draw = ImageDraw.Draw(canvas)
    draw_anti = ImageDraw.Draw(canvas_anti)
    if image_type == 'vgrid':
        pixels = []
        p = 0
        entry = 1
        while p < x_axis_length:
            if x_axis_length-p < g:
                appendix = [entry]*(x_axis_length-p)
                p = x_axis_length
            else:
                appendix = [entry]*g
                p = p+g

            pixels.extend(appendix)

            if entry == 1:
                entry = 2
            elif entry == 2:
                entry = 1


        for i in x_coordinate:
            line = ((i, 0), (i, y_axis_length))

            if pixels[i] == 1:
                factor1 = A1
                factor2 = 0
                factor1_anti = 0
                factor2_anti = A2
            if pixels[i] == 2:
                factor1 = 0
                factor2 = A2
                factor1_anti = A1
                factor2_anti = 0


            Int_dict[C1][i] = factor1
            Int_dict[C2][i] = factor2
            Int_dict_anti[C1][i] = factor1_anti
            Int_dict_anti[C2][i] = factor2_anti

            # Comment out from here if you don't want to create an Image:
            '''
            # Phase 1:
            c1sat1 = factor1 * Waves[C1][0]
            c1sat2 = factor1 * Waves[C1][1]
            c1sat3 = factor1 * Waves[C1][2]

            c1sat1_anti = factor1_anti * Waves[C1][0]
            c1sat2_anti = factor1_anti * Waves[C1][1]
            c1sat3_anti = factor1_anti * Waves[C1][2]

            # Phase 2:
            c2sat1 = factor2 * Waves[C2][0]
            c2sat2 = factor2 * Waves[C2][1]
            c2sat3 = factor2 * Waves[C2][2]

            c2sat1_anti = factor2_anti * Waves[C2][0]
            c2sat2_anti = factor2_anti * Waves[C2][1]
            c2sat3_anti = factor2_anti * Waves[C2][2]

            color = (int(c1sat1 + c2sat1), int(c1sat2 + c2sat2), int(c1sat3 + c2sat3))
            color_anti = (int(c1sat1_anti + c2sat1_anti), int(c1sat2_anti + c2sat2_anti), int(c1sat3_anti + c2sat3_anti))

            draw.line(line, fill=color, width=1)
            draw_anti.line(line, fill=color_anti, width=1)
            #'''

        del draw, draw_anti

    if image_type == 'vgrid_sin':
        for i in x_coordinate:
            line = ((i, 0), (i, y_axis_length))
            # Phase 1
            factor1 = A1 * (np.sin(i * np.pi / (2*g)))**2
            factor1_anti = A1 * (np.cos(i * np.pi / (2*g)))**2

            # Phase 2
            factor2 = A2 * (np.cos(i * np.pi / (2*g)))**2
            factor2_anti = A2 * (np.sin(i * np.pi / (2*g)))**2

            Int_dict[C1][i] = factor1
            Int_dict[C2][i] = factor2
            Int_dict_anti[C1][i] = factor1_anti
            Int_dict_anti[C2][i] = factor2_anti

            # Comment out from here if you don't want to create an Image:
            '''
            # Phase 1:
            c1sat1 = factor1 * Waves[C1][0]
            c1sat2 = factor1 * Waves[C1][1]
            c1sat3 = factor1 * Waves[C1][2]

            c1sat1_anti = factor1_anti * Waves[C1][0]
            c1sat2_anti = factor1_anti * Waves[C1][1]
            c1sat3_anti = factor1_anti * Waves[C1][2]

            # Phase 2:
            c2sat1 = factor2 * Waves[C2][0]
            c2sat2 = factor2 * Waves[C2][1]
            c2sat3 = factor2 * Waves[C2][2]

            c2sat1_anti = factor2_anti * Waves[C2][0]
            c2sat2_anti = factor2_anti * Waves[C2][1]
            c2sat3_anti = factor2_anti * Waves[C2][2]
            
            color = (int(c1sat1 + c2sat1), int(c1sat2 + c2sat2), int(c1sat3 + c2sat3))
            color_anti = (int(c1sat1_anti + c2sat1_anti), int(c1sat2_anti + c2sat2_anti), int(c1sat3_anti + c2sat3_anti))

            draw.line(line, fill=color, width=1)
            draw_anti.line(line, fill=color_anti, width=1)
            #'''

        del draw, draw_anti
    print('Signal creation took: '+str(time.time()-starttime))
    return Int_dict, Int_dict_anti

# ----------------------------------------------------------------------------------------------------------------------
# Run Code from here:
# ----------------------------------------------------------------------------------------------------------------------
'''

# Grid constant, distance between peaks. For a uniform image in color C1 set g = 1000:
g = 200


# Compute Signal: ------------------------------------------------------------------------------------------------------
Int_dict, Int_dict_anti = Input_Signal(g)


# Save Signal: ---------------------------------------------------------------------------------------------------------
dummytime = time.time()

np.savez_compressed('./Input_Signal/Input',
                    # saves as .npz -file.
                    Signal_Source=P.Signal_source,
                    ColorA=C1,
                    ColorB=C2,
                    Int_dict=Int_dict,
                    Int_dict_anti=Int_dict_anti,
                    ImageType=image_dict[image_type],
                    SignalSource=P.Signal_source
                    )


# canvas.show()
# Save the Image: ------------------------------------------------------------------------------------------------------
if not os.path.exists('./Input_Signal/'):
    os.makedirs('./Input_Signal/')

canvas.save('./Input_Signal/canvas.png')
canvas_anti.save('./Input_Signal/canvas_anti.png')

# Create a GIF:
flickering = [canvas, canvas_anti]
flickering[0].save('./Input_Signal/flickering.gif',
                   format='GIF', append_images=flickering, save_all=True, duration=500, loop=0)

print('Saving image and Int_dict took: '+str(time.time()-dummytime))
#'''
