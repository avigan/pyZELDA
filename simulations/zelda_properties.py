from __future__ import absolute_import, division, print_function
# -*- coding: utf-8 -*-
'''
simulations of ZELDA properties

arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

'''
Initialization
'''

# ----------------------------------------
# package import
# ----------------------------------------
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams

import copy

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.mft as mft
import pyzelda.utils.imutils as imutils
import pyzelda.utils.aperture as aperture
import pyzelda.utils.circle_fit as circle_fit

# definition of the color map for the colored lines
cm0 = plt.get_cmap('rainbow')

from astropy.io import fits


# path
# path = '/Users/mndiaye/Dropbox/python/zelda/pyZELDA/'
path = '/Users/avigan/Work/GitHub/pyZELDA/'
if path not in sys.path:
    sys.path.append(path)

# ----------------------------------------
# save plot in pdf file
# ----------------------------------------
do_plot = 0

# ----------------------------------------
# simulation parameters
# ----------------------------------------
# sampling parameters for simulation 
array_dim      = 500
R_pupil_pixels = 384//2
radius_vec     = np.arange(array_dim)/(2*R_pupil_pixels)-array_dim/(2*2*R_pupil_pixels)

# ----------------------------------------
# Mask characteristics for Classical one (CLA), VLT/SPHERE (SPH), and SPEED (SPD) 
# ----------------------------------------

# wavelengths
wave_CLA = 0.633e-6
wave_SPH = 1.642e-6
wave_SPD = 0.633e-6

# number of mask cases and name
sensor_list  = ['CLASSIC', 'SPHERE-IRDIS', 'SPHERE-IRDIS']
tel_pup_list = [   False,           False,           True]
wave_list    = [wave_CLA,        wave_SPH,       wave_SPH]

max_len = len(max(sensor_list))

nMask = len(sensor_list)
z_arr = []
for i in range(nMask):
    z_arr.append(zelda.Sensor(sensor_list[i], pupil_telescope=tel_pup_list[i]))

# relative size and phase shift at lam of the mask in lam0/D and radians
rel_size_vec = []
theta_vec    = []
for i in range(nMask):
    z = z_arr[i]
    rel_size_vec.append(z.mask_relative_size(wave_list[i]))
    theta_vec.append(z.mask_phase_shift(wave_list[i]))

for i in range(nMask):
    print('{0:<{1}} mask size: {2:0.3f} lam/D'.format(sensor_list[i], max_len, rel_size_vec[i]))
print(' ')

for i in range(nMask):
    print('{0:<{1}} mask size: {2:0.3f} lam/D'.format(sensor_list[i], max_len, theta_vec[i]))
print(' ')

# phase error
npts     = 1200
phi_vec  = (1.2*2*np.pi/npts)*(np.arange(npts)-npts//2)
wave_vec = phi_vec[:]/(2.*np.pi)

# nm wavefront error array
nm_vec       = []
for i in range(nMask):
    nm_vec.append(wave_list[i]*1e9*phi_vec/(2.*np.pi))


# pupil plane amplitude
pup = np.real(aperture.disc(array_dim, R_pupil_pixels, cpix=True, strict=False))

# radial profile of the pupil plane amplitude
P_vec = pup[array_dim//2]

# Reference wave
reference_wave_vec = []
expi_vec           = []
b_arr_vec          = []
b_vec              = []
avg_b_vec          = []
IC0_vec            = []
IC1_vec            = []
IC2_vec            = []
for i in range(nMask):
    npad = (array_dim - z.pupil_diameter) // 2
    pup = np.pad(z.pupil, npad, mode='constant')

    z = z_arr[i]
    reference_wave, expi = ztools.create_reference_wave_beyond_pupil(z.mask_diameter, 
                                                                     z.mask_depth,
                                                                     z.mask_substrate,
                                                                     z.mask_Fratio,
                                                                     2*R_pupil_pixels,
                                                                     pup,
                                                                     wave_list[i])
    reference_wave_vec.append(reference_wave)
    expi_vec.append(expi)
    tmp = np.real(reference_wave)
    b_arr_vec.append(tmp)
    b_vec.append(tmp[array_dim//2])                                                                
    avg_b_vec.append(np.mean(tmp[pup != 0.]))
    IC0, IC1, IC2 = ztools.zelda_analytical_intensity(phi_vec, b=avg_b_vec[i], theta=theta_vec[i])
    IC0_vec.append(IC0)
    IC1_vec.append(IC2) 
    IC2_vec.append(IC2)  

    
for i in range(nMask):
    print('{0:<{1}} mask: {2:0.3f} mean amplitude'.format(sensor_list[i], max_len, avg_b_vec[i]))
print(' ')

# Mask capture range
for i in range(nMask):
    print('{0:<{1}} mask (min range): {2:0.3f} lam'.format(sensor_list[i], max_len, wave_vec[IC0_vec[i] == min(IC0_vec[i])][0]))
    print('{0:<{1}} mask (max range): {2:0.3f} lam'.format(sensor_list[i], max_len, wave_vec[IC0_vec[i] == max(IC0_vec[i])][0]))
    print(' ')

# Mask sensitivity    
for i in range(nMask):
    print('{0:<{1}} mask sensitivity: {2:0.3f}'.format(sensor_list[i], max_len, np.amax(IC0_vec[i])-np.amin(IC0_vec[i])))
print(' ')  

# ----------------------------------------
# plot for the mask diffracted wave
# ----------------------------------------

# setup plot
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['font.size']       = 17

# show plot
# plt.show()

# position of the legend in the plot
x1_leg = 0.06
y1_leg = 0.72
 
# plot of the mask diffracted amplitude for three different cases
fig1 = plt.figure(1, figsize=(15, 9))

plt.clf()
ax = fig1.add_subplot(111)

plt.xlabel('r')
plt.ylabel('Normalized ampliutde')
plt.xticks(np.arange(-0.50, 0.51, 0.10))
plt.yticks(np.arange(0.0, 1.1, 0.1))
ax.set_color_cycle([cm0(1.*i/(nMask-1)) for i in range(nMask)])
for i in range(nMask):
    ax.plot(radius_vec, b_vec[i], label='{0} (tel_pup={1:d})'.format(sensor_list[i], z_arr[i].pupil_telescope))
ax.plot(radius_vec, P_vec, 'k--')
ax.set_xlim([-0.55, 0.55])
ax.set_ylim([-0.05, 1.05])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=(x1_leg, y1_leg), labelspacing=0.1, title='Diffracted amplitude b for:')
plt.title('Radial profile of the mask diffracted amplitude')
plt.tight_layout()

# ----------------------------------------
# plot for zelda intensity signal
# ----------------------------------------
# position of the legend in the plot
x2_leg = 0.40
y2_leg = 0.82

# plot of the mask diffracted amplitude for three different cases
fig2 = plt.figure(2, figsize=(15, 9))
plt.clf()
ax = fig2.add_subplot(111)

plt.xlabel(r'Wavefront error in $\lambda$ unit')
plt.ylabel('Normalized exit pupil intensity')
plt.xticks(np.arange(-0.50, 0.51, 0.10))
plt.yticks(np.arange(0.0, 4.1, 1.0))
ax.set_color_cycle([cm0(1.*i/(nMask-1)) for i in range(nMask)])
for i in range(nMask):
    ax.plot(wave_vec, IC0_vec[i], label='{0} (tel_pup={1:d})'.format(sensor_list[i], z_arr[i].pupil_telescope))
ax.set_xlim([-0.25, 0.45])
ax.set_ylim([-0.25, 3.75])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=(x2_leg, y2_leg), labelspacing=0.1)
for i in range(nMask):
    plt.axvline(x=0.0, color='k', linestyle='--')
    plt.axhline(y=np.amax(IC0_vec[i]), color=cm0(1.*i/(nMask-1)), linestyle='--')
    plt.axhline(y=np.amin(IC0_vec[i]), color=cm0(1.*i/(nMask-1)), linestyle='--')
#    plt.axvline(x=wave_vec[IC0_vec[i] == max(IC0_vec[i])][0], color=cm0(1.*i/(nMask-1)), linestyle='--')
    plt.axvline(x=wave_vec[IC0_vec[i] == min(IC0_vec[i])][0], color=cm0(1.*i/(nMask-1)), linestyle='--')
#    ax.text(-0.2, 3.5-0.3*i, r'{0:0.3f}$\lambda$'.format(wave_vec[IC0_vec[i] == min(IC0_vec[i])][0]), fontsize=15, horizontalalignment="center", color=cm0(1.*i/(nMask-1)))
#    ax.text(0.41, 3.5-0.3*i, r'{0:0.3f}$\lambda$'.format(wave_vec[IC0_vec[i] == max(IC0_vec[i])][0]), fontsize=15, horizontalalignment="center", color=cm0(1.*i/(nMask-1)))
#for i in range(3):
#    ax.add_patch(patches.Rectangle((wave_vec[IC0_vec[i] == min(IC0_vec[i])][0], np.amin(IC0_vec[i])), \
#    wave_vec[IC0_vec[i] == max(IC0_vec[i])][0]-wave_vec[IC0_vec[i] == min(IC0_vec[i])][0], \
#    np.amax(IC0_vec[i])-np.amin(IC0_vec[i]),\
#    alpha=(i+1)*0.1, facecolor=cm0(1.*i/(nMask-1))))
plt.title('Normalized exit pupil intensity')
plt.tight_layout()
