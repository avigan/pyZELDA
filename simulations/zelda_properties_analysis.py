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
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

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
if sys.version_info < (3, 0):
    import ConfigParser
else:
    import configparser as ConfigParser

# path
path = '/Users/mndiaye/Dropbox/python/zelda/pyZELDA/'
# path = '/Users/avigan/Work/GitHub/pyZELDA/'
if path not in sys.path:
    sys.path.append(path)

# ----------------------------------------
# save plot in pdf file
# ----------------------------------------
do_plot = 0

# ----------------------------------------
# units
# ----------------------------------------
um = 1e-6
nm = 1e-9

# ----------------------------------------
# Mask characteristics for Classical one (CLA), VLT/SPHERE (SPH), and SPEED (SPD) 
# ----------------------------------------
# number of mask cases and name
nMask     = 3
labelTab = ['Classical mask', 'SPHERE mask', 'SPEED mask']

z_CLA = zelda.Sensor('CLASSIC')
z_SPH = zelda.Sensor('SPHERE-IRDIS')
z_SPD = zelda.Sensor('SPEED')

# wavelength
wave_SPH = 1.642*um
wave_SPD = 633*nm
wave_CLA = copy.deepcopy(wave_SPD) 

# relative size of the mask in lam0/D
a_CLA = z_CLA.mask_relative_size(wave_CLA)
a_SPH = z_SPH.mask_relative_size(wave_SPH)
a_SPD = z_SPD.mask_relative_size(wave_SPD)

print('classical mask size: {0:0.3f} lam/D'.format(a_CLA))
print('SPHERE    mask size: {0:0.3f} lam/D'.format(a_SPH))
print('SPEED     mask size: {0:0.3f} lam/D'.format(a_SPD))
print(' ')

# phase shift introduced by the prototype at lam0
theta_CLA = z_CLA.mask_phase_shift(wave_CLA)
theta_SPH = z_SPH.mask_phase_shift(wave_SPH)
theta_SPD = z_SPD.mask_phase_shift(wave_SPD)

print('classical mask: {0:0.3f} pi phase shift'.format(theta_CLA/np.pi))
print('SPHERE    mask: {0:0.3f} pi phase shift'.format(theta_SPH/np.pi))
print('SPEED     mask: {0:0.3f} pi phase shift'.format(theta_SPD/np.pi))
print(' ')

# sampling parameters for simulation 
pupil_diameter = 500
R_pupil_pixels = 150


radius_vec     = np.arange(pupil_diameter)/(2*R_pupil_pixels)-pupil_diameter/(2*2*R_pupil_pixels)

# pupil plane amplitude
pup = np.real(aperture.disc(pupil_diameter, R_pupil_pixels, cpix=True, strict=True))

# Reference wave
reference_wave_CLA, expi_CLA = ztools.create_reference_wave_beyond_pupil(z_CLA.mask_diameter, 
                                                                z_CLA.mask_depth,
                                                                z_CLA.mask_substrate,
                                                                pupil_diameter, 
                                                                R_pupil_pixels, z_CLA.Fratio, wave_CLA)                                                                
                                                               
reference_wave_SPH, expi_SPH = ztools.create_reference_wave_beyond_pupil(z_SPH.mask_diameter, 
                                                                z_SPH.mask_depth,
                                                                z_SPH.mask_substrate,
                                                                pupil_diameter, 
                                                                R_pupil_pixels, z_SPH.Fratio, wave_SPH)

reference_wave_SPD, expi_SPD = ztools.create_reference_wave_beyond_pupil(z_SPD.mask_diameter, 
                                                                z_SPD.mask_depth,
                                                                z_SPD.mask_substrate,
                                                                pupil_diameter, 
                                                                R_pupil_pixels, z_SPD.Fratio, wave_SPD)

# mask diffracted wave amplitude
b_CLA = np.real(reference_wave_CLA)
b_SPH = np.real(reference_wave_SPH)
b_SPD = np.real(reference_wave_SPD)

# radial profile of the pupil plane amplitude
P_vec    = pup[pupil_diameter//2]

# radial profile of the mask diffracted wave amplitude
b_vec = np.zeros((nMask, pupil_diameter))
b_vec[0] = b_CLA[pupil_diameter//2]
b_vec[1] = b_SPH[pupil_diameter//2]
b_vec[2] = b_SPD[pupil_diameter//2]

# Averaged intensity of a given pixel as a function of b
avg_b_CLA = np.mean(b_CLA[pup != 0.])
avg_b_SPH = np.mean(b_SPH[pup != 0.])
avg_b_SPD = np.mean(b_SPD[pup != 0.])

print('classical mask: {0:0.3f} mean amplitude'.format(avg_b_CLA))
print('SPHERE    mask: {0:0.3f} mean amplitude'.format(avg_b_SPH))
print('SPEED     mask: {0:0.3f} mean amplitude'.format(avg_b_SPD))
print(' ')

# phase error
npts     = 1200
phi_vec  = (1.2*2*np.pi/npts)*(np.arange(npts)-npts//2)
wave_vec = phi_vec[:]/(2.*np.pi)

# nm wavefront error array
nm_vec_CLA   = wave_CLA*1e9*phi_vec/(2.*np.pi)
nm_vec_SPH   = wave_SPH*1e9*phi_vec/(2.*np.pi)
nm_vec_SPD   = wave_SPD*1e9*phi_vec/(2.*np.pi)

# Intensity versus pupil plane pixel
## Sinudoid expression
IC0_CLA, IC1_CLA, IC2_CLA = ztools.zelda_analytical_intensity(phi_vec, b=avg_b_CLA, theta=theta_CLA)
IC0_SPH, IC1_SPH, IC2_SPH = ztools.zelda_analytical_intensity(phi_vec, b=avg_b_SPH, theta=theta_SPH)
IC0_SPD, IC1_SPD, IC2_SPD = ztools.zelda_analytical_intensity(phi_vec, b=avg_b_SPD, theta=theta_SPD)

IC0_vec    = np.zeros((nMask, npts))
IC0_vec[0] = IC0_CLA
IC0_vec[1] = IC0_SPH
IC0_vec[2] = IC0_SPD

# Mask capture range
for i in range(3):
    print('mask (min range): {0:0.3f} lam'.format(wave_vec[IC0_vec[i] == min(IC0_vec[i])][0]))
    print('mask (max range): {0:0.3f} lam'.format(wave_vec[IC0_vec[i] == max(IC0_vec[i])][0]))
    print(' ')

# Mask sensitivity    
for i in range(3):
    print('mask sensitivity: {0:0.3f}'.format(np.amax(IC0_vec[i])-np.amin(IC0_vec[i])))
print(' ')  

# ----------------------------------------
# plot for the mask diffracted wave
# ----------------------------------------
# show plot
plt.show()

# position of the legend in the plot
x1_leg = 0.06
y1_leg = 0.72
 
# plot of the mask diffracted amplitude for three different cases
fig1 = plt.figure(1, figsize=(8, 4.5))

plt.clf()
plt.xlabel('r')
plt.ylabel('Normalized ampliutde')
plt.xticks(np.arange(-0.50, 0.51, 0.10))
plt.yticks(np.arange(0.0, 1.1, 0.1))
ax = fig1.add_subplot(111)
ax.set_color_cycle([cm0(1.*i/(nMask-1)) for i in range(nMask)])
for i in range(nMask):
    ax.plot(radius_vec, b_vec[i], label = labelTab[i])
ax.plot(radius_vec, P_vec, 'k--')
ax.set_xlim([-0.55, 0.55])
ax.set_ylim([-0.05, 1.05])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = (x1_leg, y1_leg), labelspacing=0.1, title = 'Diffracted amplitude b for:')
plt.title('Radial profile of the mask diffracted amplitude')
plt.tight_layout()
if do_plot:
    fig1.savefig('/Users/mndiaye/Dropbox/OCA/projects/Speed/ZELDA-SPEED/plots/\
mask_diffracted_wave.pdf')

# ----------------------------------------
# plot for zelda intensity signal
# ----------------------------------------
# position of the legend in the plot
x2_leg = 0.40
y2_leg = 0.82

# plot of the mask diffracted amplitude for three different cases
fig2 = plt.figure(2, figsize=(8, 4.5))
plt.clf()
plt.xlabel(r'Wavefront error in $\lambda$ unit')
plt.ylabel('Normalized exit pupil intensity')
plt.xticks(np.arange(-0.50, 0.51, 0.10))
plt.yticks(np.arange(0.0, 4.1, 1.0))
ax = fig2.add_subplot(111)
ax.set_color_cycle([cm0(1.*i/(nMask-1)) for i in range(nMask)])
for i in range(nMask):
    ax.plot(wave_vec, IC0_vec[i], label = labelTab[i])
ax.set_xlim([-0.25, 0.45])
ax.set_ylim([-0.25, 3.75])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = (x2_leg, y2_leg), labelspacing=0.1)
for i in range(3):
    plt.axvline(x=0.0, color='k', linestyle = '--')
    plt.axhline(y=np.amax(IC0_vec[i]), color=cm0(1.*i/(nMask-1)), linestyle = '--')
    plt.axhline(y=np.amin(IC0_vec[i]), color=cm0(1.*i/(nMask-1)), linestyle = '--')
#    plt.axvline(x=wave_vec[IC0_vec[i] == max(IC0_vec[i])][0], color=cm0(1.*i/(nMask-1)), linestyle = '--')
    plt.axvline(x=wave_vec[IC0_vec[i] == min(IC0_vec[i])][0], color=cm0(1.*i/(nMask-1)), linestyle = '--')
#    ax.text(-0.2, 3.5-0.3*i, r'{0:0.3f}$\lambda$'.format(wave_vec[IC0_vec[i] == min(IC0_vec[i])][0]), fontsize=15, horizontalalignment="center", color=cm0(1.*i/(nMask-1)))
#    ax.text(0.41, 3.5-0.3*i, r'{0:0.3f}$\lambda$'.format(wave_vec[IC0_vec[i] == max(IC0_vec[i])][0]), fontsize=15, horizontalalignment="center", color=cm0(1.*i/(nMask-1)))
#for i in range(3):
#    ax.add_patch(patches.Rectangle((wave_vec[IC0_vec[i] == min(IC0_vec[i])][0], np.amin(IC0_vec[i])), \
#    wave_vec[IC0_vec[i] == max(IC0_vec[i])][0]-wave_vec[IC0_vec[i] == min(IC0_vec[i])][0], \
#    np.amax(IC0_vec[i])-np.amin(IC0_vec[i]),\
#    alpha=(i+1)*0.1, facecolor=cm0(1.*i/(nMask-1))))
plt.title('Normalized exit pupil intensity')
plt.tight_layout()
if do_plot:
    fig2.savefig('/Users/mndiaye/Dropbox/OCA/projects/Speed/ZELDA-SPEED/plots/\
normalized_intensity_exit_pupil.pdf')

  