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

# wavelength
wave_SPH = 1.642*um
wave_SPD = 633*nm
wave_CLA = copy.deepcopy(wave_SPD) 

# Focal ratio
Fratio_SPH = 40
Fratio_SPD = 65
Fratio_CLA = copy.deepcopy(Fratio_SPD)

# substrate
mask_substrate            = 'fused_silica'
mask_refractive_index_CLA = ztools.refractive_index(wave_CLA, mask_substrate)
mask_refractive_index_SPH = ztools.refractive_index(wave_SPH, mask_substrate)
mask_refractive_index_SPD = ztools.refractive_index(wave_SPD, mask_substrate)

# mask diameter
mask_diameter_CLA = 1.062*wave_CLA*Fratio_CLA
mask_diameter_SPH = 70.7 *um
mask_diameter_SPD = 36*um

# mask depth
mask_depth_CLA = wave_CLA/(4.*(mask_refractive_index_CLA-1))
mask_depth_SPH = 814.6*nm
mask_depth_SPD = 275 *nm

# relative size of the mask in lam0/D
a_CLA = mask_diameter_CLA/(wave_CLA*Fratio_CLA)
a_SPH = mask_diameter_SPH/(wave_SPH*Fratio_SPH)
a_SPD = mask_diameter_SPD/(wave_SPD*Fratio_SPD)

print('classical mask: {0:0.3f} lam/D'.format(a_CLA))
print('SPHERE mask: {0:0.3f} lam/D'.format(a_SPH))
print('SPEED mask: {0:0.3f} lam/D'.format(a_SPD))

# phase shift introduced by the prototype at lam0
theta_CLA = 2.*np.pi*(mask_refractive_index_CLA-1)*mask_depth_CLA/wave_CLA
theta_SPH = 2.*np.pi*(mask_refractive_index_SPH-1)*mask_depth_SPH/wave_SPH
theta_SPD = 2.*np.pi*(mask_refractive_index_SPD-1)*mask_depth_SPD/wave_SPD

print('classical mask: {0:0.3f} pi phase shift'.format(theta_CLA/np.pi))
print('SPHERE mask: {0:0.3f} pi phase shift'.format(theta_SPH/np.pi))
print('SPEED mask: {0:0.3f} pi phase shift'.format(theta_SPD/np.pi))

# sampling parameters for simulation 
pupil_diameter = 500
R_pupil_pixels = 150


radius_tab     = np.arange(pupil_diameter)/(2*R_pupil_pixels)-pupil_diameter/(2*2*R_pupil_pixels)

ampl_PA_noaberr = aperture.disc(pupil_diameter, R_pupil_pixels, cpix=True, strict=True)                                                           

# Reference wave
reference_wave_CLA, expi_CLA = ztools.create_reference_wave_beyond_pupil(mask_diameter_CLA, 
                                                                mask_depth_CLA,
                                                                mask_substrate,
                                                                pupil_diameter, 
                                                                R_pupil_pixels, Fratio_CLA, wave_CLA)                                                                
                                                               
reference_wave_SPH, expi_SPH = ztools.create_reference_wave_beyond_pupil(mask_diameter_SPH, 
                                                                mask_depth_SPH,
                                                                mask_substrate,
                                                                pupil_diameter, 
                                                                R_pupil_pixels, Fratio_SPH, wave_SPH)

reference_wave_SPD, expi_SPD = ztools.create_reference_wave_beyond_pupil(mask_diameter_SPD, 
                                                                mask_depth_SPD,
                                                                mask_substrate,
                                                                pupil_diameter, 
                                                                R_pupil_pixels, Fratio_SPD, wave_SPD)

# pupil plane amplitude
P_term     = np.real(ampl_PA_noaberr)

# mask diffracted wave amplitude
b_term_CLA = np.real(reference_wave_CLA)
b_term_SPH = np.real(reference_wave_SPH)
b_term_SPD = np.real(reference_wave_SPD)

# radial profile of the pupil plane amplitude
pupil_term = P_term[pupil_diameter//2]

# radial profile of the mask diffracted wave amplitude
ampl_term = np.zeros((nMask, pupil_diameter))
ampl_term[0] = b_term_CLA[pupil_diameter//2]
ampl_term[1] = b_term_SPH[pupil_diameter//2]
ampl_term[2] = b_term_SPD[pupil_diameter//2]

# Averaged intensity of a given pixel as a function of b
P = 1.0
b_CLA = np.mean(b_term_CLA[P_term != 0.])
b_SPH = np.mean(b_term_SPH[P_term != 0.])
b_SPD = np.mean(b_term_SPD[P_term != 0.])

# phase error
npts    = 1200
phi_arr = (1.2*2*np.pi/npts)*(np.arange(npts)-npts//2)

# wavelength array
wave_Tab = (1.2/npts)*(np.arange(npts)-npts//2)
nm_Tab_CLA   = wave_CLA*1e9*wave_Tab
nm_Tab_SPH   = wave_SPH*1e9*wave_Tab
nm_Tab_SPD   = wave_SPD*1e9*wave_Tab

# Intensity versus pupil plane pixel
## Sinudoid expression
IC0_arr = np.zeros((nMask, npts))
IC0_CLA = P**2 + 2.*b_CLA**2*(1.- np.cos(theta_CLA)) + 2*P*b_CLA*(np.sin(phi_arr)*np.sin(theta_CLA)-np.cos(phi_arr)*(1.-np.cos(theta_CLA))) 
IC0_SPH = P**2 + 2.*b_SPH**2*(1.- np.cos(theta_SPH)) + 2*P*b_SPH*(np.sin(phi_arr)*np.sin(theta_SPH)-np.cos(phi_arr)*(1.-np.cos(theta_SPH))) 
IC0_SPD = P**2 + 2.*b_SPD**2*(1.- np.cos(theta_SPD)) + 2*P*b_SPD*(np.sin(phi_arr)*np.sin(theta_SPD)-np.cos(phi_arr)*(1.-np.cos(theta_SPD))) 

IC0_arr[0] = IC0_CLA
IC0_arr[1] = IC0_SPH
IC0_arr[2] = IC0_SPD

## Linear intensity expression
#IC1_CLA = P**2 + 2.*b_CLA**2*(1.- np.cos(theta_CLA)) + 2*P*b_CLA*(phi_arr*np.sin(theta_CLA)-(1.-np.cos(theta_CLA)))
#IC1_SPH = P**2 + 2.*b_SPH**2*(1.- np.cos(theta_SPH)) + 2*P*b_SPH*(phi_arr*np.sin(theta_SPH)-(1.-np.cos(theta_SPH)))
#IC1_SPD = P**2 + 2.*b_SPD**2*(1.- np.cos(theta_SPD)) + 2*P*b_SPD*(phi_arr*np.sin(theta_SPD)-(1.-np.cos(theta_SPD)))

## Quadratic intensity expression
#IC2_CLA = P**2 + 2.*b_CLA**2*(1.- np.cos(theta_CLA)) + 2*P*b_CLA*(phi_arr*np.sin(theta_CLA)-(1.-0.5*phi_arr**2)*(1.-cos(theta_CLA)))
#IC2_SPH = P**2 + 2.*b_SPH**2*(1.- np.cos(theta_SPH)) + 2*P*b_SPH*(phi_arr*np.sin(theta_SPH)-(1.-0.5*phi_arr**2)*(1.-cos(theta_SPH)))
#IC2_SPD = P**2 + 2.*b_SPD**2*(1.- np.cos(theta_SPD)) + 2*P*b_SPD*(phi_arr*np.sin(theta_SPD)-(1.-0.5*phi_arr**2)*(1.-cos(theta_SPD)))


print('classical mask: {0:0.3f} mean amplitude'.format(b_CLA))
print('SPHERE mask: {0:0.3f} mean amplitude'.format(b_SPH))
print('SPEED mask: {0:0.3f} mean amplitude'.format(b_SPD))

for i in range(3):
    print('mask (min): {0:0.3f}'.format(wave_Tab[IC0_arr[i] == min(IC0_arr[i])][0]))
    print('mask (max): {0:0.3f}'.format(wave_Tab[IC0_arr[i] == max(IC0_arr[i])][0]))
    print(' ')
    
for i in range(3):
    print('mask sensitivity: {0:0.3f}'.format(np.amax(IC0_arr[i])-np.amin(IC0_arr[i])))
    print(' ')  

# ----------------------------------------
# plots
# ----------------------------------------
# show plot
plt.show()

# 2D display of the mask diffracted amplitude for SPEED
fig = plt.figure(0, figsize=(4, 4))
plt.clf()
ax = fig.add_subplot(111)
ax.imshow(b_term_CLA, cmap = cm.viridis)
ax.set_title('Mask diffracted wave for classical mask')

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
    ax.plot(radius_tab, ampl_term[i], label = labelTab[i])
ax.plot(radius_tab, pupil_term, 'k--')
ax.set_xlim([-0.55, 0.55])
ax.set_ylim([-0.05, 1.05])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = (x1_leg, y1_leg), labelspacing=0.1, title = 'Diffracted amplitude b for:')
plt.title('Radial profile of the mask diffracted amplitude')
plt.tight_layout()
if do_plot:
    fig1.savefig('/Users/mndiaye/Dropbox/OCA/projects/Speed/ZELDA-SPEED/plots/\
mask_diffracted_wave.pdf')

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
    ax.plot(wave_Tab, IC0_arr[i], label = labelTab[i])
ax.set_xlim([-0.25, 0.45])
ax.set_ylim([-0.25, 3.75])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = (x2_leg, y2_leg), labelspacing=0.1)
for i in range(3):
    plt.axvline(x=0.0, color='k', linestyle = '--')
    plt.axhline(y=np.amax(IC0_arr[i]), color=cm0(1.*i/(nMask-1)), linestyle = '--')
    plt.axhline(y=np.amin(IC0_arr[i]), color=cm0(1.*i/(nMask-1)), linestyle = '--')
#    plt.axvline(x=wave_Tab[IC0_arr[i] == max(IC0_arr[i])][0], color=cm0(1.*i/(nMask-1)), linestyle = '--')
    plt.axvline(x=wave_Tab[IC0_arr[i] == min(IC0_arr[i])][0], color=cm0(1.*i/(nMask-1)), linestyle = '--')
#    ax.text(-0.2, 3.5-0.3*i, r'{0:0.3f}$\lambda$'.format(wave_Tab[IC0_arr[i] == min(IC0_arr[i])][0]), fontsize=15, horizontalalignment="center", color=cm0(1.*i/(nMask-1)))
#    ax.text(0.41, 3.5-0.3*i, r'{0:0.3f}$\lambda$'.format(wave_Tab[IC0_arr[i] == max(IC0_arr[i])][0]), fontsize=15, horizontalalignment="center", color=cm0(1.*i/(nMask-1)))
#for i in range(3):
#    ax.add_patch(patches.Rectangle((wave_Tab[IC0_arr[i] == min(IC0_arr[i])][0], np.amin(IC0_arr[i])), \
#    wave_Tab[IC0_arr[i] == max(IC0_arr[i])][0]-wave_Tab[IC0_arr[i] == min(IC0_arr[i])][0], \
#    np.amax(IC0_arr[i])-np.amin(IC0_arr[i]),\
#    alpha=(i+1)*0.1, facecolor=cm0(1.*i/(nMask-1))))
plt.title('Normalized exit pupil intensity')
plt.tight_layout()
if do_plot:
    fig2.savefig('/Users/mndiaye/Dropbox/OCA/projects/Speed/ZELDA-SPEED/plots/\
normalized_intensity_exit_pupil.pdf')

  