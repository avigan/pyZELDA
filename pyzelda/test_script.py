from pyzelda import zelda
import numpy as np
import matplotlib.pyplot as plt
from pyzelda import ztools
from pyzelda.utils import aperture

import imp
imp.reload(zelda)
imp.reload(zelda.ztools)

pupil_diameter = 430
wave = 670.7e-9

basis = np.nan_to_num(zelda.zernike.zernike_basis(npix=430)) * 5e-9
b2 = zelda.zernike.zernike_basis(npix=430)*5e-9
sensor = zelda.Sensor('MITHIC')
pupil = zelda.aperture.disc(pupil_diameter, pupil_diameter, diameter=True, cpix=True, strict=False)

#%% Sensitivity test for the primary aberrations

sensor._corono = 200
nbpoints = 10

zzpod = []
stack2 = np.zeros((nbpoints,15,430,430))
stackl = np.zeros((nbpoints, 15, 430, 430))
extended_basis = np.array([i*basis for i in range(nbpoints)])
rmss = 5*np.arange(10)*1e-9 # RMS in m
coeffstack2 = np.zeros((nbpoints, 15))
coeffstackl = np.zeros((nbpoints, 15))

def zelda_analysis(opd, sensor):

    clear = zelda.ztools.propagate_corono(opd, wave, sensor.corono, pupil_diameter, pupil)
    clearmod = abs(clear)**2
    clearmod /= clearmod.max()
    zeldagram = sensor.propagate_opd_map(opd, wave, corono=sensor.corono)
    zopd = (sensor.analyze(clearmod, zeldagram, wave)).squeeze()

    return zopd*1e-9

def zelda_linear_analyze(sensor, clear_pup, zelda_pup, reference_wave=None, wave=wave, corono=0):
    n_substrate = ztools.refractive_index(wave, 'fused_silica')
    theta = 2*np.pi*(n_substrate-1)*sensor._mask_depth/wave

    pupil_diameter = 430
    pupil = aperture.disc(430,430, diameter=True, cpix=True, strict=False)
     
    if not(reference_wave):
        reference_wave, _ = ztools.create_reference_wave(sensor._mask_diameter, sensor._mask_depth, \
                                                          sensor._mask_substrate, sensor._Fratio, \
                                                          pupil_diameter, pupil, wave, corono=corono)
    opd = 1/np.sin(theta) * (zelda_pup/2/clear_pup/reference_wave - clear_pup/2/reference_wave + (1-reference_wave/clear_pup)*(1-np.cos(theta)))

    return opd

def zlinanal(opd, sensor):

    modified_clear = zelda.ztools.propagate_corono(opd, wave, sensor.corono, pupil_diameter, pupil)
    clear = aperture.disc(430, 430, diameter=True, cpix=True, strict=False)
    zeldagram = sensor.propagate_opd_map(opd, wave, corono=sensor.corono)
    zopd = zelda_linear_analyze(sensor, modified_clear, zeldagram,  corono=sensor.corono) * wave / 2 /np.pi

    return zopd


def zelda_analysis_sscorono(opd, sensor):
    clear = pupil
    zeldagram = sensor.propagate_opd_map(opd, wave, corono=0)
    zopd = sensor.analyze(clear, zeldagram, wave).squeeze()

    return zopd*1e-9

for i, basi in enumerate(extended_basis):
    for j, basij in enumerate(basi):
        zopd2 = zelda_analysis(basij, sensor).squeeze()
        zopdl = zlinanal(basij, sensor)
        stackl[i, j] = zopdl
        stack2[i, j] = zopd2
        
        _, rcoeff2, _ = ztools.zernike_expand(zopd2, 15)
        _, rcoeffl, _ = ztools.zernike_expand(zopdl, 15)
        
        coeffstack2[i, j] = rcoeff2[0,j]
        coeffstackl[i, j] = rcoeffl[0,j]
#%% Plot 2nd order

plt.figure(figsize=(12,21))
plt.subplot(531)

for i in range(15):
    plt.subplot(5,3,i+1)
    plt.plot(rmss, coeffstack2[:,i], 'b*')
    plt.plot(rmss, rmss, 'r')
    plt.axis('equal')
    #plt.xlabel('Zernike number {}'.format(str(i)))

plt.savefig('/mnt/c/Users/rpourcelot/Desktop/test2{}.png'.format(sensor.corono))
plt.tight_layout()
plt.show()

#%% Plot 2nd order

plt.figure(figsize=(12,21))
plt.subplot(531)

for i in range(15):
    plt.subplot(5,3,i+1)
    plt.plot(rmss, coeffstackl[:,i], 'b*')
    plt.plot(rmss, rmss, 'r')
    plt.axis('equal')
    #plt.xlabel('Zernike number {}'.format(str(i)))

plt.savefig('/mnt/c/Users/rpourcelot/Desktop/testl{}.png'.format(sensor.corono))
plt.tight_layout()
plt.show()
        
'''
Test the Zernike modes for different coronograph sizes, see what happens.

Plot the sensor response for each mode: measured rms of the mode vs introduced rms of the mode
+
36666666666666666666666666666
.03
++++++++.

......
Try to substract the piston pattern, see if what happens is a static pattern.

Compute the low order modes on a smaller part of the pupil.

Generate figures  '''






def zelda_loop(nloop, initial_opd, sensor, gain=1):

    stack = []
    opd = initial_opd

    for i in range(nloop):
        clear = zelda.ztools.propagate_corono(opd, wave, sensor.corono, pupil_diameter, pupil)
        clearmod = abs(clear)**2
        zeldagram = sensor.propagate_opd_map(opd, wave, corono=sensor.corono)

        zopd = (sensor.analyze(clearmod, zeldagram, wave)).squeeze()

        stack.append(zopd)

        opd=gain*1e-9*zopd
        

    return np.array(stack)





'''plt.figure(figsize=(15, 15))
plt.subplot(5, 3, 1)
zzopd = []
b = []
for i, j in enumerate(basis):
    zopd = zelda_analysis(j, sensor).squeeze()
    b.append(zopd)
    #b[-1] = b[-1] - b[0]
    toto1, toto2, toto3 = zelda.ztools.zernike_expand(zopd)
    zzopd.append(toto2[0,i])
    

    plt.subplot(5,3,i+1)
    plt.imshow(b[i])
    plt.colorbar()

diff = basis - b
relerr = diff / basis

plt.tight_layout()
plt.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9, wspace=.00001, hspace=.1)
plt.show()'''
