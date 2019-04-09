#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:55:25 2019

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%%
"""
### Initialization
"""
import numpy as np
import pylab as pl

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools

from pathlib import Path 

from astropy.io import fits

import scipy.ndimage as ndimage

#%%
"""
### Visir near pupil function
"""
#def visir_near_pupil(dim=356):
#    ''' NEAR pupil
#    
#    Parameters
#    ----------
#    dim : int
#        Size of the output array. Default is 384
#    
#    Returns
#    -------
#    pup : array
#        An array containing a disc with the specified parameters
#        
#    '''
#
#    # fixed diameter
#    diameter = 356
#
#    if dim < diameter:
#        raise ValueError('Image dimensions cannot be smaller than 356 pixels')
#    
#    x = (2/diameter)*(np.arange(dim)-dim/2)
#    y = (2/diameter)*(np.arange(dim)-dim/2)    
#    xx, yy = np.meshgrid(x, y)
#    
#    indx0 = xx <= -0.11
#    indx1 = xx >= 0.11
#    
#    xx1 = np.ones((dim, dim))
#    xx1[indx0] = 0.
#    xx1[indx1] = 0.
#    
#    indy0 = yy <= 0.0
#    indy1 = yy >= 0.22
#    
#    yy1 = np.ones((dim, dim))
#    yy1[indy0] = 0.
#    yy1[indy1] = 0.
#    
#    tt1 = xx1*yy1
#    
#    angle = -39
#    tt2 = ndimage.rotate(tt1, angle)
#    
#    dima = np.shape(tt2)[0]
#    tt3 = tt2[dima//2-dim//2:dima//2+dim//2, dima//2-dim//2:dima//2+dim//2]
#    tt3 = np.round(tt3)
#
#    pupil00 = zelda.aperture.vlt_pupil(dim, diameter, spiders_orientation=0, 
#                        dead_actuators=[],dead_actuator_diameter=0.)
#    pupil11 = pupil00*(1-tt3)
#    pupil12 = ndimage.rotate(pupil11, 97)
#    
#    dimb = np.shape(pupil12)[0]
#    near_pupil = pupil12[dimb//2-dim//2:dimb//2+dim//2, dimb//2-dim//2:dimb//2+dim//2]
#    near_pupil = np.round(near_pupil) 
#    
#    return near_pupil



#%%
"""
### Paths and directories
"""
#fdir = Path('/Users/mndiaye/Dropbox/LAM/Research projects/ZELDA/ZELDA-IR/20190406').resolve()
fdir = Path('/Users/mndiaye/Dropbox/LAM/Research projects/ZELDA/ZELDA-IR/20190408').resolve()

clear_pupil_files = ['20190408_zelda_pah2_onsky_offaxis_01']
zelda_pupil_files = ['20190408_zelda_pah2_onsky_01']


dark_file  = '20190408_zelda_pah2_onsky_background_01'
pupil_tel  = True

#%%
"""
### read files
"""
fname = clear_pupil_files[0] + '.fits'
fpath = fdir / fname
clear_pupil_init = fits.getdata(str(fpath))

fname = zelda_pupil_files[0] + '.fits'
fpath = fdir / fname
zelda_pupil_init = fits.getdata(str(fpath))

fname = dark_file + '.fits'
fpath = fdir / fname
bckgd_pupil_init = fits.getdata(str(fpath))

#%%
nPup= 356

xi = 338
xf = xi + nPup
yi = 227
yf = yi + nPup

clear_pupil = clear_pupil_init[xi:xf,yi:yf]
zelda_pupil = zelda_pupil_init[xi:xf,yi:yf]
bckgd_pupil = bckgd_pupil_init[xi:xf,yi:yf]

clear_pupil = [clear_pupil-bckgd_pupil]
zelda_pupil = [zelda_pupil-bckgd_pupil]

clear_pupil = np.asarray(clear_pupil)
zelda_pupil = np.asarray(zelda_pupil)

#%%
"""
### Parameters
"""
#wave = 10.6e-6
wave = 11.25e-6

do_pdf = True

#%%
"""
### Generate near pupil
"""
near_pupil = zelda.aperture.visir_near_pupil(nPup)

pl.figure(1)
pl.clf()
pl.imshow(near_pupil)
pl.show()


#%%
"""
### Zelda 
"""
# Sensor
z = zelda.Sensor('NEAR', pupil_telescope = True)

#%%
pl.figure(10)
pl.clf()
pl.imshow(z.pupil)
pl.show()

#%%
# display of the maps
fname = 'clear_pupil.pdf'
fpath = fdir / fname


pl.figure(0, figsize = (5,5))
pl.clf()
pl.imshow(clear_pupil[0])
pl.title('clear pupil')
pl.show()
if do_pdf:
    pl.savefig(str(fpath), transparent = True)


#%%
fname = 'zelda_pupil.pdf'
fpath = fdir / fname


pl.figure(1, figsize = (5,5))
pl.clf()
pl.imshow(zelda_pupil[0])
pl.title('zelda pupil')
pl.show()
if do_pdf:
    pl.savefig(str(fpath), transparent = True)



#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave, ratio_limit=20)

#%%
# display of the opd map
fname = 'zelda.pdf'
fpath = fdir / fname


f2 = pl.figure(10, figsize = (6,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(opd_map[0], vmin=-3000, vmax=3000)
ax0.set_title('opd map')

f2.subplots_adjust(bottom=0.13, top=0.87, left=0.1, right=0.75,
                    wspace=0.02, hspace=0.02)
#
#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.8, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('wavefront error in nm', rotation=270, labelpad = 16)

if do_pdf:
    pl.savefig(str(fpath), transparent = True)

pl.show()
pl.tight_layout()


#%%
# decomposition on Zernike polynomials
basis0, coeff0, opd_zern0 = ztools.zernike_expand(opd_map[0], 20)


#%%
# display of the opd map
fname = 'zelda_opd_reconstr.pdf'
fpath = fdir / fname


f2 = pl.figure(12, figsize = (6,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(opd_zern0[0], vmin=-3000, vmax=3000)
ax0.set_title('opd map, from zernike coefficients - ctr 01')

f2.subplots_adjust(bottom=0.13, top=0.87, left=0.1, right=0.75,
                    wspace=0.02, hspace=0.02)
#
#f2.subplots_adjust(right=0.8)
cbar_ax = f2.add_axes([0.8, 0.15, 0.05, 0.7])
cbar    = f2.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('wavefront error in nm', rotation=270, labelpad = 16)

if do_pdf:
    pl.savefig(str(fpath), transparent = True)

pl.show()
pl.tight_layout()


#%%
"""
### plot
"""

width0=0.4


fname = 'zelda_coeffs.pdf'
fpath = fdir / fname


bins = np.linspace(1, 20, 20)

f2 = pl.figure(20, figsize=(8, 4.5))
pl.clf()
pl.bar(bins, coeff0[0], width=width0, label='zelda map')
#pl.bar(bins+width0/2, coeff1[0], width=width0, label='ctr 02')
pl.xticks(bins)
pl.xlabel('Zernike modes')
pl.ylabel('wavefront error in nm rms')
pl.legend(loc='upper right')

if do_pdf:
    pl.savefig(str(fpath), transparent = True)

pl.show()
pl.tight_layout()

#%%
pupil00 = zelda.aperture.vlt_pupil(nPup, nPup, 
                                   spiders_orientation=0,  
                                   dead_actuators=[],
                                   dead_actuator_diameter=0.)


pupilm3 = zelda.aperture.vlt_pupil(nPup, nPup, 
                                   spiders_orientation=97,  
                                   dead_actuators=[],
                                   dead_actuator_diameter=0.)

pl.figure(30)
pl.clf()
pl.imshow(pupilm3*clear_pupil[0])
pl.show()

#%%
x = (2/nPup)*(np.arange(nPup)-nPup/2)
y = (2/nPup)*(np.arange(nPup)-nPup/2)

xx, yy = np.meshgrid(x, y)

indx0 = xx <= -0.11
indx1 = xx >= 0.11

xx1 = np.ones((nPup, nPup))
xx1[indx0] = 0.
xx1[indx1] = 0.

indy0 = yy <= 0.0
indy1 = yy >= 0.22

yy1 = np.ones((nPup, nPup))
yy1[indy0] = 0.
yy1[indy1] = 0.

tt1 = xx1*yy1


#%%
angle = -39
tt2 = rotate(tt1, angle)

dim = np.shape(tt2)[0]
tt3 = tt2[dim//2-nPup//2:dim//2+nPup//2, dim//2-nPup//2:dim//2+nPup//2]
tt3 = np.round(tt3)

#%%

pupil11 = pupil00*(1-tt3)
pupil12 = rotate(pupil11, 97)
dimb = np.shape(pupil12)[0]
pupil13 = pupil12[dimb//2-nPup//2:dimb//2+nPup//2, dimb//2-nPup//2:dimb//2+nPup//2]
pupil13 = np.round(pupil13) 

pl.figure(30)
pl.clf()
pl.imshow(pupil13)
pl.show()


pl.figure(31)
pl.clf()
pl.imshow(clear_pupil[0]*pupil13)
pl.show()


fname = 'vlt_pupil_near.fits'
fpath = fdir / fname
fits.writeto(fpath, pupil13)