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

near_pupil = zelda.aperture.visir_near_pupil(dim=356)
dark = bckgd_pupil[near_pupil == 1.].mean()

clear_pupil = [clear_pupil-dark]
zelda_pupil = [zelda_pupil-dark]

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
im = ax0.imshow(opd_map[0], vmin=-300, vmax=300)
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
im = ax0.imshow(opd_zern0[0], vmin=-300, vmax=300)
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
