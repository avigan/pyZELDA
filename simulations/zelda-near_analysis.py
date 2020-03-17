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

#%%
"""
### Paths and directories
"""

fdir = Path('/Users/mndiaye/Dropbox/LAM/Research projects/ZELDA/ZELDA-IR/4zelda').resolve()

# internal data
clear_pupil_files = ['20190223_sourcem663_beamp2595_pointm2000_pah2_withoutiris_pupil_decentered']
zelda_pupil_files = ['20190223_sourcem563_beamp2595_pointm2000_pah2_withoutiris_pupil_centered',
                     '20190223_sourcem563_beamp2595_pointp2000_pah2_withoutiris_pupil_centered']
dark_file  = 'NEAR_BACKGROUND'
pupil_tel  = False

#%%
# create background
#bkg = np.empty((1024, 1024))
#
#fname = clear_pupil_files[0] + '.fits'
#fpath = fdir / fname
#
#data0 = fits.getdata(str(fpath))
#
##%%
#pl.figure(0)
#pl.imshow(data0[600:,600:])
#pl.show()
#
#
##%%
#bkg_val = np.mean(data0[600:, 600:])
#print(bkg_val)
#
#bkg += bkg_val
#
##%%
#fname = 'NEAR_BACKGROUND.fits'
#fpath = fdir / fname
#fits.writeto(fpath,bkg, overwrite = True)

#%%
"""
### read files
"""
#%%
# check pupil center
xc = 404
yc = 516
x0 = 150
y0 = 250
sz = 356


#%%
"""
### Parameters
"""
wave = 11.25e-6

#%%
"""
### Zelda 
"""
# Sesnor
z = zelda.Sensor('NEAR', pupil_telescope = pupil_tel)

#%%
# File reading
clear_pupil, zelda_pupil, center = z.read_files(fdir, clear_pupil_files, zelda_pupil_files, dark_file,
                                                center = (xc-x0, yc-y0),
                                                collapse_clear=False, collapse_zelda=False)

#%%
pl.figure(0)
pl.imshow(clear_pupil[0])
pl.title('clear pupil')
pl.show()

pl.figure(1)
pl.imshow(zelda_pupil[0])
pl.title('zelda pupil, wo defocus')
pl.show()

pl.figure(2)
pl.imshow(zelda_pupil[1])
pl.title('zelda pupil, with defocus')
pl.show()

#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave)

#%%
pl.figure(10)
pl.imshow(opd_map[0], vmin=-3000, vmax=3000)
pl.title('opd map, wo defocus')
pl.show()

pl.figure(11)
pl.imshow(opd_map[1], vmin=-3000, vmax=3000)
pl.title('opd map, with defocus')
pl.show()

#%%
# decomposition on Zernike polynomials
basis0, coeff0, opd_zern0 = ztools.zernike_expand(opd_map[0], 20)
basis1, coeff1, opd_zern1 = ztools.zernike_expand(opd_map[1], 20)

#%%
pl.figure(12)
pl.imshow(opd_zern0[0])
pl.title('opd map, from zernike coefficients - wo defocus')
pl.show()

pl.figure(13)
pl.imshow(opd_zern1[0])
pl.title('opd map, from zernike coefficients - with defocus')
pl.show()

#%%
"""
### plot
"""
#fig = pl.figure(0, figsize=(16, 4))
#pl.clf()
#ax = fig.add_subplot(141)
#ax.imshow(clear_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000, origin=1)
#ax.set_title('Clear pupil')
#
#ax = fig.add_subplot(142)
#ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000, origin=1)
#ax.set_title('ZELDA pupil')
#
#ax = fig.add_subplot(143)
#ax.imshow(opd_map.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma', origin=1)
#ax.set_title('OPD map')
#
#ax = fig.add_subplot(144)
#cax = ax.imshow(opd_zern.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma', origin=1)
#ax.set_title('Zernike projected OPD map')
#
#cbar = fig.colorbar(cax)
#cbar.set_label('OPD [nm]')
#
#pl.tight_layout()
#pl.show()


