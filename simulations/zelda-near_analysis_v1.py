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

#fdir = Path('/Users/mndiaye/Dropbox/LAM/Research projects/ZELDA/ZELDA-IR/20190406').resolve()
fdir = Path('/Users/mndiaye/Dropbox/LAM/Research projects/ZELDA/ZELDA-IR/20190407').resolve()

# internal data
#clear_pupil_files = ['20190406_zelda_offmask1p1mm']
#zelda_pupil_files = ['20190406_zelda_centered',
#                     '20190406_zelda_centered_defocus2mm']

clear_pupil_files = ['20190407_zelda_pah2_decentered']
zelda_pupil_files = ['20190407_zelda_pah2_centered_01',
                     '20190407_zelda_pah2_centered_02']


dark_file  = 'NEAR_BACKGROUND'
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
zelda_pupil0_init = fits.getdata(str(fpath))

fname = zelda_pupil_files[1] + '.fits'
fpath = fdir / fname
zelda_pupil1_init = fits.getdata(str(fpath))

#%%
#xi = 226
#xf = 582
#yi = 227
#yf = 583

nPup= 356

xi = 77
xf = xi + nPup
yi = 227
yf = yi + nPup

clear_pupil0 = clear_pupil_init[xi:xf,yi:yf]
zelda_pupil0 = zelda_pupil0_init[xi:xf,yi:yf]
zelda_pupil1 = zelda_pupil1_init[xi:xf,yi:yf]

clear_pupil = [clear_pupil0]
zelda_pupil = [zelda_pupil0, zelda_pupil1]

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
z = zelda.Sensor('NEAR')

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
fname = 'zelda_ctr=01.pdf'
fpath = fdir / fname


pl.figure(1, figsize = (5,5))
pl.clf()
pl.imshow(zelda_pupil[0])
pl.title('zelda pupil, centered 01')
pl.show()
if do_pdf:
    pl.savefig(str(fpath), transparent = True)


fname = 'zelda_ctr=02.pdf'
fpath = fdir / fname

pl.figure(2, figsize = (5,5))
pl.clf()
pl.imshow(zelda_pupil[1])
pl.title('zelda pupil, centered 02')
pl.show()
if do_pdf:
    pl.savefig(str(fpath), transparent = True)


#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave)

#%%
# display of the opd map
fname = 'zelda_ctr=01_opd.pdf'
fpath = fdir / fname


f2 = pl.figure(10, figsize = (6,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(opd_map[0], vmin=-3000, vmax=3000)
ax0.set_title('opd map, wo defocus')

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
fname = 'zelda_ctr=02_opd.pdf'
fpath = fdir / fname


f2 = pl.figure(11, figsize = (6,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(opd_map[1], vmin=-3000, vmax=3000)
ax0.set_title('opd map, with defocus')
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
basis1, coeff1, opd_zern1 = ztools.zernike_expand(opd_map[1], 20)


#%%
# display of the opd map
fname = 'zelda_ctr=01_opd_reconstr.pdf'
fpath = fdir / fname


f2 = pl.figure(12, figsize = (6,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(opd_zern0[0], vmin=-3000, vmax=3000)
ax0.set_title('opd map, from zernike coefficients - wo defocus')

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
fname = 'zelda_ctr=02_opd_reconstr.pdf'
fpath = fdir / fname


f2 = pl.figure(13, figsize = (6,4.5))
pl.clf()
ax0 = f2.add_subplot(111)
im = ax0.imshow(opd_zern1[0], vmin=-3000, vmax=3000)
ax0.set_title('opd map, from zernike coefficients - with defocus')
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

#%%
width0=0.4


fname = 'zelda_coeffs.pdf'
fpath = fdir / fname


bins = np.linspace(1, 20, 20)

f2 = pl.figure(20, figsize=(8, 4.5))
pl.clf()
pl.bar(bins-width0/2, coeff0[0], width=width0, label='wo defocus')
pl.bar(bins+width0/2, coeff1[0], width=width0, label='with defocus')
pl.xticks(bins)
pl.xlabel('Zernike modes')
pl.ylabel('wavefront error in nm rms')
pl.legend(loc='upper right')

if do_pdf:
    pl.savefig(str(fpath), transparent = True)

pl.show()
pl.tight_layout()