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
from matplotlib.patches import Circle

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools

from pathlib import Path 

from astropy.io import fits

#%%
"""
### Parameters
"""
wave = 10.7e-6
#wave = 11.25e-6

do_pdf = True

# number of Zernike coefficients for the opd map projection
nZern = 20

#%%
"""
### Paths and directories
"""
# Directory
fdir = Path('/Users/mndiaye/Dropbox/LAM/Research projects/ZELDA/ZELDA-IR/20190410').resolve()

# internal source data
clear_pupil_files = ['20190409_zelda_b10p7_cyZm3325_cu_background_01']
zelda_pupil_files = ['20190409_zelda_b10p7_cyZm3325_cu_01',
                     '20190409_zelda_b10p7_cyZm3325_cu_02']

nFiles = np.shape(zelda_pupil_files)[0]


#%%
"""
### read files
"""
# clear pupil
fname = clear_pupil_files[0] + '.fits'
fpath = fdir / fname
clear_pupil_init = fits.getdata(str(fpath))

# clear pupil dimensions
dims = np.shape(clear_pupil_init)
dimx = dims[0]
dimy = dims[1]

# zelda pupil
zelda_pupil_init = np.zeros((nFiles,dimx, dimy))
for i, val in enumerate(zelda_pupil_files):  
    fname = val + '.fits'
    fpath = fdir / fname
    zelda_pupil_init[i] = fits.getdata(str(fpath))


#%%
"""
### Image cropping
"""
# pupil dimension (make sure this matches with instrument 'NEAR2')
nPup=334

# initial and final coordinates for image extraction
xi = 341
xf = xi + nPup
yi = 244
yf = yi + nPup

# image cropping for clear pupil
clear_pupil0 = clear_pupil_init[xi:xf,yi:yf]
clear_pupil = [clear_pupil0]
clear_pupil = np.asarray(clear_pupil)

# image cropping for zelda pupil
zelda_pupil = np.empty((nFiles, nPup, nPup)) 
for i in range(nFiles):
    zelda_pupil[i] = zelda_pupil_init[i, xi:xf,yi:yf]

#%%
"""
### Image display
"""
# parameters for circle definition
cx0 = nPup/2
cy0 = nPup/2
cr0 = nPup/2

# clear pupil image
fname = 'clear_pupil.pdf'
fpath = fdir / fname

# definition of a circle for the pupil
circle0 = Circle((cx0, cy0), cr0, color='white', fill = False)

# displau of the clear pupil image
f0 = pl.figure(0, figsize = (5,5))
pl.clf()
ax = f0.add_subplot(111)
ax.imshow(clear_pupil[0])
ax.add_artist(circle0)
ax.set_title('clear pupil')
pl.show()
if do_pdf:
    pl.savefig(str(fpath), transparent = True)

#%%
    
for i in range(nFiles):
    # zelda pupil image 1
    fname = 'zelda_ctr={0:02d}.pdf'.format(i+1,)
    fpath = fdir / fname
    
    # definition of a circle for the pupil
    circle1 = Circle((cx0, cy0), cr0, color='white', fill = False)
    
    # displau of the zelda pupil image 1
    f1 = pl.figure(i+1, figsize = (5,5))
    pl.clf()
    ax0 = f1.add_subplot(111)
    ax0.imshow(zelda_pupil[0])
    ax0.add_artist(circle1)
    ax0.set_title('zelda pupil, ctr {0:02d}'.format(i+1,))
    if do_pdf:
        pl.savefig(str(fpath), transparent = True)

pl.show()

#%%
"""
### Zelda 
"""
# Sensor definition (instrument is NEAR, clear aperture for internal source)
z = zelda.Sensor('NEAR')

#%%
# OPD map extraction
opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave, ratio_limit=20)

#%%
# decomposition on Zernike polynomials
basis, coeff, opd_zern = ztools.zernike_expand(opd_map, nZern)

#%%
"""
### Display OPD maps
"""
for i in range(nFiles):
    # zelda opd map 1
    fname = 'zelda_ctr={0:02d}_opd.pdf'.format(i+1)
    fpath = fdir / fname
    
    # display of the zelda opd map 1
    f2 = pl.figure(11+i, figsize = (6,4.5))
    pl.clf()
    ax0 = f2.add_subplot(111)
    im = ax0.imshow(opd_map[i], vmin=-3000, vmax=3000)
    ax0.set_title('opd map, ctr {0:02d}'.format(i+1))
    
    f2.subplots_adjust(bottom=0.13, top=0.87, left=0.1, right=0.75,
                        wspace=0.02, hspace=0.02)
    cbar_ax = f2.add_axes([0.8, 0.15, 0.05, 0.7])
    cbar    = f2.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('wavefront error in nm', rotation=270, labelpad = 16)
    
    if do_pdf:
        pl.savefig(str(fpath), transparent = True)
    
    pl.tight_layout()

pl.show()


#%%
"""
### Display reconstructed OPD maps
"""
for i in range(nFiles):
    # reconstructed opd map 1
    fname = 'zelda_ctr={0:02d}_opd_reconstr.pdf'.format(i+1)
    fpath = fdir / fname
    
    # display of the reconstructed opd map 1
    f2 = pl.figure(21+i, figsize = (6,4.5))
    pl.clf()
    ax0 = f2.add_subplot(111)
    im = ax0.imshow(opd_zern[i], vmin=-3000, vmax=3000)
    ax0.set_title('opd map, from zernike coefficients - ctr {0:02d}'.format(i+1))
    
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
### Zernike coefficient display
"""
# bar width
width0=0.4

# Zernike coefficients
fname = 'zelda_coeffs.pdf'
fpath = fdir / fname

# number of bins for the 
bins = np.linspace(1, nZern, nZern)

# display of the Zernike
f2 = pl.figure(30, figsize=(8, 4.5))
pl.clf()
for i in range(nFiles):
    pl.bar(bins-((2*i-1)*(width0/2)), coeff[i], width=width0, label='ctr {0:02d}'.format(i+1))
pl.xticks(bins)
pl.xlabel('Zernike modes')
pl.ylabel('wavefront error in nm rms')
pl.legend(loc='upper right')

if do_pdf:
    pl.savefig(str(fpath), transparent = True)

pl.show()
pl.tight_layout()
