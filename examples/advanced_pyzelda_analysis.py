import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

from pathlib import Path

#%% parameters

# path = Path('/Users/mndiaye/Dropbox/python/zelda/pyZELDA/')
path = Path('/Users/avigan/Work/GitHub/pyZELDA/data/')
# path = Path('D:/Programmes/GitHub/pyZELDA/')

wave = 1.642e-6

z = zelda.Sensor('CLASSIC')

#%%################################
# Creation of apodized pupil data #
###################################

# Aberration: astigmatism
basis = np.nan_to_num(zernike.zernike_basis(nterms=5, npix=z.pupil_diameter))*1e-9
aberration = 20*basis[4]
gaussian_std = 1

# Gaussian apodized entrance pupil
x_vals = np.arange(z.pupil_diameter)
xx, yy = np.meshgrid(x_vals, x_vals)
cx, cy = x_vals[z.pupil_diameter//2], x_vals[z.pupil_diameter//2]
r = np.sqrt((xx-cx)**2+(yy-cy)**2)
r = r/r[0, z.pupil_diameter//2]

apodizer = np.exp(-(r**2)/2/gaussian_std**2)

#%%#############################
# Simulation of the wavefront. #
################################

# Clear pupil image
clear_pupil = abs(apodizer)**2

zelda_pupil = ztools.propagate_opd_map(aberration, z.mask_diameter, z.mask_depth, z.mask_substrate,
                                        z.mask_Fratio, z.pupil_diameter, apodizer, wave)

#%%#########################
# Wavefront reconstruction #
############################

# Standard analysis
z_opd_standard = z.analyze(clear_pupil, zelda_pupil, wave)

# Advanced analysis
pupil_roi = aperture.disc(z.pupil_diameter, z.pupil_diameter, diameter=True, cpix=False)
z_opd_advanced = z.analyze(clear_pupil, zelda_pupil, wave,
                           use_arbitrary_amplitude=True,
                           refwave_from_clear=True,
                           cpix=False, pupil_roi=pupil_roi)

#%% plot

fig = plt.figure(0, figsize=(24, 4))
plt.clf()

gs = gridspec.GridSpec(ncols=7, nrows=1, figure=fig, width_ratios=[.1,1,1,1,1,1,.1])

ax = fig.add_subplot(gs[0,1])
mappable = ax.imshow(clear_pupil, aspect='equal', vmin=0, vmax=1)
ax.set_title('Clear pupil')

ax = fig.add_subplot(gs[0,0])
cbar1 = fig.colorbar(mappable=mappable, cax=ax)
cbar1.set_label('Normalized intensity')

ax = fig.add_subplot(gs[0,2])
ax.imshow(zelda_pupil, aspect='equal', vmin=0, vmax=1)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(gs[0,3])
ax.imshow(aberration*1e9, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Introduced aberration (nm)')

ax = fig.add_subplot(gs[0,4])
cax = ax.imshow(z_opd_standard, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Reconstructed aberration - standard')

ax = fig.add_subplot(gs[0,5])
cax = ax.imshow(z_opd_advanced, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Reconstructed aberration - advanced')

ax = fig.add_subplot(gs[0,6])
cbar = fig.colorbar(mappable=cax, cax=ax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()
