import matplotlib.pyplot as plt
import numpy as np

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

from pathlib import Path

# path = Path('/Users/mndiaye/Dropbox/python/zelda/pyZELDA/')
path = Path('/Users/avigan/Work/GitHub/pyZELDA/data/')
# path = Path('D:/Programmes/GitHub/pyZELDA/')

wave = 1.642e-6

z = zelda.Sensor('CLASSIC')

###################################
# Creation of apodized pupil data #
###################################

# Aberration: astigmatism

basis = np.nan_to_num(zernike.zernike_basis(nterms=5, npix=z.pupil_diameter))*1e-9
aberration = 8*basis[4]
gaussian_std = 10

# Gaussian apodized entrance pupil

x_vals = np.arange(z.pupil_diameter)
xx, yy = np.meshgrid(x_vals, x_vals)
cx, cy = x_vals[z.pupil_diameter//2], x_vals[z.pupil_diameter//2]
r = np.sqrt((xx-cx)**2+(yy-cy)**2)
r = r/r[0, z.pupil_diameter//2]

apodizer = np.exp(-(r**2)/2/gaussian_std**2)
################################
# Simulation of the wavefront. #
################################

# Clear pupil image
clear_pupil = abs(apodizer)**2

# physical diameter and depth, in m
d_m = z.mask_diameter
z_m = z.mask_depth

# substrate refractive index
n_substrate = ztools.refractive_index(wave, z.mask_substrate)

# R_mask: mask radius in lam0/D unit
R_mask = 0.5 * d_m / (wave * z.mask_Fratio)

# ++++++++++++++++++++++++++++++++++
# Dimensions
# ++++++++++++++++++++++++++++++++++

# array and pupil
array_dim = z.pupil.shape[-1]
pupil_radius = z.pupil_diameter // 2

# mask sampling in the focal plane
D_mask_pixels = 300

# ++++++++++++++++++++++++++++++++++
# Numerical simulation part
# ++++++++++++++++++++++++++++++++++

# --------------------------------
# plane A (Entrance pupil plane)

# definition of m1 parameter for the Matrix Fourier Transform (MFT)
# here equal to the mask size
m1 = 2 * R_mask * (array_dim / (2. * pupil_radius))

# definition of the electric field in plane A in the presence of aberrations
ampl_PA = apodizer * np.exp(1j * 2. * np.pi * aberration / wave)

# --------------------------------
# plane B (Focal plane)

# calculation of the electric field in plane B with MFT within the Zernike
# sensor mask
ampl_PB = mft.mft(ampl_PA, array_dim, D_mask_pixels, m1)

# restriction of the MFT with the mask disk of diameter D_mask_pixels/2
ampl_PB *= aperture.disc(D_mask_pixels, D_mask_pixels, diameter=True, cpix=True, strict=False)

# --------------------------------
# plane C (Relayed pupil plane)

# mask phase shift theta (mask in transmission)
theta = 2 * np.pi * (n_substrate - 1) * z_m / wave

# phasor term associated  with the phase shift
expi = np.exp(1j * theta)

# --------------------------------
# definition of parameters for the phase estimate with Zernike

# b1 = reference_wave: parameter corresponding to the wave diffracted by the mask in the relayed pupil
ampl_PC = ampl_PA - (1 - expi) * mft.imft(ampl_PB, D_mask_pixels, array_dim, m1)

zelda_pupil = np.abs(ampl_PC) ** 2

#zelda_pupil = ztools.propagate_opd_map(aberration, z.mask_diameter, z.mask_depth, z.mask_substrate,
#                                        z.mask_Fratio, z.pupil_diameter, apodizer, wave)

# Standard analysis
z_opd_standard = z.analyze(clear_pupil, zelda_pupil, wave)

# Advanced analysis
pupil_roi = aperture.disc(z.pupil_diameter, z.pupil_diameter, diameter=True, cpix=False)
z_opd_advanced = z.analyze(clear_pupil, zelda_pupil, wave,
                           use_arbitrary_amplitude=True, refwave_from_clear=True,
                           cpix=False, pupil_roi=pupil_roi)



# plot
fig = plt.figure(0, figsize=(20, 4))
plt.clf()

ax = fig.add_subplot(151)
ax.imshow(clear_pupil, aspect='equal')#, vmin=0, vmax=1, origin=1)
ax.set_title('Clear pupil')

ax = fig.add_subplot(152)
toto = ax.imshow(zelda_pupil, aspect='equal')#, vmin=0, vmax=10, origin=1)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(153)
ax.imshow(aberration*1e9, aspect='equal', vmin=-15, vmax=15, cmap='magma', origin=1)
ax.set_title('Introduced aberration (nm)')

ax = fig.add_subplot(154)
cax = ax.imshow(z_opd_standard, aspect='equal', vmin=-15, vmax=15, cmap='magma', origin=1)
ax.set_title('Reconstructed aberration - standard')

ax = fig.add_subplot(155)
cax = ax.imshow(z_opd_advanced, aspect='equal', vmin=-15, vmax=15, cmap='magma', origin=1)
ax.set_title('Reconstructed aberration - advanced')

cbar = fig.colorbar(toto)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()
