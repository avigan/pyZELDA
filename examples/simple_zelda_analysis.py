import matplotlib.pyplot as plt
import numpy as np

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture

from pathlib import Path

# path = Path('/Users/mndiaye/Dropbox/python/zelda/pyZELDA/')
path = Path('/Users/avigan/Work/GitHub/pyZELDA/data/')
# path = Path('D:/Programmes/GitHub/pyZELDA/')

wave = 1.642e-6

# internal data
# clear_pupil_files = ['SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3', 'SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3']
# zelda_pupil_files = ['SPHERE_ZELDA_PUPIL_CUBE1_NDIT=3', 'SPHERE_ZELDA_PUPIL_CUBE2_NDIT=3']
# dark_file  = 'SPHERE_BACKGROUND'
# pupil_tel  = False

# on-sky data
clear_pupil_files = ['SPHERE_GEN_IRDIS057_0002']
zelda_pupil_files = ['SPHERE_GEN_IRDIS057_0001']
dark_file  = 'SPHERE_GEN_IRDIS057_0003'
pupil_tel  = True

# ZELDA analysis
z = zelda.Sensor('SPHERE-IRDIS', pupil_telescope=pupil_tel)

clear_pupil, zelda_pupil, center = z.read_files(path, clear_pupil_files, zelda_pupil_files, dark_file,
                                                collapse_clear=True, collapse_zelda=True)

opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave)

# decomposition on Zernike polynomials
basis, coeff, opd_zern = ztools.zernike_expand(opd_map.mean(axis=0), 20)

# plot
fig = plt.figure(0, figsize=(16, 4))
plt.clf()
<<<<<<< HEAD
ax = fig.add_subplot(141)
ax.imshow(clear_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000, origin=1)
ax.set_title('Clear pupil')

ax = fig.add_subplot(142)
ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000, origin=1)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(143)
ax.imshow(opd_map.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma', origin=1)
ax.set_title('OPD map')

ax = fig.add_subplot(144)
cax = ax.imshow(opd_zern.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma', origin=1)
ax.set_title('Zernike projected OPD map')

cbar = fig.colorbar(cax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()
=======
ax = fig.add_subplot(131)
ax.imshow(clear_pupil, aspect='equal')
ax.title('Clear pupil')

ax = fig.add_subplot(132)
ax.imshow(zelda_pupil.mean(axis=0), aspect='equal')
ax.title('ZELDA pupil')

ax = fig.add_subplot(133)
ax.imshow(opd_map.mean(axis=0), aspect='equal', vmin=-100, vmax=-100)
ax.title('OPD map')

plt.tight_layout()
>>>>>>> Added new example SPHERE data
