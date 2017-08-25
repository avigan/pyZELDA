# compatibility with python 2.7
from __future__ import absolute_import, division, print_function

import sys
import os
import matplotlib.pyplot as plt

path = '/Users/mndiaye/Dropbox/python/zelda/pyZELDA/'
#path = '/Users/avigan/Work/GitHub/pyZELDA/'
print(path)
if path not in sys.path:
    sys.path.append(path)
print(path)

import pyzelda.zelda as zelda


data_path = os.path.join(path, 'data/')

wave = 1.642e-6

clear_pupil_files = ['SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3', 'SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3']
zelda_pupil_files = ['SPHERE_ZELDA_PUPIL_CUBE1_NDIT=3', 'SPHERE_ZELDA_PUPIL_CUBE2_NDIT=3']
dark_file = 'SPHERE_BACKGROUND'

clear_pupil, zelda_pupil, center = zelda.read_files(data_path, clear_pupil_files, zelda_pupil_files, dark_file,
                                                    pupil_diameter=384, collapse_clear=False, collapse_zelda=False)

opd_map = zelda.analyze(clear_pupil, zelda_pupil, wave=wave)

basis, coeff, opd_zern = zelda.zernike_expand(opd_map.mean(axis=0), 20)

fig = plt.figure(0, figsize=(16, 4))
plt.clf()
ax = fig.add_subplot(141)
ax.imshow(clear_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000)
ax.set_title('Clear pupil')

ax = fig.add_subplot(142)
ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(143)
ax.imshow(opd_map.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma')
ax.set_title('OPD map')

ax = fig.add_subplot(144)
cax = ax.imshow(opd_zern.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma')
ax.set_title('Zernike projected OPD map')

cbar = fig.colorbar(cax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()
