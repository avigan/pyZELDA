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


fig = plt.figure(0, figsize=(16, 5))
plt.clf()
ax = fig.add_subplot(131)
ax.imshow(clear_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000)
ax.set_title('Clear pupil')

ax = fig.add_subplot(132)
ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(133)
cax = ax.imshow(opd_map.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma')
ax.set_title('OPD map')

cbar = fig.colorbar(cax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
