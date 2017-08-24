import sys
import os
import matplotlib.pyplot as plt

path = '/Users/avigan/Work/GitHub/pyZELDA/'
if path not in sys.path:
    sys.path.append(path)
import pyzelda.zelda as zelda


data_path = os.path.join(path, 'data/')

clear_pupil_files = ['SPHERE_GEN_CLEAR_PUPIL_1', 'SPHERE_GEN_CLEAR_PUPIL_2']
zelda_pupil_files = ['SPHERE_GEN_ZELDA_PUPIL_1', 'SPHERE_GEN_ZELDA_PUPIL_2']
dark_file = 'SPHERE_GEN_BACKGROUND'

wave = 1.642e-6

clear_pupil, zelda_pupil, center = zelda.read_files(data_path, clear_pupil_files, zelda_pupil_files, dark_file, dim=390)

opd_map = zelda.analyze(clear_pupil, zelda_pupil, wave=wave, pupil_diameter=384)


fig = plt.figure(0, figsize=(16, 5))
plt.clf()
ax = fig.add_subplot(131)
ax.imshow(clear_pupil, aspect='equal', vmin=0, vmax=15000)
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
