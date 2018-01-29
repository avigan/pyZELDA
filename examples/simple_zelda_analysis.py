# compatibility with python 2.7
from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import numpy as np

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture

path = '/Users/mndiaye/Dropbox/python/zelda/pyZELDA/'
# path = '/Users/avigan/Work/GitHub/pyZELDA/'
# path = 'D:/Programmes/GitHub/pyZELDA/'

data_path = os.path.join(path, 'data/')

wave = 1.642e-6

clear_pupil_files = ['SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3', 'SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3']
zelda_pupil_files = ['SPHERE_ZELDA_PUPIL_CUBE1_NDIT=3', 'SPHERE_ZELDA_PUPIL_CUBE2_NDIT=3']
dark_file = 'SPHERE_BACKGROUND'

z = zelda.Sensor('SPHERE-IRDIS')

clear_pupil, zelda_pupil, center = z.read_files(data_path, clear_pupil_files, zelda_pupil_files, dark_file,
                                                collapse_clear=True, collapse_zelda=True)

opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave)
opd_map = opd_map.squeeze()+10000
print('{0}'.format(opd_map.mean()))

Dpup=opd_map.shape[-1]
pupil_mask = aperture.sphere_pupil(Dpup, Dpup, dead_actuator_diameter=0.025, spiders=False)

print('std dev from map: {0} nm rms'.format(opd_map[pupil_mask != 0].std()))

freq_cutoff = 100
freq_min1    = 0
freq_max1    = 50
freq_min2    = 50
freq_max2    = 100
psd_2d, psd_1d, freq = ztools.compute_psd(opd_map, mask=pupil_mask, freq_cutoff=freq_cutoff)
sigma1  = ztools.integrate_psd(psd_2d, freq_cutoff, freq_min1, freq_max1)
sigma2  = ztools.integrate_psd(psd_2d, freq_cutoff, freq_min2, freq_max2)

#print('std dev 1 from psd: {0} nm rms'.format(sigma1))
#print('std dev 2 from psd: {0} nm rms'.format(sigma2))

sigma = np.sqrt(sigma1**2+sigma2**2)
print('std dev from psd: {0} nm rms'.format(sigma))

# basis, coeff, opd_zern = ztools.zernike_expand(opd_map.mean(axis=0), 20)


# fig = plt.figure(0, figsize=(16, 4))
# plt.clf()
# ax = fig.add_subplot(141)
# ax.imshow(clear_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000)
# ax.set_title('Clear pupil')
# 
# ax = fig.add_subplot(142)
# ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=15000)
# ax.set_title('ZELDA pupil')
# 
# ax = fig.add_subplot(143)
# ax.imshow(opd_map.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma')
# ax.set_title('OPD map')
# 
# ax = fig.add_subplot(144)
# cax = ax.imshow(opd_zern.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma')
# ax.set_title('Zernike projected OPD map')
# 
# cbar = fig.colorbar(cax)
# cbar.set_label('OPD [nm]')
# 
# plt.tight_layout()
# plt.show()
# 
