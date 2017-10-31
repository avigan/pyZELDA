# compatibility with python 2.7
from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import numpy as np

from astropy.io import fits

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools

import pyzelda.utils as utils

path = '/Users/mndiaye/Dropbox/python/zelda/pyZELDA/'
# path = '/Users/avigan/Work/GitHub/pyZELDA/'
# path = 'D:/Programmes/GitHub/pyZELDA/'

data_path = os.path.join(path, 'data/')

opd = fits.getdata(data_path + 'opd.fits')
mask = (opd != 0)

Dpup      = opd.shape[-1]
dim       = 2**(np.ceil(np.log(2*Dpup)/np.log(2)))
pad_width = int((dim - Dpup)/2)
padded_opd = np.pad(opd, pad_width, 'constant')

psd_2d_fft, psd_1d_fft, rad_cpup_fft = ztools.compute_psd(opd, mask=mask)
integral_psd_fft = np.sum(psd_2d_fft)


psd_2d_mft, psd_1d_mft, rad_cpup_mft = ztools.compute_psd(opd, mask=mask, freq_max=40.)
integral_psd_mft = np.sum(psd_2d_mft)

print('var dev: {0}'.format(np.var(opd[mask])))
print('int psd fft: {0}'.format(integral_psd_fft))
print('int psd mft: {0}'.format(integral_psd_mft))


###########################
#plot 
plt.clf()
plt.semilogy(rad_cpup_fft, psd_1d_fft)
plt.semilogy(rad_cpup_mft, psd_1d_mft, alpha=0.5)
plt.xlim(0, 300)
plt.ylim(1e-5, 1e2)


###########################
freq0 = 0.
freq1 = 4.
freq2 = 20.
freq3 = 384
Dpup  = 384


sigma1 = ztools.integrate_psd(psd_2d_fft, Dpup, freq0, freq1)
sigma2 = ztools.integrate_psd(psd_2d_fft, Dpup, freq1, freq2)
sigma3 = ztools.integrate_psd(psd_2d_fft, Dpup, freq2, freq3)
print('{0:2f}'.format(sigma1))
print('{0:2f}'.format(sigma2))
print('{0:2f}'.format(sigma3))

sigma_from_psd = np.sqrt(sigma1**2+sigma2**2+sigma3**2)

print('{0:2f}'.format(sigma_from_psd))
print('std dev: {0}'.format(np.std(opd[mask])))


