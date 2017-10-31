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


psd_2d_mft, psd_1d_mft, rad_cpup_mft = ztools.compute_psd(opd, mask=mask, freq_cutoff=150.)
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
freq_cutoff = 150.

sigma1_fft = ztools.integrate_psd(psd_2d_fft, Dpup/2, freq0, freq1)
sigma2_fft = ztools.integrate_psd(psd_2d_fft, Dpup/2, freq1, freq2)
sigma3_fft = ztools.integrate_psd(psd_2d_fft, Dpup/2, freq2, freq3)
print('sigma1 fft: {0:2f}'.format(sigma1_fft))
print('sigma2 fft: {0:2f}'.format(sigma2_fft))
print('sigma3 fft: {0:2f}'.format(sigma3_fft))

sigma_from_psd_fft = np.sqrt(sigma1_fft**2+sigma2_fft**2+sigma3_fft**2)

print('total sigma fft: {0:2f}'.format(sigma_from_psd_fft))
print('total std dev: {0}'.format(np.std(opd[mask])))
print()

sigma1_mft = ztools.integrate_psd(psd_2d_mft, freq_cutoff, freq0, freq1)
sigma2_mft = ztools.integrate_psd(psd_2d_mft, freq_cutoff, freq1, freq2)
sigma3_mft = ztools.integrate_psd(psd_2d_mft, freq_cutoff, freq2, freq3)
print('sigma1 mft: {0:2f}'.format(sigma1_mft))
print('sigma2 mft: {0:2f}'.format(sigma2_mft))
print('sigma3 mft: {0:2f}'.format(sigma3_mft))

sigma_from_psd_mft = np.sqrt(sigma1_mft**2+sigma2_mft**2+sigma3_mft**2)

print('total sigma mft: {0:2f}'.format(sigma_from_psd_mft))
print('total std dev: {0}'.format(np.std(opd[mask])))


