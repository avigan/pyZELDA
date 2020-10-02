# -*- coding: utf-8 -*-
'''
pyZELDA utility methods
arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

import numpy as np
import scipy.ndimage as ndimage
import numpy.fft as fft

from astropy.io import fits
from pathlib import Path

import pyzelda.utils.mft as mft
import pyzelda.utils.imutils as imutils
import pyzelda.utils.aperture as aperture
import pyzelda.utils.circle_fit as circle_fit
import pyzelda.utils.zernike as zernike
import pyzelda.utils.prof as prof


def number_of_frames(path, data_files):
    '''
    Returns the total number of frames in a sequence of files
    Parameters
    ----------
    path : Path
        Path to the directory that contains the FITS files

    data_files : str
        List of files that contains the data, without the .fits
    Returns
    -------
    nframes_total : int
        Total number of frames
    '''
    if type(data_files) is not list:
        data_files = [data_files]

    nframes_total = 0
    for fname in data_files:
        img = fits.getdata(path / '{0}.fits'.format(fname))
        if img.ndim == 2:
            nframes_total += 1
        elif img.ndim == 3:
            nframes_total += img.shape[0]

    return nframes_total


def load_data(path, data_files, width, height, origin):
    '''
    read data from a file and check the nature of data (single frame or cube)
    Parameters:
    ----------
    path : Path
        Path to the directory that contains the FITS files

    data_files : str
        List of files that contains the data, without the .fits
    width : int
        Width of the detector window to be extracted

    height : int
        Height of the detector window to be extracted

    origin : tuple
        Origin point of the detector window to be extracted in the raw files

    Returns
    -------
    clear_cube : array_like
        Array containing the collapsed data
    '''

    # make sure we have a list
    if type(data_files) is not list:
        data_files = [data_files]

    # get number of frames
    nframes_total = number_of_frames(path, data_files)

    # load data
    data_cube = np.zeros((nframes_total, height, width))
    frame_idx = 0
    for fname in data_files:
        data = fits.getdata(path / '{0}.fits'.format(fname))
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        nframes = data.shape[0]
        data_cube[frame_idx:frame_idx + nframes] = data[:, origin[1]:origin[1] + height, origin[0]:origin[0] + width]
        frame_idx += nframes

    return data_cube


def pupil_center(clear_pupil, center_method):
    '''
    find the center of the clear pupil

    Parameters:
    ----------
    clear_pupil : array_like
        Array containing the collapsed clear pupil data
    center_method : str, optional
        Method to be used for finding the center of the pupil:
         - 'fit': least squares circle fit (default)
         - 'com': center of mass
    Returns
    -------
    c : vector_like
        Vector containing the (x,y) coordinates of the center in 1024x1024 raw data format
    '''

    # recenter
    tmp = clear_pupil / np.max(clear_pupil)
    tmp = (tmp >= 0.2).astype(int)

    if (center_method == 'fit'):
        # circle fit
        kernel = np.ones((10, 10), dtype=int)
        tmp = ndimage.binary_fill_holes(tmp, structure=kernel).astype(int)

        kernel = np.ones((3, 3), dtype=int)
        tmp_flt = ndimage.binary_erosion(tmp, structure=kernel).astype(int)

        diff = tmp - tmp_flt
        cc = np.where(diff != 0)

        cx, cy, R, residuals = circle_fit.least_square_circle(cc[0], cc[1])
        c = np.array((cx, cy))
        c = np.roll(c, 1)
    elif (center_method == 'com'):
        # center of mass (often fails)
        c = np.array(ndimage.center_of_mass(tmp))
        c = np.roll(c, 1)
    else:
        raise NameError('Unkown centring method ' + center_method)

    return c


def recentred_data_files(path, data_files, dark, dim, center, collapse, origin, anamorphism):
    '''
    Read data cubes from disk and recenter them
    Parameters
    ----------
    path : Path
        Path to the directory that contains the TIFF files

    data_files : str
        List of files to read, without the .fits

    dark : array_like
        Dark frame to be subtracted to all images
    dim : int, optional
        Size of the final arrays
    center : vector_like
        Center of the pupil in the images
    collapse : bool
        Collapse or not the cubes

    origin : tuple
        Origin point of the detector window to be extracted in the raw files
    anamorphism : tuple
        Pupil anamorphism. If not None, it must be a 2-elements tuple
        with the scaling to apply along the x and y
    '''
    center = np.array(center)
    cint = center.astype(np.int)
    cc = dim // 2

    # read zelda pupil data (all frames)
    if type(data_files) is not list:
        data_files = [data_files]

    # determine total number of frames
    nframes_total = number_of_frames(path, data_files)

    ext = 10
    data_cube = np.empty((nframes_total, dim + 2 * ext, dim + 2 * ext))
    frame_idx = 0
    for fname in data_files:
        # read data
        data = fits.getdata(path / '{0}.fits'.format(fname))
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        nframes = data.shape[0]
        data_cube[frame_idx:frame_idx + nframes] = data[:,
                                                   origin[1] + cint[1] - cc - ext:origin[1] + cint[1] + cc + ext,
                                                   origin[0] + cint[0] - cc - ext:origin[0] + cint[0] + cc + ext]
        frame_idx += nframes

        del data

    # collapse if needed
    if collapse:
        data_cube = data_cube.mean(axis=0, keepdims=True)

    # clean and recenter images
    dark_sub = dark[cint[1] - cc - ext:cint[1] + cc + ext, cint[0] - cc - ext:cint[0] + cc + ext]
    for idx, img in enumerate(data_cube):
        img = img - dark_sub

        img = imutils.sigma_filter(img, box=5, nsigma=3, iterate=True)
        img = imutils.shift(img, cint - center)

        if anamorphism is not None:
            img = imutils.scale(img, anamorphism, method='interp', center=(cc + ext, cc + ext))

        data_cube[idx] = img

    data_cube = data_cube[:, ext:dim + ext, ext:dim + ext]

    return data_cube


def recentred_data_cubes(cube, dim, center, collapse, origin, anamorphism):
    '''
    Recenter already loaded data cubes
    Parameters
    ----------
    cube : array
        Data cube
    dim : int, optional
        Size of the final arrays
    center : vector_like
        Center of the pupil in the images
    collapse : bool
        Collapse or not the cubes

    origin : tuple
        Origin point of the detector window to be extracted in the raw files
    anamorphism : tuple
        Pupil anamorphism. If not None, it must be a 2-elements tuple
        with the scaling to apply along the x and y
    '''
    center = np.array(center)
    cint = center.astype(np.int)
    cc = dim // 2

    # extract useful data
    ext = 10
    data_cube = cube[:,
                origin[1] + cint[1] - cc - ext:origin[1] + cint[1] + cc + ext,
                origin[0] + cint[0] - cc - ext:origin[0] + cint[0] + cc + ext]

    del cube

    # collapse if needed
    if collapse:
        data_cube = data_cube.mean(axis=0, keepdims=True)

    # clean and recenter images
    for idx, img in enumerate(data_cube):
        img = imutils.sigma_filter(img, box=5, nsigma=3, iterate=True)
        img = imutils.shift(img, cint - center)

        if anamorphism is not None:
            img = imutils.scale(img, anamorphism, method='interp', center=(cc + ext, cc + ext))

        data_cube[idx] = img

    data_cube = data_cube[:, ext:dim + ext, ext:dim + ext]

    return data_cube


def refractive_index(wave, substrate, T=293):
    '''
    Compute the refractive index of a subtrate at a given wavelength,
    using values from the refractice index database:
    https://refractiveindex.info/

    Parameters
    ----------
    wave: float
        wavelength in m

    substrate: string
        Name of the substrate

    temperature: float
        temperature in K

    Returns
    -------

    n: the refractive index value using the Sellmeier formula
    '''
    # convert wave from m to um
    wave = wave * 1e6

    if substrate == 'fused_silica':
        params = {'B1': 0.6961663, 'B2': 0.4079426, 'B3': 0.8974794,
                  'C1': 0.0684043, 'C2': 0.1162414, 'C3': 9.896161,
                  'wavemin': 0.21, 'wavemax': 3.71}
        if (wave > params['wavemin']) and (wave < params['wavemax']):
            n = np.sqrt(1 + params['B1'] * wave ** 2 / (wave ** 2 - params['C1'] ** 2) +
                        params['B2'] * wave ** 2 / (wave ** 2 - params['C2'] ** 2) +
                        params['B3'] * wave ** 2 / (wave ** 2 - params['C3'] ** 2))
        else:
            raise ValueError('Wavelength is out of range for the refractive index')

    elif substrate == 'germanium':
        # from H. H. Li et al. 1980, value at T < 293K
        #    params = {'A0': 2.5381, 'A1': 1.8260e-3, 'A2': 2.8888e-6,
        #              'wave0': 0.168, 'wavemin': 1.9, 'wavemax': 18.0}
        #    if (wave > params['wavemin']) and (wave < params['wavemax'] and (T>=100) and (T <= 1200)):
        #        eps = 15.2892 +1.4549e-3*T +3.5078e-6*T**2 -1.2071e-9*T**3
        #        if (T>=100) and (T < 293):
        #            dLoverL = 2.626e-6*(T-100) +1.463e-8*(T-100)**2 -2.221e-11*(T-100)**3
        #            #dLoverL = -0.089 +2.626e-6*(T-100) +1.463e-8*(T-100)**2 -2.221e-11*(T-100)**3
        #        else:
        #            dLoverL = 5.790e-6*(T-293) +1.768e-9*(T-293)**2 -4.562e-13*(T-293)**3
        #        L = np.exp(-3.*dLoverL)
        #        n = np.sqrt(eps + (L/wave**2)*(params['A0'] +params['A1']*T +params['A2']*T**2))
        #        print('refractive index of Germanium from Li et al. (1980): to be checked')
        # from Barnes & Piltch (1979)
        params = {'A1': -6.040e-3, 'A0': 11.05128,
                  'B1': 9.295e-3, 'B0': 4.00536,
                  'C1': -5.392e-4, 'C0': 0.599034,
                  'D1': 4.151e-4, 'D0': 0.09145,
                  'E1': 1.51408, 'E0': 3426.5,
                  'wavemin': 2.5, 'wavemax': 14, 'Tmin': 50, 'Tmax': 300}
        if (wave >= params['wavemin']) and (
                wave <= params['wavemax'] and (T >= params['Tmin']) and (T <= params['Tmax'])):
            A = params['A1'] * T + params['A0']
            B = params['B1'] * T + params['B0']
            C = params['C1'] * T + params['C0']
            D = params['D1'] * T + params['D0']
            E = params['E1'] * T + params['E0']
            # print('{0}, {1}, {2}, {3}, {4}'.format(A, B, C, D, E))
            n = np.sqrt(A + B * wave ** 2 / (wave ** 2 - C) + D * wave ** 2 / (wave ** 2 - E))
            print('refractive index of Germanium from Barnes & Piltch (1979): to be checked')
        else:
            raise ValueError('Wavelength or Temperature is out of range for the refractive index')

    else:
        raise ValueError('Unknown substrate {0}!'.format(substrate))

    return n


def create_reference_wave_beyond_pupil(mask_diameter, mask_depth, mask_substrate, mask_Fratio,
                                       pupil_diameter, pupil, wave, clear=np.array([]), 
                                       sign_mask=np.array([]), cpix=False):
    '''
    Simulate the ZELDA reference wave
    Parameters
    ----------
    mask_diameter : float
        Mask physical diameter, in m

    mask_depth : float
        Mask physical depth, in m

    mask_substrate : str
        Mask substrate
    mask_Fratio : float
        Focal ratio at the mask focal plane

    pupil_diameter : int
        Instrument pupil diameter, in pixel
    pupil : array
        Instrument pupil
    wave : float, optional
        Wavelength of the data, in m
    

    clear : array
        Clear intensity map, optional. If provided, used to estimate the input field amplitude
        and therefore the reference wave. If not provided, the analytical pupil will be used.
    sign_mask : array
        -1 and 1 analytical input, optional. Used to take into account pi-shift in the input
        field (or changes in input field amplitude, after FPM filter for example)
        If not provided, will be considered 1 everywhere.
    cpix : bool, default is False
        if True, it centers the apertures / FFTs on a single pixel, otherwise between 4 pixels
        
    Returns
    -------
    reference_wave : array_like
        Reference wave as a complex array
    expi : complex
        Phasor term associated  with the phase shift
    '''

    # ++++++++++++++++++++++++++++++++++
    # Zernike mask parameters
    # ++++++++++++++++++++++++++++++++++

    # physical diameter and depth, in m
    d_m = mask_diameter
    z_m = mask_depth

    # substrate refractive index
    n_substrate = refractive_index(wave, mask_substrate)

    # R_mask: mask radius in lam0/D unit
    R_mask = 0.5 * d_m / (wave * mask_Fratio)

    # ++++++++++++++++++++++++++++++++++
    # Dimensions
    # ++++++++++++++++++++++++++++++++++

    # array and pupil
    array_dim = pupil.shape[-1]
    pupil_radius = pupil_diameter // 2

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

    # definition of the electric field in plane A in the absence of aberrations
    # If clear and sign_mask are provided, they will be used.
    # Otherwise, the analytical pupil will be used.

    if clear.any():
        P = np.nan_to_num(np.sqrt(clear))
        if sign_mask.any():
            P = P * sign_mask
        ampl_PA_noaberr = P
    else:
        ampl_PA_noaberr = pupil

    # --------------------------------
    # plane B (Focal plane)

    # calculation of the electric field in plane B with MFT within the Zernike
    # sensor mask
    ampl_PB_noaberr = mft.mft(ampl_PA_noaberr, array_dim, D_mask_pixels, m1, cpix=cpix)

    # import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    # plt.figure(figsize=(15, 15))
    # plt.clf()
    # plt.imshow(np.abs(ampl_PB_noaberr)**2, norm=colors.LogNorm())

    # restriction of the MFT with the mask disk of diameter D_mask_pixels/2
    ampl_PB_noaberr = ampl_PB_noaberr * aperture.disc(D_mask_pixels, D_mask_pixels, diameter=True, cpix=cpix,
                                                      strict=False)

    # normalization term using the expression of the field in the absence of aberrations without mask
    # if no clear was provided. Otherwise, normalization is automatic as MFT keeps the energy.
    if clear.any():
        norm_ampl_PC_noaberr = 1
    else:
        norm_ampl_PC_noaberr = 1 / np.max(np.abs(ampl_PA_noaberr))

    # --------------------------------
    # plane C (Relayed pupil plane)

    # mask phase shift theta (mask in transmission)
    theta = 2 * np.pi * (n_substrate - 1) * z_m / wave

    # phasor term associated  with the phase shift
    expi = np.exp(1j * theta)

    # --------------------------------
    # definition of parameters for the phase estimate with Zernike

    # b1 = reference_wave: parameter corresponding to the wave diffracted by the mask in the relayed pupil
    reference_wave = norm_ampl_PC_noaberr * mft.imft(ampl_PB_noaberr, D_mask_pixels, array_dim, m1, cpix=cpix)

    return reference_wave, expi


def propagate_opd_map(opd_map, mask_diameter, mask_depth, mask_substrate, mask_Fratio,
                      pupil_diameter, pupil, wave):
    '''
    Propagate an OPD map through a ZELDA sensor
    Parameters
    ----------
    opd_map : array
        OPD map, in m
    mask_diameter : float
        Mask physical diameter, in m

    mask_depth : float
        Mask physical depth, in m

    mask_substrate : str
        Mask substrate
    mask_Fratio : float
        Focal ratio at the mask focal plane

    pupil_diameter : int
        Instrument pupil diameter, in pixel
    pupil : array
        Instrument pupil
    wave : float, optional
        Wavelength of the data, in m

    Returns
    -------
    intensity_PC : array_like
        Intensity map in the re-imaged pupil plane for a given opd_map
    '''

    # ++++++++++++++++++++++++++++++++++
    # Zernike mask parameters
    # ++++++++++++++++++++++++++++++++++

    # physical diameter and depth, in m
    d_m = mask_diameter
    z_m = mask_depth

    # substrate refractive index
    n_substrate = refractive_index(wave, mask_substrate)

    # R_mask: mask radius in lam0/D unit
    R_mask = 0.5 * d_m / (wave * mask_Fratio)

    # ++++++++++++++++++++++++++++++++++
    # Dimensions
    # ++++++++++++++++++++++++++++++++++

    # array and pupil
    array_dim = pupil.shape[-1]
    pupil_radius = pupil_diameter // 2

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
    ampl_PA = pupil * np.exp(1j * 2. * np.pi * opd_map / wave)

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

    intensity_PC = np.abs(ampl_PC) ** 2

    return intensity_PC


def create_reference_wave(mask_diameter, mask_depth, mask_substrate, mask_Fratio, pupil_diameter, pupil, wave,
                          clear=np.array([]), sign_mask=np.array([]), pupil_roi=np.array([]), cpix=False):
    '''
    Simulate the ZELDA reference wave
    Parameters
    ----------
    mask_diameter : float
        Mask physical diameter, in m.

    mask_depth : float
        Mask physical depth, in m.

    mask_substrate : str
        Mask substrate
    mask_Fratio : float
        Focal ratio at the mask focal plane

    pupil_diameter : int
        Instrument pupil diameter, in pixel.
    pupil : array
        Instrument pupil
    wave : float, optional
        Wavelength of the data, in m.

    clear : array
        Clear intensity map, optional. If provided, used to estimate the input field amplitude
        and therefore the reference wave. If not provided, the analytical pupil will be used.
    sign_mask : array
        -1 and 1 analytical input, optional. Used to take into account pi-shift in the input
        field (or changes in input field amplitude, after FPM filter for example)
        If not provided, will be considered 1 everywhere.
    pupil_roi : boolean array
        Region Of Interest (ROI) in the pupil where phase computation will be performed. Optional.
        If not provided, the pupil parameter will be used instead.
    cpix : bool, default is False
        if True, it centers the apertures / FFTs on a single pixel, otherwise between 4 pixels

    Returns
    -------
    reference_wave : array_like
        Reference wave as a complex array
    expi : complex
        Phasor term associated  with the phase shift
    '''

    # compute reference wave
    reference_wave, expi = create_reference_wave_beyond_pupil(mask_diameter, mask_depth, mask_substrate,
                                                              mask_Fratio, pupil_diameter, pupil, wave,
                                                              clear=clear, sign_mask=sign_mask, cpix=cpix)
    if pupil_roi.any():
        return reference_wave * pupil_roi, expi
    else:
        return reference_wave * pupil, expi

def zernike_expand(opd, nterms=32):
    '''
    Expand an OPD map into Zernike polynomials
    Parameters
    ----------
    opd : array_like
        OPD map in nanometers

    nterms : int, optional
        Number of polynomials used in the expension. Default is 15
    Returns
    -------
    basis : array_like
        Cube containing the array of 2D polynomials

    coeffs : vector_like
        Vector with the coefficients corresponding to each polynomial

    reconstructed_opd : array_like
        Reconstructed OPD map using the basis and determined coefficients
    '''

    print('Zernike decomposition')

    if opd.ndim == 2:
        opd = opd[np.newaxis, ...]
    nopd = opd.shape[0]

    Rpuppix = opd.shape[-1] / 2

    # rho, theta coordinates for the aperture
    rho, theta = aperture.coordinates(opd.shape[-1], Rpuppix, cpix=True, strict=False, outside=np.nan)

    wgood = np.where(np.isfinite(rho))
    ngood = (wgood[0]).size

    wbad = np.where(np.logical_not(np.isfinite(rho)))
    rho[wbad] = 0
    theta[wbad] = 0

    # create the Zernike polynomiales basis
    basis = zernike.zernike_basis(nterms=nterms, rho=rho, theta=theta, outside=0)

    coeffs = np.zeros((nopd, nterms))
    reconstructed_opd = np.zeros_like(opd)
    for i in range(nopd):
        # determines the coefficients
        coeffs_tmp = [(opd[i] * b)[wgood].sum() / ngood for b in basis]
        coeffs[i] = np.array(coeffs_tmp)

        # reconstruct the OPD
        for z in range(nterms):
            reconstructed_opd[i] += coeffs_tmp[z] * basis[z, :, :]

    return basis, coeffs, reconstructed_opd


def zelda_analytical_intensity(phi, b=0.5, theta=np.pi / 2):
    '''
    Compute the analytical expression of the zelda signal
    for a given value of b

    Parameters
    ----------

    b: float
        value of the mask diffracted wave at a given pixel
    theta: float
        value of the phase delay of the mask in rad

    phi: vector_like
        array with the phase error in rad

    Returns
    -------
    IC0: vector_like
        array with the analytical expression of the zelda signal

    IC1: vector_like
        array with the expression of the zelda signal with Taylor expansion to the 1st order

    IC2: vector_like
        array with the expression of the zelda signal with Taylor expansion to the 2nd order
    '''

    npts = phi.size

    # intensity with sinusoid, linear, and quadratic expressions
    IC0 = np.zeros((npts))
    IC1 = np.zeros((npts))
    IC2 = np.zeros((npts))

    # Normalized entrance pupil plane amplitude
    P = 1.0

    # Sinudoid intensity expression
    IC0 = P ** 2 + 2 * b ** 2 * (1 - np.cos(theta)) + 2 * P * b * (
                np.sin(phi) * np.sin(theta) - np.cos(phi) * (1 - np.cos(theta)))

    # Linear intensity expression
    IC1 = P ** 2 + 2 * b ** 2 * (1 - np.cos(theta)) + 2 * P * b * (phi * np.sin(theta) - (1 - np.cos(theta)))

    # Quadratic intensity expression
    IC2 = P ** 2 + 2 * b ** 2 * (1 - np.cos(theta)) + 2 * P * b * (
                phi * np.sin(theta) - (1 - 0.5 * phi ** 2) * (1 - np.cos(theta)))

    return IC0, IC1, IC2


def compute_fft_opd(opd, mask=None, freq_cutoff=None):
    '''
    Compute the fft of the opd normalized in physical units (nm/cycle_per_pupil)

    Parameters
    ----------
    opd : array_like
        OPD map in nanometers

    mask : array_like
        Pupil mask

    freq_cutoff : float
        Maxium spatial frequency of the psd

    Returns
    -------
    fft_opd: array_like
        Normalized fft of the opd
    '''

    Dpup = opd.shape[-1]
    dim = 2 ** (np.ceil(np.log(2 * Dpup) / np.log(2)))
    sampling = dim / Dpup

    # compute the surface of the mask pupil
    if mask is None:
        norm = np.sqrt(1 / ((Dpup ** 2) * np.pi / 4))
    else:
        opd = opd * mask
        norm = np.sqrt(1 / mask.sum())

    # compute psd with fft or mft
    if freq_cutoff is None:
        pad_width = int((dim - Dpup) / 2)
        pad_opd = np.pad(opd, pad_width, 'constant')
        fft_opd = norm * fft.fftshift(fft.fft2(fft.fftshift(pad_opd), norm='ortho'))
    else:
        fft_opd = norm * mft.mft(opd, Dpup, np.int(2*freq_cutoff*sampling), 2*freq_cutoff)

    return fft_opd


def compute_psd(opd, mask=None, freq_cutoff=None):
    '''
    Compute the power spectral density fro a given phase map

    When freq_Cutoff is specified, psd is computed with mft, using the
    same sampling as the fft that would normally be used to compute
    the psd.  This smapling makes the computation of the normalization
    factor consistent with the standard fft case.
    Parameters
    ----------
    opd : array_like
        OPD map in nanometers

    mask : array_like
        Pupil mask

    freq_cutoff : float
        Maxium spatial frequency of the psd

    Returns
    -------
    psd_2d: array_like
        PSD map

    psd_1d: vector
        Azimuthal averaged profile of the PSD map

    freq: vector
        Vector of spatial frequencies corresponding to psd_1d
    '''

    Dpup = opd.shape[-1]
    dim = 2 ** (np.ceil(np.log(2 * Dpup) / np.log(2)))
    sampling = dim / Dpup

    # remove piston
    if mask is not None:
        idx = (mask != 0)
        opd[idx] -= opd[idx].mean()

    fft_opd = compute_fft_opd(opd, mask, freq_cutoff)
    psd_2d = np.abs(fft_opd) ** 2
    psd_1d, rad = prof.mean(psd_2d)

    # compute psd with fft or mft
    if freq_cutoff is None:
        freq = rad * Dpup / dim
    else:
        freq = rad / sampling

    return psd_2d, psd_1d, freq


def integrate_psd(psd_2d, freq_cutoff, freq_min, freq_max):
    '''
    Compute the integration of the psd between two spatial frequency bounds

    Parameters
    ----------
    psd_2d: array_like
        PSD map normalized in (nm/cycle per pupil)^2

    freq_cutoff : float
        Maxium spatial frequency of the psd

    freq_min : float
        Lower bound of the spatial frequencies for integration

    freq_max : float
        Upper bound of the spatial frequencies for integration

    Returns
    -------
    sigma : float
        Integrated value of the psd in nanometers

    '''

    dim = psd_2d.shape[-1]
    freq_min_pix = freq_min * dim / (2 * freq_cutoff)
    freq_max_pix = freq_max * dim / (2 * freq_cutoff)

    if freq_min == 0:
        disc = aperture.disc(dim, freq_max_pix, diameter=False)
    else:
        disc = aperture.disc(dim, freq_max_pix, diameter=False) \
               - aperture.disc(dim, freq_min_pix, diameter=False)

    sigma = np.sqrt(psd_2d[disc == 1].sum())

    return sigma


def fourier_filter(opd, freq_cutoff=40, lowpass=True, window='hann', mask=None):
    '''
    High-pass or low-pass filtering of an OPD map
    Parameters
    ----------
    opd : array_like
        OPD map in nanometers
    freq_cutoff : float
        Cutoff frequency of the PSD in cycle/pupil. Default is 40
    lowpass : bool
        Apply a low-pass filter or a high-pass filter. Default is
        True, i.e. apply a low-pass filter.
    window : bool
        Filtering window type. Possible valeus are Hann and rect.
        Default is Hann

    mask : array_like
        Pupil mask

    Returns
    -------
    opd_filtered : array_like
        Filtered OPD
    '''

    Dpup = opd.shape[-1]

    # filtering window
    M = freq_cutoff
    xx, yy = np.meshgrid(np.arange(2 * M) - M, np.arange(2 * M) - M)
    rr = M + np.sqrt(xx ** 2 + yy ** 2)
    if window.lower() == 'rect':
        window = np.ones((2 * M, 2 * M))
    elif window.lower() == 'hann':
        window = 0.5 - 0.5 * np.cos(2 * np.pi * rr / (2 * M - 1))
    window[rr >= 2 * M] = 0
    window = np.pad(window, (Dpup - 2 * M) // 2, mode='constant', constant_values=0)

    # pupil mask
    if mask is None:
        mask = (opd != 0)

    # filter opd map
    opd_fft = fft.fftshift(fft.fft2(fft.fftshift(opd)))
    opd_filtered = fft.fftshift(fft.ifft2(fft.fftshift(opd_fft * window)))
    opd_filtered = opd_filtered.real
    opd_filtered *= mask

    return opd_filtered