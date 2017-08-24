# -*- coding: utf-8 -*-

'''
pyZELDA main module

arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

import os
import numpy as np

import pyzelda.utils.mft as mft
import pyzelda.utils.imutils as imutils
import pyzelda.utils.aperture as aperture
import pyzelda.utils.circle_fit as circle_fit

import poppy.zernike as zernike
import scipy.ndimage as ndimage

from astropy.io import fits


def number_of_frames(path, data_files):
    '''
    Returns the total number of frames in a sequence of files

    Parameters
    ----------
    path : str
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
        img = fits.getdata(os.path.join(path, fname+'.fits'))
        if img.ndim == 2:
            nframes_total += 1
        elif img.ndim == 3:
            nframes_total += img.shape[0]
            
    return nframes_total  


def load_data(path, data_files):
    '''
    read data from a file and check the nature of data (single frame or cube) 

    Parameters:
    ----------
    path : str
        Path to the directory that contains the FITS files
    
    data_files : str
        List of files that contains the data, without the .fits

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
    data_cube = np.zeros((nframes_total, 1024, 1024))
    frame_idx = 0
    for fname in data_files:
        data = fits.getdata(path+fname+'.fits')        
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        
        nframes = data.shape[0]
        data_cube[frame_idx:frame_idx+nframes] = data[:, :, 1024:]
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
        tmp = ndimage.binary_fill_holes(tmp, structure=kernel)

        kernel = np.ones((3, 3), dtype=int)
        tmp_flt = ndimage.binary_erosion(tmp, structure=kernel)

        diff = tmp-tmp_flt
        cc = np.where(diff != 0)

        cx, cy, R, residuals = circle_fit.least_square_circle(cc[0], cc[1])
        c = np.array((cx, cy))
        c = np.roll(c, 1)
    elif (center_method == 'com'):
        # center of mass (often fails)
        c = np.array(ndimage.center_of_mass(tmp))
        c = np.roll(c, 1)
    else:
        raise NameError('Unkown centring method '+center_method)

    print('Center: {0:.2f}, {1:.2f}'.format(c[0], c[1]))
        
    return c


def recentred_data_cubes(path, data_files, dark, dim, center, collapse):
    '''
    Read data cubes from disk and recenter them

    Parameters
    ----------
    path : str
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
    '''
    center = np.array(center)
    cint = center.astype(np.int)
    cc   = dim//2
        
    # read zelda pupil data (all frames)
    if type(data_files) is not list:
        data_files = [data_files]

    # determine total number of frames
    nframes_total = number_of_frames(path, data_files)

    ext = 5
    data_cube = np.empty((nframes_total, dim+2*ext, dim+2*ext))
    frame_idx = 0
    for fname in data_files:
        # read data
        data = fits.getdata(path+fname+'.fits')
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        nframes = data.shape[0]
        data_cube[frame_idx:frame_idx+nframes] = data[:, cint[1]-cc-ext:cint[1]+cc+ext, 1024+cint[0]-cc-ext:1024+cint[0]+cc+ext]
        frame_idx += nframes
        
        del data

    # collapse if needed
    if collapse:
        data_cube = data_cube.mean(axis=0, keepdims=True)

    # clean and recenter images
    dark_sub = dark[cint[1]-cc-ext:cint[1]+cc+ext, cint[0]-cc-ext:cint[0]+cc+ext]
    for idx, img in enumerate(data_cube):
        img = img - dark_sub

        img = imutils.sigma_filter(img, box=5, nsigma=3, iterate=True)
        img = imutils.shift(img, cint-center-ext)
        
        data_cube[idx] = img
    
    data_cube = data_cube[:, :dim, :dim]
        
    return data_cube


def read_files(path, clear_pupil_files, zelda_pupil_files, dark_files, dim=500, center=(), center_method='fit',
               collapse_clear=False, collapse_zelda=False):
    '''
    Read a sequence of ZELDA files from disk and prepare them for analysis
    
    Parameters
    ----------
    path : str
        Path to the directory that contains the TIFF files
    
    clear_pupil_files : str
        List of files that contains the clear pupil data, without the .fits
    
    zelda_pupil_files : str
        List of files that contains the ZELDA pupil data, without the .fits
    
    dark_files : str
        List of files that contains the dark data, without the .fits
    
    dim : int, optional
        Size of the final array. Default is 500
    
    center : tuple, optional
        Specify the center of the pupil in raw data coordinations.
        Default is '()', i.e. the center will be determined by the routine
    
    center_method : str, optional
        Method to be used for finding the center of the pupil:
         - 'fit': least squares circle fit (default)
         - 'com': center of mass

    collapse_clear : bool
        Collapse the clear pupil images. Default is False
    
    collapse_zelda : bool
        Collapse the zelda pupil images. Default is False
    
    Returns
    -------
    clear_pupil : array_like
        Array containing the collapsed clear pupil data
    
    zelda_pupil : array_like
        Array containing the zelda pupil data
    
    c : vector_like
        Vector containing the (x,y) coordinates of the center in 1024x1024 raw data format
    '''

    ##############################
    # Deal with files
    ##############################
    
    # read number of frames
    nframes_clear = number_of_frames(path, clear_pupil_files)	
    nframes_zelda = number_of_frames(path, zelda_pupil_files)	

    print('Clear pupil: nframes={0}, collapse={1}'.format(nframes_clear, collapse_clear))
    print('ZELDA pupil: nframes={0}, collapse={1}'.format(nframes_zelda, collapse_zelda))
    
    # make sure we have compatible data sets
    if (nframes_zelda == 1) or collapse_zelda:
        if nframes_clear != 1:
            collapse_clear = True
            print(' * automatic collapse of clear pupil to match ZELDA data')
    else:
        if (nframes_zelda != nframes_clear) and (not collapse_clear) and (nframes_clear != 1):
            raise ValueError('Incompatible number of frames between ZELDA and clear pupil. You could use collapse_clear=True.')
    
    # read dark data	
    dark = load_data(path, dark_files)
    dark = dark.mean(axis=0)

    # read clear pupil data
    clear_pupil = load_data(path, clear_pupil_files)
    
    ##############################
    # Center determination
    ##############################
    
    # collapse clear pupil image
    clear_pupil_collapse = clear_pupil.mean(axis=0, keepdims=True)

    # subtract background and correct for bad pixels
    clear_pupil_collapse -= dark
    clear_pupil_collapse = imutils.sigma_filter(clear_pupil_collapse.squeeze(), box=5, nsigma=3, iterate=True)
    
    # search for the pupil center
    if len(center) == 0:
        center = pupil_center(clear_pupil_collapse, center_method)
    elif len(center) != 2:
        raise ValueError('Error, you must pass 2 values for center')

    ##############################
    # Clean and recenter images
    ##############################
    clear_pupil = recentred_data_cubes(path, clear_pupil_files, dark, dim, center, collapse_clear)
    zelda_pupil = recentred_data_cubes(path, zelda_pupil_files, dark, dim, center, collapse_zelda)
    
    return clear_pupil, zelda_pupil, center


def refractive_index(wave, material):
    '''
    compute the refractive index of a material at a given wavelength
    database: https://refractiveindex.info/
    
    Parameters
    ----------
    
    wave: wavelength in m
    
    material: name of the material
    
    Returns
    -------
    
    n: the refractive index value using the Sellmeier formula

    '''
    # convert wave from m to um
    wave = wave*1e6 
    
    if material == 'fused_silica':
        params = {'B1': 0.6961663, 'B2': 0.4079426, 'B3': 0.8974794, 
                  'C1': 0.0684043, 'C2': 0.1162414, 'C3': 9.896161,
                  'wavemin': 0.21, 'wavemax': 3.71}
    else:
        raise ValueError('Unknown material!')
    
    if (wave > params['wavemin']) and (wave < params['wavemax']):
        n = np.sqrt(1 + params['B1']*wave**2/(wave**2-params['C1']**2) +
                params['B2']*wave**2/(wave**2-params['C2']**2) +
                params['B3']*wave**2/(wave**2-params['C3']**2))
    else:
        raise ValueError('Wavelength is out of range for the refractive index')
        
    return n


def create_reference_wave(dim, wave=1.642e-6, pupil_diameter=384, material='fused_silica'):
    '''
    Simulate the ZELDA reference wave
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    wave : float, optional
        Wavelength of the data. Default is 1.642 micron, corresponding to the
        FeII filter in SPHERE/IRDIS for which the ZELDA mask has been optimized

    pupil_diameter : int
        Diameter of the pupil, in pixels. Default is 384 (IRDIS)
            
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
    d_m = 70.7e-6
    z_m = 0.8146e-6
    
    # substrate refractive index
    n_substrate = refractive_index(wave, material)

    # F ratio in coronagraphic plane
    Fratio = 40
    
    # R_mask: mask radius in lam0/D unit
    R_mask = 0.5*d_m / (wave * Fratio)

    # ++++++++++++++++++++++++++++++++++
    # Dimensions
    # ++++++++++++++++++++++++++++++++++
    
    # mask sampling in the focal plane
    D_mask_pixels = 300
    
    # entrance pupil radius
    R_pupil_pixels = pupil_diameter/2
    
    # ++++++++++++++++++++++++++++++++++
    # Numerical simulation part
    # ++++++++++++++++++++++++++++++++++

    # --------------------------------
    # plane A (Entrance pupil plane)

    # definition of m1 parameter for the Matrix Fourier Transform (MFT)
    # here equal to the mask size
    m1 = 2*R_mask

    # defintion of the electric field in plane A in the absence of aberrations
    ampl_PA_noaberr = aperture.disc(dim, R_pupil_pixels, cpix=True, strict=True)
    
    # --------------------------------
    # plane B (Focal plane)

    # scaling of the parameter m1 to account for the wavelength of work
    m1bis = m1 * (dim/pupil_diameter)
  
    # calculation of the electric field in plane B with MFT within the Zernike
    # sensor mask
    ampl_PB_noaberr = mft.mft(ampl_PA_noaberr, dim, D_mask_pixels, m1bis)
        
    # restriction of the MFT with the mask disk of diameter D_mask_pixels/2
    ampl_PB_noaberr = ampl_PB_noaberr * aperture.disc(D_mask_pixels, D_mask_pixels, diameter=True, cpix=True, strict=True)
      
    # expression of the field in the absence of aberrations without mask
    ampl_PC0_noaberr = ampl_PA_noaberr
  
    # normalization term
    norm_ampl_PC_noaberr = 1./np.max(np.abs(ampl_PC0_noaberr))

    # --------------------------------
    # plane C (Relayed pupil plane)
  
    # mask phase shift phi (mask in transmission)
    phi = 2*np.pi*(n_substrate-1)*z_m/wave
    
    # phasor term associated  with the phase shift
    expi = np.exp(1j*phi)
      
    # --------------------------------
    # definition of parameters for the phase estimate with Zernike
    
    # b1 = reference_wave: parameter corresponding to the wave diffracted by the mask in the relayed pupil
    reference_wave = norm_ampl_PC_noaberr * mft.mft(ampl_PB_noaberr, D_mask_pixels, dim, m1bis) * \
                     aperture.disc(dim, R_pupil_pixels, cpix=True, strict=True)

    return reference_wave, expi


def analyze(clear_pupil, zelda_pupil, wave=1.642e-6, pupil_diameter=384, overwrite=False, silent=False):
    '''Performs the ZELDA data analysis using the outputs provided by the read_files() function.
    
    Parameters
    ----------
    clear_pupil : array_like
        Array containing the clear pupil data
    
    zelda_pupil : array_like
        Array containing the zelda pupil data

    wave : float, optional
        Wavelength of the data. Default is 1.642 micron, corresponding to the
        FeII filter in SPHERE/IRDIS for which the ZELDA mask has been optimized

    pupil_diameter : int
        Diameter of the pupil, in pixels. Default is 384 (IRDIS)
        
    overwrite : bool
        If set to True, the OPD maps are saved inside the zelda_pupil
        array to save memory. Otherwise, a distinct OPD array is
        returned. Do not use if you're not a ZELDA High Master :-)
    
    silent : bool, optional
        Remain silent during the data analysis

    Returns
    -------
    opd : array_like
        Optical path difference map in nanometers

    '''

    #make sure we have 3D cubes
    if clear_pupil.ndim == 2:
        clear_pupil = clear_pupil[np.newaxis, ...]

    if zelda_pupil.ndim == 2:
        zelda_pupil = zelda_pupil[np.newaxis, ...]
        
    # create a copy of the zelda pupil array if needed
    if not overwrite:
        zelda_pupil = zelda_pupil.copy()

    # make sure wave is an array
    if type(wave) is not list:
        wave = [wave]
    wave  = np.array(wave)
    nwave = wave.size
    
    # ++++++++++++++++++++++++++++++++++
    # Geometrical parameters
    # ++++++++++++++++++++++++++++++++++
    dim = clear_pupil.shape[-1]
    R_pupil_pixels = pupil_diameter/2
    
    # ++++++++++++++++++++++++++++++++++
    # Reference wave(s)
    # ++++++++++++++++++++++++++++++++++
    mask_diffraction_prop = []
    for w in wave:
        reference_wave, expi = create_reference_wave(dim, wave=w, pupil_diameter=pupil_diameter)
        mask_diffraction_prop.append((reference_wave, expi))        
    
    # ++++++++++++++++++++++++++++++++++
    # Phase reconstruction from data
    # ++++++++++++++++++++++++++++++++++
    pup = aperture.disc(dim, R_pupil_pixels, mask=True, cpix=True, strict=True)

    print('ZELDA analysis')
    nframes_clear = len(clear_pupil)
    nframes_zelda = len(zelda_pupil)

    # (nframes_clear, nframes_zelda) is either (1, N) or (N, N). (N, 1) is not allowed.
    if (nframes_clear != nframes_zelda) and (nframes_clear != 1):
        raise ValueError('Incompatible number of frames between clear and ZELDA pupil images')

    if (nwave != 1) and (nwave != nframes_zelda):
        raise ValueError('Incompatible number of wavelengths and ZELDA pupil images')
    
    for idx in range(nframes_zelda):
        print(' * frame {0} / {1}'.format(idx+1, nframes_zelda))

        # normalization
        if nframes_clear == 1:
            zelda_norm = zelda_pupil[idx] / clear_pupil
        else:
            zelda_norm = zelda_pupil[idx] / clear_pupil[idx]
        zelda_norm = zelda_norm.squeeze()
        zelda_norm[~pup] = 0

        # mask_diffraction_prop array contains the mask diffracted properties:
        #  - [0] reference wave
        #  - [1] dephasing term
        if nwave == 1:
            cwave = wave[0]
            reference_wave = mask_diffraction_prop[0][0]
            expi = mask_diffraction_prop[0][1]
        else:
            cwave = wave[idx]
            reference_wave = mask_diffraction_prop[idx][0]
            expi = mask_diffraction_prop[idx][1]
            
        # determinant calculation
        delta = (expi.imag)**2 - 2*(reference_wave-1) * (1-expi.real)**2 - \
                ((1-zelda_norm) / reference_wave) * (1-expi.real)
        delta = delta.real
        delta[~pup] = 0

        # check for negative values
        neg_values = ((delta < 0) & pup)
        neg_count  = neg_values.sum()
        ratio = neg_count / pup.sum() * 100

        if (silent is False):
            print('Negative values: {0} ({1:0.3f}%)'.format(neg_count, ratio))

        # too many nagative values
        if (ratio > 1):
            raise NameError('Too many negative values in determinant (>1%)')

        # replace negative values by 0
        delta[neg_values] = 0

        # phase calculation
        theta = (1 / (1-expi.real)) * (-expi.imag + np.sqrt(delta))
        theta[~pup] = 0

        # optical path difference in nm
        kw = 2*np.pi / cwave
        opd_nm = (1/kw) * theta * 1e9

        # statistics
        if (silent is False):
            print('OPD statistics:')
            print(' * min = {0:0.2f} nm'.format(opd_nm.min()))
            print(' * max = {0:0.2f} nm'.format(opd_nm.max()))
            print(' * std = {0:0.2f} nm'.format(opd_nm.std()))        

        # save
        zelda_pupil[idx] = opd_nm

    # variable name change
    opd_nm = zelda_pupil
    
    return opd_nm


def opd_expand(opd, nterms=32):
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
    
    NA = 384.
    Rpuppix = NA/2

    # rho, theta coordinates for the aperture
    rho, theta = aperture.coordinates(opd.shape[0], Rpuppix, cpix=True, strict=True, outside=np.nan)

    wgood = np.where(np.isfinite(rho))
    ngood = (wgood[0]).size

    wbad = np.where(np.logical_not(np.isfinite(rho)))
    rho[wbad]   = 0
    theta[wbad] = 0

    # create the Zernike polynomiales basis
    basis  = zernike.zernike_basis(nterms=nterms, rho=rho, theta=theta)

    # determines the coefficients
    coeffs = [(opd * b)[wgood].sum() / ngood for b in basis]

    # reconstruct the OPD
    reconstructed_opd = np.zeros(opd.shape, dtype=opd.dtype)
    for z in range(nterms):
        reconstructed_opd += coeffs[z] * basis[z, :, :]

    return basis, coeffs, reconstructed_opd

