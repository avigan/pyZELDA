# -*- coding: utf-8 -*-

'''
pyZELDA main module

arthur.vigan@lam.fr
'''

import numpy as np

import utils.mft as mft
import utils.imutils as imutils
import utils.aperture as aperture

import poppy.zernike as zernike
import scipy.ndimage as ndimage

from astropy.io import fits


def read_files(path, clear_pupil_file, zelda_pupil_file, dark_file, dim=500, center=(), center_method='fit'):
    '''
    Read the ZELDA files from disk and prepare them for analysis
    
    Parameters
    ----------
    path : str
        Path to the directory that contains the TIFF files
    
    clear_pupil_file : str
        Name of the file that contains the clear pupil data, without the .fits
    
    zelda_pupil_file : str
        Name of the file that contains the ZELDA pupil data, without the .fits
    
    dark_file : str
        Name of the file that contains the dark data, without the .fits
    
    dim : int, optional
        Size of the final array. Default is 500
    
    center : tuple, optional
        Specify the center of the pupil in raw data coordinations.
        Default is '()', i.e. the center will be determined by the routine
    
    center_method : str, optional
        Method to be used for finding the center of the pupil:
         - 'fit': least squares circle fit (default)
         - 'com': center of mass
    
    Returns
    -------
    clear_pupil : array_like
        Array containing the clear pupil data
    
    zelda_pupil : array_like
        Array containing the zelda pupil data
    
    c : vector_like
        Vector containing the (x,y) coordinates of the center in 1024x1024 raw data format
    '''
    
    # read data
    clear_pupil = fits.getdata(path+clear_pupil_file+'.fits')
    if clear_pupil.ndim == 3:
        clear_pupil = clear_pupil.mean(axis=0)
    clear_pupil = clear_pupil[:, 1024:]
    
    zelda_pupil = fits.getdata(path+zelda_pupil_file+'.fits')
    if zelda_pupil.ndim == 3:
        zelda_pupil = zelda_pupil.mean(axis=0)
    zelda_pupil = zelda_pupil[:, 1024:]
    
    dark = fits.getdata(path+dark_file+'.fits')
    if dark.ndim == 3:
        dark = dark.mean(axis=0)
    dark = dark[:, 1024:]
    
    # subtract background and correct for bad pixels
    clear_pupil = clear_pupil - dark
    clear_pupil = imutils.sigma_clip(clear_pupil, box=5, nsigma=3, iterate=True)
    
    zelda_pupil = zelda_pupil - dark
    zelda_pupil = imutils.sigma_clip(zelda_pupil, box=5, nsigma=3, iterate=True)

    # center
    if (len(center) == 0):
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
            
            cx, cy, R, residuals = imutils.least_square_circle(cc[0], cc[1])
            c = np.array((cx, cy))
            c = np.roll(c, 1)
        elif (center_method == 'com'):
            # center of mass (often fails)
            c = np.array(ndimage.center_of_mass(tmp))
            c = np.roll(c, 1)
        else:
            raise NameError('Unkown centring method '+center_method)

        print('Center: {0:.2f}, {1:.2f}'.format(c[0], c[1]))
    elif (len(center) == 2):
        c = np.array(center)
    else:
        raise NameError('Error, you must pass 2 values for center')
        return 0
        
    cc = dim//2
    
    clear_pupil = shift(clear_pupil, cc-c)
    clear_pupil = clear_pupil[:dim, :dim]
    
    zelda_pupil = shift(zelda_pupil, cc-c)
    zelda_pupil = zelda_pupil[:dim, :dim]

    return clear_pupil, zelda_pupil, c


def read_files_sequence(path, clear_pupil_files, zelda_pupil_files, dark_files, dim=500, center=(), center_method='fit'):
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
    
    Returns
    -------
    clear_pupil : array_like
        Array containing the collapsed clear pupil data
    
    zelda_pupil : array_like
        Array containing the zelda pupil data
    
    c : vector_like
        Vector containing the (x,y) coordinates of the center in 1024x1024 raw data format
    '''

    # read clear pupil data (collapsed)
    if type(clear_pupil_files) is not list:
        clear_pupil = fits.getdata(path+clear_pupil_files+'.fits')
        if clear_pupil.ndim == 3:
            clear_pupil = clear_pupil.mean(axis=0)
    else:
        nfiles = len(clear_pupil_files)
        clear_pupil = np.zeros((1024, 2048))
        for fname in clear_pupil_files:
            data = fits.getdata(path+fname+'.fits')
            if data.ndim == 3:
                data = data.mean(axis=0)
            clear_pupil += data / nfiles
    clear_pupil = clear_pupil[:, 1024:]

    # read dark data (collapsed)
    if type(dark_files) is not list:
        dark = fits.getdata(path+dark_files+'.fits')
        if dark.ndim == 3:
            dark = dark.mean(axis=0)
    else:
        nfiles = len(dark_files)
        dark = np.zeros((1024, 2048))
        for fname in dark_files:
            data = fits.getdata(path+fname+'.fits')
            if data.ndim == 3:
                data = data.mean(axis=0)
            dark += data / nfiles                        
    dark = dark[:, 1024:]
    
    # subtract background and correct for bad pixels
    clear_pupil = clear_pupil - dark
    clear_pupil = imutils.sigma_filter(clear_pupil, box=5, nsigma=3, iterate=True)

    # center
    if (len(center) == 0):
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
            
            cx, cy, R, residuals = imutils.least_square_circle(cc[0], cc[1])
            c = np.array((cx, cy))
            c = np.roll(c, 1)
        elif (center_method == 'com'):
            # center of mass (often fails)
            c = np.array(ndimage.center_of_mass(tmp))
            c = np.roll(c, 1)
        else:
            raise NameError('Unkown centring method '+center_method)

        print('Center: {0:.2f}, {1:.2f}'.format(c[0], c[1]))
    elif (len(center) == 2):
        c = np.array(center)
    else:
        raise NameError('Error, you must pass 2 values for center')
        return 0

    cint = c.astype(np.int)
    cc = dim//2
    
    clear_pupil = shift(clear_pupil, cc-c)
    clear_pupil = clear_pupil[:dim, :dim]
    
    # read zelda pupil data (all frames)
    if type(zelda_pupil_files) is not list:
        zelda_pupil_files = [zelda_pupil_files]

    zelda_pupils = []
    for fname in zelda_pupil_files:
        print(fname)
        
        zelda_pupil = fits.getdata(path+fname+'.fits')
        if zelda_pupil.ndim == 3:
            zelda_pupil = zelda_pupil[:, :, 1024:]
        else:
            zelda_pupil = zelda_pupil[np.newaxis, :, 1024:]        

        nframes = len(zelda_pupil)
        for idx in range(nframes):
            print(' * frame {0} / {1}'.format(idx+1, nframes))
            img = zelda_pupil[idx]
            img = img - dark

            nimg = img[cint[1]-dim//2-20:cint[1]+dim//2+20, cint[0]-dim//2-20:cint[0]+dim//2+20]
            nimg = imutils.sigma_filter(nimg, box=5, nsigma=3, iterate=True)
            
            img[cint[1]-dim//2-20:cint[1]+dim//2+20, cint[0]-dim//2-20:cint[0]+dim//2+20] = nimg

            zelda_pupil[idx] = shift(img, cc-c)

        zelda_pupil = zelda_pupil[:, :dim, :dim]

        zelda_pupils.append(zelda_pupil)
    
    return clear_pupil, zelda_pupils, c    
    

def create_reference_wave(dimtab, wave=1.642e-6, pupil_diameter=384):
    '''
    Simulate the ZELDA reference wave
    
    Parameters
    ----------
    dimtab : int
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
    # Chromaticity parameters
    # ++++++++++++++++++++++++++++++++++
    
    # design wavelength of Zernike mask
    wave0_m = 1.6255e-6
    
    # source wavelength
    wave_m = wave

    # ratio between design wavelength and source wavelength
    ratio_wave = wave0_m/wave_m

    # ++++++++++++++++++++++++++++++++++
    # Zernike mask parameters
    # ++++++++++++++++++++++++++++++++++
    
    # physical diameter and depth, in m
    d_m = 70.7e-6
    z_m = 0.8146e-6
    
    # glass refractive index @ lam=1.6255 um
    n_glass = 1.44311

    # F ratio in coronagraphic plane
    Fratio = 40
    
    # R_mask: mask radius in lam0/D unit, R_mask=0.502
    R_mask = 0.5*d_m / (wave0_m * Fratio)
 
    # OPDx: mask phase shift lam0 unit with n the refractive index of the
    # substrate at the design wavelength lam0
    OPDx = (n_glass-1)*z_m / wave0_m

    # ++++++++++++++++++++++++++++++++++
    # Dimensions in each plane
    # ++++++++++++++++++++++++++++++++++
    
    # pupil size in the entrance pupil plane
    NA = pupil_diameter
  
    # mask size in the focal plane
    NB = 300
    
    # pupil size in the relayed pupil plane
    # NC = NA
    
    # entrance pupil radius
    Rpuppix = NA/2
    
    # Lyot stop size in entrance pupil diameter
    ratio_Lyot = 1
    
    # ++++++++++++++++++++++++++++++++++
    # Numerical simulation part
    # ++++++++++++++++++++++++++++++++++

    # --------------------------------
    # plane A (Entrance pupil plane)

    # definition of m1 parameter for the Matrix Fourier Transform (MFT)
    # here equal to the mask size
    m1 = 2*R_mask

    # defintion of the electric field in plane A in the absence of aberrations
    ampl_PA_noaberr = aperture.disc(dimtab, Rpuppix, cpix=True, strict=True)
    
    # --------------------------------
    # plane B (Focal plane)

    # scaling of the parameter m1 to account for the wavelength of work
    m1bis = m1 * (dimtab/NA) * ratio_wave
  
    # calculation of the electric field in plane B with MFT within the Zernike
    # sensor mask in a NBxNB size table
    ampl_PB_noaberr = mft.mft(ampl_PA_noaberr, dimtab, NB, m1bis)
        
    # restriction of the MFT with the mask disk of diameter NB/2
    ampl_PB_noaberr = ampl_PB_noaberr * aperture.disc(NB, NB, diameter=True, cpix=True, strict=True)
      
    # expression of the field in the absence of aberrations without mask
    ampl_PC0_noaberr = ampl_PA_noaberr
  
    # normalization term
    norm_ampl_PC_noaberr = 1./np.max(np.abs(ampl_PC0_noaberr))

    # --------------------------------
    # plane C (Relayed pupil plane)
  
    # mask phase shift phi
    phi = 2*np.pi*OPDx*wave0_m/wave_m
    
    # phasor term associated  with the phase shift
    expi = np.exp(1j*phi)
      
    # --------------------------------
    # definition of parameters for the phase estimate with Zernike
    
    # b1 = reference_wave: parameter corresponding to the wave diffracted by the mask in the relayed pupil
    reference_wave = norm_ampl_PC_noaberr * mft.mft(ampl_PB_noaberr, NB, dimtab, m1bis) * \
                     aperture.disc(dimtab, Rpuppix*ratio_Lyot, cpix=True, strict=True)

    return reference_wave, expi


def analyse(clear_pupil, zelda_pupil, wave=1.642e-6, pupil_diameter=384, silent=False):
    '''
    Performs the ZELDA data analysis using the outputs provided by the read_files() function.
    
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
        
    silent : bool, optional
        Remain silent during the data analysis

    Returns
    -------
    opd : array_like
        Optical path difference map in nanometers
    '''

    # ++++++++++++++++++++++++++++++++++
    # Geometrical parameters
    # ++++++++++++++++++++++++++++++++++
    dimtab  = clear_pupil.shape[0]
    Rpuppix = pupil_diameter/2
    
    # ++++++++++++++++++++++++++++++++++
    # Reference wave
    # ++++++++++++++++++++++++++++++++++
    reference_wave, expi = create_reference_wave(dimtab, wave=wave, pupil_diameter=pupil_diameter)
    
    # ++++++++++++++++++++++++++++++++++
    # Phase reconstruction from data
    # ++++++++++++++++++++++++++++++++++
    pup = aperture.disc(dimtab, Rpuppix, mask=True, cpix=True, strict=True)

    if zelda_pupil.ndim == 2:
        #
        # 2D image
        #

        # normalization
        zelda_norm = zelda_pupil / clear_pupil
        zelda_norm[~pup] = 0

        # determinant calculation
        delta = (expi.imag)**2 - 2*(reference_wave-1) * (1-expi.real)**2 - \
                ((1-zelda_norm) / reference_wave) * (1-expi.real)
        delta = delta.real
        delta[~pup] = 0

        # check for negative values
        neg_values = ((delta < 0) & pup)
        neg_count  = neg_values.sum()
        ratio = neg_count / pup.sum() * 100

        # warning
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
        kw = 2*np.pi / wave
        opd_nm = (1/kw) * theta * 1e9

        # statistics
        if (silent is False):
            print('OPD statistics:')
            print(' * min = {0:0.2f} nm'.format(opd_nm.min()))
            print(' * max = {0:0.2f} nm'.format(opd_nm.max()))
            print(' * std = {0:0.2f} nm'.format(opd_nm.std()))        
    elif zelda_pupil.ndim == 3:
        #
        # 3D cube
        #

        print('ZELDA cube analysis')
        nframes = len(zelda_pupil)
        for idx in range(nframes):
            print(' * frame {0} / {1}'.format(idx+1, nframes))
            
            # normalization
            zelda_norm = zelda_pupil[idx] / clear_pupil
            zelda_norm[~pup] = 0
            
            # determinant calculation
            delta = (expi.imag)**2 - 2*(reference_wave-1) * (1-expi.real)**2 - \
                    ((1-zelda_norm) / reference_wave) * (1-expi.real)
            delta = delta.real
            delta[~pup] = 0

            # check for negative values
            neg_values = ((delta < 0) & pup)
            neg_count  = neg_values.sum()
            ratio = neg_count / pup.sum() * 100

            # warning
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
            kw = 2*np.pi / wave
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
    else:
        raise ValueError('zelda_pupil has a wrong number of dimensions ({0})'.format(zelda_pupil.ndim))
    
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

