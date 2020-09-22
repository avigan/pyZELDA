# -*- coding: utf-8 -*-
'''
pyZELDA main module
The pyZELDA package provides facility functions to analyse data from a
Zernike wavefront sensor, e.g. the ZELDA wavefront sensor implemented
into VLT/SPHERE
arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

# compatibility with python 2.7
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import scipy.ndimage as ndimage

from astropy.io import fits
from pathlib import Path

import pyzelda.utils.mft as mft
import pyzelda.utils.imutils as imutils
import pyzelda.utils.aperture as aperture
import pyzelda.utils.circle_fit as circle_fit
import pyzelda.ztools as ztools
import pyzelda.utils.zernike as zernike

if sys.version_info < (3, 0):
    import ConfigParser
else:
    import configparser as ConfigParser


class Sensor():
    '''
    Zernike wavefront sensor class
    '''

    ##################################################
    # Constructor
    ##################################################

    def __init__(self, instrument, **kwargs):
        '''Initialization of the Sensor class
        Parameters
        ----------
        instrument : str
            Instrument associated with the sensor
        Optional keywords
        -----------------
        It is possible to override the default parameters of the
        instrument by providing any of the following keywords. If none
        is provided, the default value is used.
        mask_Fratio : float
            Focal ratio at the mask focal plane
        mask_depth : float
            Physical depth of the ZELDA mask, in meters
        mask_diameter : float
            Physical diameter of the ZELDA mask, in meters
        mask_substrate : str
            Material of the ZELDA mask substrate
        pupil_diameter : int
            Pupil diameter on the detector, in pixels
        pupil_anamorphism : tuple
            Pupil anamorphism in the (x,y) directions. Can be None
        pupil_telescope : bool
            Use full telescope pupil
        pupil_function : str, function
            Name of function or function to generate the full
            telescope pupil
        detector_width : int
            Detector sub-window width, in pixels
        detector_height : int
            Detector sub-window height, in pixels
        detector_origin : tuple (2 int)
            Origin of the detector sub-window, in pixels
        silent : bool
            Control output. Default is False
        '''

        self._instrument = instrument

        # read configuration file
        package_directory = Path(__file__).resolve().parent
        configfile = package_directory / 'instruments' / '{0}.ini'.format(instrument)
        config = ConfigParser.ConfigParser()

        try:
            config.read(str(configfile.as_posix()))

            # mask physical parameters
            self._Fratio = kwargs.get('mask_Fratio', float(config.get('mask', 'Fratio')))
            self._mask_depth = kwargs.get('mask_depth', float(config.get('mask', 'depth')))
            self._mask_diameter = kwargs.get('mask_diameter', float(config.get('mask', 'diameter')))
            self._mask_substrate = kwargs.get('mask_substrate', config.get('mask', 'substrate'))

            # pupil parameters
            self._pupil_diameter = kwargs.get('pupil_diameter', int(config.get('pupil', 'diameter')))
            self._pupil_telescope = kwargs.get('pupil_telescope', bool(eval(config.get('pupil', 'telescope'))))
            self._pupil_anamorphism = kwargs.get('pupil_anamorphism', eval(config.get('pupil', 'anamorphism')))

            pupil_function = kwargs.get('pupil_function', config.get('pupil', 'function'))
            if callable(pupil_function):
                self._pupil_function = pupil_function
            else:
                if pupil_function in dir(aperture):
                    self._pupil_function = eval('aperture.{0}'.format(pupil_function))
                else:
                    self._pupil_function = None

            # detector sub-window parameters
            self._width = kwargs.get('detector_width', int(config.get('detector', 'width')))
            self._height = kwargs.get('detector_height', int(config.get('detector', 'height')))
            cx = int(config.get('detector', 'origin_x'))
            cy = int(config.get('detector', 'origin_y'))
            self._origin = kwargs.get('origin', (cx, cy))

            # create pupil
            if self._pupil_telescope:
                pupil_func = self._pupil_function
                if pupil_func is None:
                    raise ValueError('Pupil function is not designed for this sensor')

                self._pupil = pupil_func(self._pupil_diameter)
            else:
                self._pupil = aperture.disc(self._pupil_diameter, self._pupil_diameter // 2,
                                            mask=True, cpix=True, strict=False)

            self._silent = kwargs.get('silent', False)

        except ConfigParser.Error as e:
            raise ValueError('Error reading configuration file for instrument {0}: {1}'.format(instrument, e.message))

        # dictionary to save mask diffraction properties: speed-up the
        # analysis when multiple analyses are performed at the same
        # wavelength with the same Sensor object
        self._mask_diffraction_prop = {}

    ##################################################
    # Properties
    ##################################################

    @property
    def instrument(self):
        return self._instrument

    @property
    def mask_depth(self):
        return self._mask_depth

    @property
    def mask_diameter(self):
        return self._mask_diameter

    @property
    def mask_substrate(self):
        return self._mask_substrate

    @property
    def mask_Fratio(self):
        return self._Fratio

    @property
    def pupil_diameter(self):
        return self._pupil_diameter

    @property
    def pupil_anamorphism(self):
        return self._pupil_anamorphism

    @property
    def pupil_telescope(self):
        return self._pupil_telescope

    @property
    def pupil_function(self):
        return self._pupil_function

    @property
    def pupil(self):
        return self._pupil

    @property
    def detector_subwindow_width(self):
        return self._width

    @property
    def detector_subwindow_height(self):
        return self._height

    @property
    def detector_subwindow_origin(self):
        return self._origin

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, status):
        self._silent = bool(status)

    ##################################################
    # Methods
    ##################################################

    def read_files(self, path, clear_pupil_files, zelda_pupil_files, dark_files, center=(), center_method='fit',
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
            List of files that contains the dark data, without the .fits. If None,
            assumes that the zelda and clear pupil images are already background-subtracted
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

        # path
        path = Path(path)

        # read number of frames
        nframes_clear = ztools.number_of_frames(path, clear_pupil_files)
        nframes_zelda = ztools.number_of_frames(path, zelda_pupil_files)

        if not self.silent:
            print('Clear pupil: nframes={0}, collapse={1}'.format(nframes_clear, collapse_clear))
            print('ZELDA pupil: nframes={0}, collapse={1}'.format(nframes_zelda, collapse_zelda))

        # make sure we have compatible data sets
        if (nframes_zelda == 1) or collapse_zelda:
            if nframes_clear != 1:
                collapse_clear = True
                if not self.silent:
                    print(' * automatic collapse of clear pupil to match ZELDA data')
        else:
            if (nframes_zelda != nframes_clear) and (not collapse_clear) and (nframes_clear != 1):
                raise ValueError('Incompatible number of frames between ZELDA and clear pupil. ' +
                                 'You could use collapse_clear=True.')

        # read clear pupil data
        clear_pupil = ztools.load_data(path, clear_pupil_files, self._width, self._height, self._origin)

        # read dark data
        if dark_files:
            dark = ztools.load_data(path, dark_files, self._width, self._height, self._origin)
            dark = dark.mean(axis=0)
        else:
            dark = np.zeros((clear_pupil.shape[-2], clear_pupil.shape[-1]))

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
            center = ztools.pupil_center(clear_pupil_collapse, center_method)
        elif len(center) != 2:
            raise ValueError('Error, you must pass 2 values for center')

        ##############################
        # Clean and recenter images
        ##############################
        clear_pupil = ztools.recentred_data_files(path, clear_pupil_files, dark, self._pupil_diameter,
                                                  center, collapse_clear, self._origin, self._pupil_anamorphism)
        zelda_pupil = ztools.recentred_data_files(path, zelda_pupil_files, dark, self._pupil_diameter,
                                                  center, collapse_zelda, self._origin, self._pupil_anamorphism)

        return clear_pupil, zelda_pupil, center

    def process_cubes(self, clear_pupil, zelda_pupil, center=(), center_method='fit',
                      collapse_clear=False, collapse_zelda=False):
        '''
        Alternative to read_files(): use already loaded data cubes to
        generate the clear_pupil and zelda_pupil. The images must
        already be dark-subtracted.
        Parameters
        ----------
        clear_pupil : array
            Cube of frames that contain the clear pupil data
        zelda_pupil : str
            Cube of frames that contain the ZELDA pupil data
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
        nframes_clear = len(clear_pupil)
        nframes_zelda = len(zelda_pupil)

        if not self.silent:
            print('Clear pupil: nframes={0}, collapse={1}'.format(nframes_clear, collapse_clear))
            print('ZELDA pupil: nframes={0}, collapse={1}'.format(nframes_zelda, collapse_zelda))

        # make sure we have compatible data sets
        if (nframes_zelda == 1) or collapse_zelda:
            if nframes_clear != 1:
                collapse_clear = True
                if not self.silent:
                    print(' * automatic collapse of clear pupil to match ZELDA data')
        else:
            if (nframes_zelda != nframes_clear) and (not collapse_clear) and (nframes_clear != 1):
                raise ValueError('Incompatible number of frames between ZELDA and clear pupil. ' +
                                 'You could use collapse_clear=True.')

        # make sure we have cubes
        if clear_pupil.ndim == 2:
            clear_pupil = clear_pupil[np.newaxis, ...]

        if zelda_pupil.ndim == 2:
            zelda_pupil = zelda_pupil[np.newaxis, ...]

        ##############################
        # Center determination
        ##############################

        # collapse clear pupil image
        clear_pupil_collapse = clear_pupil.mean(axis=0, keepdims=True)

        # subtract background and correct for bad pixels
        clear_pupil_collapse = imutils.sigma_filter(clear_pupil_collapse.squeeze(), box=5, nsigma=3, iterate=True)

        # search for the pupil center
        if len(center) == 0:
            center = ztools.pupil_center(clear_pupil_collapse, center_method)
        elif len(center) != 2:
            raise ValueError('Error, you must pass 2 values for center')

        ##############################
        # Clean and recenter images
        ##############################
        clear_pupil = ztools.recentred_data_cubes(clear_pupil, self._pupil_diameter, center, collapse_clear,
                                                  self._origin, self._pupil_anamorphism)
        zelda_pupil = ztools.recentred_data_cubes(zelda_pupil, self._pupil_diameter, center, collapse_zelda,
                                                  self._origin, self._pupil_anamorphism)

        return clear_pupil, zelda_pupil, center

    def analyze(self, clear_pupil, zelda_pupil, wave, overwrite=False, ratio_limit=1, use_arbitrary_amplitude=False,
                pupil_roi=np.array([]), cpix=False, sign_mask=np.array([]), refwave_from_clear=False):

        '''Performs the ZELDA data analysis using the outputs provided by the read_files() function.
        Parameters
        ----------
        clear_pupil : array_like
            Array containing the clear pupil data
        zelda_pupil : array_like
            Array containing the zelda pupil data
        wave : float, optional
            Wavelength of the data, in m.
        overwrite : bool
            If set to True, the OPD maps are saved inside the zelda_pupil
            array to save memory. Otherwise, a distinct OPD array is
            returned. Do not use if you're not a ZELDA High Master :-)
        ratio_limit : float
            Percentage of negative pixel above which the analysis is considered as
            failed. Default value is 1%
        use_arbitrary_amplitude : boolean, default is False
            If set to True, will use the formula (12) in N'Diaye 2013 that considers the pupil amplitude
            P as arbitrary. Otherwise will use the standard approximation P = 1
        pupil_roi : boolean array
            Region Of Interest (ROI) in the pupil where phase computation will be performed. Optional.
            If not provided, the pupil parameter will be used instead.
        cpix : bool, default is False
            if True, it centers the apertures / FFTs on a single pixel, otherwise between 4 pixels
        sign_mask : array
            -1 and 1 analytical input, optional. Used to take into account pi-shift in the input
            field (or changes in input field amplitude, after FPM filter for example)
            If not provided, will be considered 1 everywhere.
        refwave_from_clear : bool, default is False
            if True, the reference wave will be calculated with the square root of the clear image.
            Otherwise, it will be computed from the self._pupil theoretical data.
        Returns
        -------
        opd : array_like
            Optical path difference map in nanometers
        '''
        # make sure we have 3D cubes
        if clear_pupil.ndim == 2:
            clear_pupil = clear_pupil[np.newaxis, ...]

        if zelda_pupil.ndim == 2:
            zelda_pupil = zelda_pupil[np.newaxis, ...]

        # Dimensions
        nframes_clear = len(clear_pupil)
        nframes_zelda = len(zelda_pupil)

        # create a copy of the zelda pupil array if needed
        if not overwrite:
            zelda_pupil = zelda_pupil.copy()

        # make sure wave is an array
        if type(wave) not in [list, np.ndarray]:
            wave = [wave]
            # TODO : strange assertion, why not ndarray ?
        if type(wave) is not np.ndarray:
            wave = np.array(wave)

        nwave = wave.size

        # ++++++++++++++++++++++++++++++++++
        # Pupil
        # ++++++++++++++++++++++++++++++++++
        pupil_diameter = self._pupil_diameter
        pupil = self._pupil

        # ++++++++++++++++++++++++++++++++++
        # Reference wave(s)
        # ++++++++++++++++++++++++++++++++++

        # The reference waves are stored in a dictionnary to speed up the execution
        # of the method analyze when using several times the same configuration.
        # The reference waves depends on the wavelength, and if it is computed
        # with the clear data, then it also depends on the clear. Therefore,
        # the dictionnary keys are a hash of the tuple (wave, bytes(clear))

        keys = self._mask_diffraction_prop.keys()

        # Creating arrays for easy access of clear and wavelength values.
        # If the only one value is given, then arrays are filled with the same
        # value nframes_zelda times.

        if nwave == 1:
            waves = wave[0] * np.ones(nframes_zelda)
        else:
            waves = wave.copy()

        # If using the calculation of reference wave from clear, then the list
        # of clears is created with real clear data

        if refwave_from_clear:

            if nframes_clear == 1:
                clear_array_for_refw = np.full((nframes_zelda, clear_pupil.shape[1],
                                                clear_pupil.shape[2]), clear_pupil[0])

            else:
                clear_array_for_refw = clear_pupil.copy()

        # If using the calculation of reference wave from theoretical pupil,
        # this array is filled with self.pupil
        else:
            clear_array_for_refw = np.full((nframes_zelda, clear_pupil.shape[1],
                                            clear_pupil.shape[2]), self.pupil)

        # Creation of reference waves not in dictionnary
        # ----------------------------------------------

        for idx, w in enumerate(waves):

            key = hash((w, bytes(clear_array_for_refw[idx])))

            if key not in keys:
                if refwave_from_clear:
                    reference_wave, expi = ztools.create_reference_wave(
                        self._mask_diameter, self._mask_depth,
                        self._mask_substrate, self._Fratio,
                        pupil_diameter, pupil, w, clear=clear_array_for_refw[idx],
                        sign_mask=sign_mask, pupil_roi=pupil_roi, cpix=cpix)
                else:
                    reference_wave, expi = ztools.create_reference_wave(
                        self._mask_diameter, self._mask_depth,
                        self._mask_substrate, self._Fratio,
                        pupil_diameter, pupil, w, clear=np.array([]),
                        sign_mask=sign_mask, pupil_roi=pupil_roi, cpix=cpix)

                self._mask_diffraction_prop[key] = (reference_wave, expi)

        # ++++++++++++++++++++++++++++++++++
        # Phase reconstruction from data
        # ++++++++++++++++++++++++++++++++++

        # boolean pupil
        pup = pupil.astype(bool)

        # (nframes_clear, nframes_zelda) is either (1, N) or (N, N). (N, 1) is not allowed.
        if (nframes_clear != nframes_zelda) and (nframes_clear != 1):
            raise ValueError('Incompatible number of frames between clear and ZELDA pupil images')

        if (nwave != 1) and (nwave != nframes_zelda):
            raise ValueError('Incompatible number of wavelengths and ZELDA pupil images')

        for idx in range(nframes_zelda):
            if not self.silent:
                print(' * frame {0} / {1}'.format(idx + 1, nframes_zelda))

            # Defining the parameters for this analyzis

            if nwave == 1:
                cwave = wave[0]
            else:
                cwave = wave[idx]

            if nframes_clear == 1:
                clear = clear_pupil[0]
            else:
                clear = clear_pupil[idx]

            if refwave_from_clear:
                key = hash((cwave, bytes(clear)))
            else:
                key = hash((cwave, bytes(self.pupil)))

            # mask_diffraction_prop array contains the mask diffracted properties:
            #  - [0] reference wave
            #  - [1] dephasing term
            reference_wave, expi = self._mask_diffraction_prop[key]

            if use_arbitrary_amplitude:

                # Uses the full formula from N'Diaye 2013
                zelda_norm = zelda_pupil[idx]
                zelda_norm[pupil_roi == 0] = 0

                ###########################
                # determinant calculation #
                ###########################

                if sign_mask.any():
                    P = np.nan_to_num(np.sqrt(clear)) * sign_mask  # P from N'Diaye 2013
                else:
                    P = np.nan_to_num(np.sqrt(clear))

                P[pupil_roi == 0] = 0

                delta = (expi.imag) ** 2 - 2 / P * (reference_wave - P) * (1 - expi.real) ** 2 - \
                        (clear - zelda_norm) * (1 - expi.real) / (P * reference_wave)

                delta = (delta.real).squeeze()
                delta = np.nan_to_num(delta)
                delta[pupil_roi == 0] = 0


            else:
                zelda_norm = zelda_pupil[idx] / clear

                zelda_norm = zelda_norm.squeeze()
                zelda_norm[~pup] = 0
                ###########################
                # determinant calculation #
                ###########################

                delta = (expi.imag) ** 2 - 2 * (reference_wave - 1) * (1 - expi.real) ** 2 - \
                        ((1 - zelda_norm) / reference_wave) * (1 - expi.real)
                delta = delta.real
                delta[~pup] = 0

            # check for negative values
            neg_values = ((delta < 0) & pup)
            neg_count = neg_values.sum()
            ratio = neg_count / pup.sum() * 100

            if not self.silent:
                print('Negative values: {0} ({1:0.3f}%)'.format(neg_count, ratio))

            # too many nagative values
            if (ratio > ratio_limit):
                raise NameError('Too many negative values in determinant (>1%)')

            # replace negative values by 0
            delta[neg_values] = 0

            # phase calculation
            theta = (1 / (1 - expi.real)) * (-expi.imag + np.sqrt(delta))
            if pupil_roi.any():
                theta[pupil_roi == 0] = 0
            else:
                theta[~pup] = 0

            # optical path difference in nm
            kw = 2 * np.pi / cwave
            opd_nm = (1 / kw) * theta  *1e9

            # remove piston
            if pupil_roi.any():
                opd_nm[pupil_roi == True] -= opd_nm[pupil_roi == True].mean()
            else:
                opd_nm[pup] -= opd_nm[pup].mean()

            # statistics
            if not self.silent:
                print('OPD statistics:')
                print(' * min = {0:0.2f} nm'.format(opd_nm[pup].min()))
                print(' * max = {0:0.2f} nm'.format(opd_nm[pup].max()))
                print(' * std = {0:0.2f} nm'.format(opd_nm[pup].std()))

                # save
            zelda_pupil[idx] = opd_nm

        # variable name change
        opd_nm = zelda_pupil.squeeze()

        return opd_nm

    def propagate_opd_map(self, opd_map, wave):
        '''
        Propagate an OPD map through a ZELDA sensor
        Parameters
        ----------
        opd_map : array
            OPD map, in meter. The pupil format must be the same as
            the one used by the sensor.
        wave : float
            Wavelength, in meter
        Returns
        -------
        zelda_signal : array
            Expected ZELDA signal, in normalized intensity
        '''

        # propagate OPD map
        zelda_signal = ztools.propagate_opd_map(opd_map, self._mask_diameter, self._mask_depth,
                                                self._mask_substrate, self._Fratio,
                                                self._pupil_diameter, self._pupil, wave)
        return zelda_signal

    def mask_phase_shift(self, wave):
        '''
        Compute the phase delay introduced by the mask at a given wavelength
        Parameters
        ----------
        wave: float
            wavelength of work, in m
        Return
        ------
        phase_shift: float
            the mask phase delay, in radians
        '''

        mask_refractive_index = ztools.refractive_index(wave, self.mask_substrate)
        phase_shift = 2. * np.pi * (mask_refractive_index - 1) * self.mask_depth / wave

        return phase_shift

    def mask_relative_size(self, wave):
        '''
        Compute the relative size of the phase mask in resolution element at wave
        Parameters
        ----------
        wave: float
            wavelength of work, in m
        Return
        ------
        mask_rel_size: float
            the mask relative size in lam/D
        '''

        mask_rel_size = self.mask_diameter / (wave * self.mask_Fratio)

        return mask_rel_size