# -*- coding: utf-8 -*-
'''
ZELDA sequences analysis module

This module is dedicated to the analysis of ZELDA sequences acquired
with VLT/SPHERE. It is not directly applicable to other sensors but
could easily be modified or dupplicated for this purpose.

arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

import numpy as np
import glob
import pyzelda.zelda as zelda
import pandas as pd
import os
import astropy.coordinates as coord
import astropy.units as units
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import numpy.fft as fft
import logging as log
import multiprocessing as mp
import ctypes

from astropy.io import fits
from astropy.time import Time, TimeDelta
from scipy.stats import pearsonr 

import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike


def parallatic_angle(ha, dec, geolat):
    '''
    Parallactic angle of a source in degrees

    Parameters
    ----------
    ha : array_like
        Hour angle, in hours

    dec : float
        Declination, in degrees

    geolat : float
        Observatory declination, in degrees

    Returns
    -------
    pa : array_like
        Parallactic angle values
    '''
    pa = -np.arctan2(-np.sin(ha),
                     np.cos(dec) * np.tan(geolat) - np.sin(dec) * np.cos(ha))

    if (dec >= geolat):
        pa[ha < 0] += 360*units.degree
    
    return np.degrees(pa)


def sort_files(root):
    '''Sort the raw files of a ZELDA sequence

    Results are saved in 2 csv files containing the information
    about all the files and the individual frames.

    Parameters
    ----------
    root : str
        Root directory where the data is stored

    Returns
    -------
    info_files : DataFrame
        Data frame with information on all files

    info_frames : DataFrame
        Data frame with information on all frames of all files

    '''
    
    # find files
    files = sorted(glob.glob(os.path.join(root, 'raw', '*.fits')))

    #
    # files information
    #
    print('Raw files information')
    info_files = pd.DataFrame({'file': files})
    for idx, file in enumerate(files):
        print(' * {0} ({1}/{2})'.format(os.path.basename(file), idx+1, len(files)))
        hdu = fits.open(file)
        hdr = hdu[0].header

        # create data frame
        info_files.loc[info_files.index[idx], 'file']     = os.path.splitext(os.path.basename(file))[0]
        info_files.loc[info_files.index[idx], 'source']   = hdr.get('HIERARCH ESO INS4 LAMP1 ST', default=False)
        info_files.loc[info_files.index[idx], 'nd_cal']   = hdr['HIERARCH ESO INS4 FILT1 NAME']
        info_files.loc[info_files.index[idx], 'nd_cpi']   = hdr['HIERARCH ESO INS4 FILT2 NAME']
        info_files.loc[info_files.index[idx], 'coro']     = hdr['HIERARCH ESO INS4 OPTI11 NAME']
        info_files.loc[info_files.index[idx], 'filt']     = hdr['HIERARCH ESO INS1 FILT NAME']
        info_files.loc[info_files.index[idx], 'DIT']      = hdr['HIERARCH ESO DET SEQ1 DIT']
        info_files.loc[info_files.index[idx], 'NDIT']     = hdr['HIERARCH ESO DET NDIT']
        info_files.loc[info_files.index[idx], 'drot_beg'] = hdr['HIERARCH ESO INS4 DROT2 BEGIN']
        info_files.loc[info_files.index[idx], 'drot_end'] = hdr['HIERARCH ESO INS4 DROT2 END']

    # file types
    info_files.loc[np.logical_not(info_files.source), 'type'] = 'B'
    info_files.loc[info_files.source & (info_files.coro == 'ZELDA'), 'type'] = 'Z'
    info_files.loc[info_files.source & (info_files.coro == ''), 'type'] = 'R'

    # save
    if not os.path.exists(os.path.join(root, 'products')):
        os.mkdir(os.path.join(root, 'products'))
    info_files.to_csv(os.path.join(root, 'products', 'info_files.csv'))

    #
    # ZELDA frames information
    #
    nframes = int(info_files.loc[info_files.type == 'Z', 'NDIT'].sum())
    columns = ('file', 'img', 'nd_cal', 'nd_cpi', 'coro', 'filt', 'DIT',
               'time', 'time_start', 'time_end',
               'drot', 'lst', 'ha', 'pa')
    info_frames = pd.DataFrame(index=range(0, nframes), columns=columns)

    index = 0
    for idx, row in info_files.loc[info_files.type == 'Z', :].iterrows():
        hdu = fits.open(os.path.join(root, 'raw', row.file+'.fits'))
        hdr = hdu[0].header

        # RA/DEC
        ra_drot = hdr['HIERARCH ESO INS4 DROT2 RA']
        ra_drot_h = np.floor(ra_drot/1e4)
        ra_drot_m = np.floor((ra_drot - ra_drot_h*1e4)/1e2)
        ra_drot_s = ra_drot - ra_drot_h*1e4 - ra_drot_m*1e2
        ra = coord.Angle((ra_drot_h, ra_drot_m, ra_drot_s), units.hour)

        dec_drot = hdr['HIERARCH ESO INS4 DROT2 DEC']
        sign = np.sign(dec_drot)
        udec_drot  = np.abs(dec_drot)
        dec_drot_d = np.floor(udec_drot/1e4)
        dec_drot_m = np.floor((udec_drot - dec_drot_d*1e4)/1e2)
        dec_drot_s = udec_drot - dec_drot_d*1e4 - dec_drot_m*1e2
        dec_drot_d *= sign
        dec = coord.Angle((dec_drot_d, dec_drot_m, dec_drot_s), units.degree)

        # observatory location
        geolon = coord.Angle(hdr.get('HIERARCH ESO TEL GEOLON', -70.4045), units.degree)
        geolat = coord.Angle(hdr.get('HIERARCH ESO TEL GEOLAT', -24.6268), units.degree)
        geoelev = hdr.get('HIERARCH ESO TEL GEOELEV', 2648.0)
        
        # timestamps
        start_time = Time(hdr['DATE-OBS'], location=(geolon, geolat, geoelev))
        end_time   = Time(hdr['DATE'], location=(geolon, geolat, geoelev))
        DIT        = TimeDelta(hdr['HIERARCH ESO DET SEQ1 DIT'], format='sec')
        NDIT       = row.NDIT
        delta      = (end_time - start_time)/NDIT

        time_beg = start_time + delta * np.arange(NDIT)
        time_mid = start_time + delta * np.arange(NDIT) + DIT/2
        time_end = start_time + delta * np.arange(NDIT) + DIT

        # other useful values
        start_drot = row.drot_beg
        end_drot   = row.drot_end
        delta      = (end_drot - start_drot)/NDIT
        drot = start_drot + delta * np.arange(NDIT)
        
        lst  = time_mid.sidereal_time('apparent')
        ha   = lst - ra
        pa   = parallatic_angle(ha, dec, geolat)

        # create data frame
        idx0 = index
        idx1 = index+NDIT-1

        info_frames.loc[idx0:idx1, 'file']       = row.file
        info_frames.loc[idx0:idx1, 'img']        = np.arange(0, NDIT, dtype=int)
        info_frames.loc[idx0:idx1, 'nd_cal']     = row.nd_cal
        info_frames.loc[idx0:idx1, 'nd_cpi']     = row.nd_cpi
        info_frames.loc[idx0:idx1, 'coro']       = row.coro
        info_frames.loc[idx0:idx1, 'filt']       = row.filt
        info_frames.loc[idx0:idx1, 'DIT']        = DIT
        info_frames.loc[idx0:idx1, 'time_start'] = time_beg
        info_frames.loc[idx0:idx1, 'time']       = time_mid
        info_frames.loc[idx0:idx1, 'time_end']   = time_end

        info_frames.loc[idx0:idx1, 'lst']        = lst.hour
        info_frames.loc[idx0:idx1, 'ha']         = ha.hour
        info_frames.loc[idx0:idx1, 'pa']         = pa
        info_frames.loc[idx0:idx1, 'drot']         = drot

        index += NDIT

    # save
    info_frames.to_csv(os.path.join(root, 'products', 'info_frames.csv'))
    
    #
    # CLEAR frames information
    #
    nframes = int(info_files.loc[info_files.type == 'R', 'NDIT'].sum())
    columns = ('file', 'img', 'nd_cal', 'nd_cpi', 'coro', 'filt', 'DIT',
               'drot', 'time', 'time_start', 'time_end')
    info_frames = pd.DataFrame(index=range(0, nframes), columns=columns)

    index = 0
    for idx, row in info_files.loc[info_files.type == 'R', :].iterrows():
        hdu = fits.open(os.path.join(root, 'raw', row.file+'.fits'))
        hdr = hdu[0].header

        # RA/DEC
        ra_drot = hdr['HIERARCH ESO INS4 DROT2 RA']
        ra_drot_h = np.floor(ra_drot/1e4)
        ra_drot_m = np.floor((ra_drot - ra_drot_h*1e4)/1e2)
        ra_drot_s = ra_drot - ra_drot_h*1e4 - ra_drot_m*1e2
        ra = coord.Angle((ra_drot_h, ra_drot_m, ra_drot_s), units.hour)

        dec_drot = hdr['HIERARCH ESO INS4 DROT2 DEC']
        sign = np.sign(dec_drot)
        udec_drot  = np.abs(dec_drot)
        dec_drot_d = np.floor(udec_drot/1e4)
        dec_drot_m = np.floor((udec_drot - dec_drot_d*1e4)/1e2)
        dec_drot_s = udec_drot - dec_drot_d*1e4 - dec_drot_m*1e2
        dec_drot_d *= sign
        dec = coord.Angle((dec_drot_d, dec_drot_m, dec_drot_s), units.degree)

        # observatory location
        geolon = coord.Angle(hdr.get('HIERARCH ESO TEL GEOLON', -70.4045), units.degree)
        geolat = coord.Angle(hdr.get('HIERARCH ESO TEL GEOLAT', -24.6268), units.degree)
        geoelev = hdr.get('HIERARCH ESO TEL GEOELEV', 2648.0)
        
        # timestamps
        start_time = Time(hdr['DATE-OBS'], location=(geolon, geolat, geoelev))
        end_time   = Time(hdr['DATE'], location=(geolon, geolat, geoelev))
        DIT        = TimeDelta(hdr['HIERARCH ESO DET SEQ1 DIT'], format='sec')
        NDIT       = row.NDIT
        delta      = (end_time - start_time)/NDIT

        time_beg = start_time + delta * np.arange(NDIT)
        time_mid = start_time + delta * np.arange(NDIT) + DIT/2
        time_end = start_time + delta * np.arange(NDIT) + DIT

        # other useful values
        start_drot = row.drot_beg
        end_drot   = row.drot_end
        delta      = (end_drot - start_drot)/NDIT
        drot = start_drot + delta * np.arange(NDIT)
        
        lst  = time_mid.sidereal_time('apparent')
        ha   = lst - ra
        pa   = parallatic_angle(ha, dec, geolat)

        # create data frame
        idx0 = index
        idx1 = index+NDIT-1

        info_frames.loc[idx0:idx1, 'file']       = row.file
        info_frames.loc[idx0:idx1, 'img']        = np.arange(0, NDIT, dtype=int)
        info_frames.loc[idx0:idx1, 'nd_cal']     = row.nd_cal
        info_frames.loc[idx0:idx1, 'nd_cpi']     = row.nd_cpi
        info_frames.loc[idx0:idx1, 'coro']       = row.coro
        info_frames.loc[idx0:idx1, 'filt']       = row.filt
        info_frames.loc[idx0:idx1, 'DIT']        = DIT
        info_frames.loc[idx0:idx1, 'time_start'] = time_beg
        info_frames.loc[idx0:idx1, 'time']       = time_mid
        info_frames.loc[idx0:idx1, 'time_end']   = time_end

        info_frames.loc[idx0:idx1, 'lst']        = lst.hour
        info_frames.loc[idx0:idx1, 'ha']         = ha.hour
        info_frames.loc[idx0:idx1, 'pa']         = pa
        info_frames.loc[idx0:idx1, 'drot']         = drot

        index += NDIT

    # save
    info_frames.to_csv(os.path.join(root, 'products', 'info_frames_ref.csv'))


def read_info(root):
    '''Read the files and frames info from disk

    Parameters
    ----------
    root : str
        Root directory where the data is stored

    Returns
    -------
    info_files : DataFrame    
        Data frame with information on all files.

    info_frames : DataFrame
        Data frame with information on all frames of all files.

    info_frames_ref : DataFrame (optional)
        Data frame with information on reference frames of all files.
     
    '''

    # read files info
    path = os.path.join(root, 'products', 'info_files.csv')
    if not os.path.exists(path):
        raise ValueError('info_files.csv does not exist in {0}'.format(root))    
    info_files = pd.read_csv(path, index_col=0)
        
    # read files info
    path = os.path.join(root, 'products', 'info_frames.csv')
    if not os.path.exists(path):
        raise ValueError('info_frames.csv does not exist in {0}'.format(root))
    info_frames = pd.read_csv(path, index_col=0)

    # reference frames info
    path = os.path.join(root, 'products', 'info_frames_ref.csv')
    if not os.path.exists(path):
        raise ValueError('info_frames_ref.csv does not exist in {0}'.format(root))    
    info_frames_ref = pd.read_csv(path, index_col=0)
    
    return info_files, info_frames, info_frames_ref
    

def process(root, sequence_type='temporal'):
    '''Process a complete sequence of ZELDA data

    The processing centers the data and performs the ZELDA analysis to
    obtain a sequence of OPD maps.

    Parameters
    ----------
    root : str
        Root directory where the data is stored

    sequence_type : str
        Type of sequence. The possible values are temporal, derotator
        or telescope. The processing of the data will be different
        depending on the type of the sequence. Default is temporal

    '''

    # read info
    info_files, info_frames, info_frames_ref = read_info(root)
    
    # list of files
    clear_pupil_files = info_files.loc[info_files['type'] == 'R', 'file'].values.tolist()
    zelda_pupil_files = info_files.loc[info_files['type'] == 'Z', 'file'].values.tolist()
    dark_files = info_files.loc[info_files['type'] == 'B', 'file'].values.tolist()

    if not os.path.exists(os.path.join(root, 'processed')):
        os.mkdir(os.path.join(root, 'processed'))

    # create sensor
    z = zelda.Sensor('SPHERE-IRDIS')
        
    # read and analyse
    print('ZELDA analysis')
    if sequence_type == 'temporal':
        for f in range(len(zelda_pupil_files)):
            print(' * {0} ({1}/{2})'.format(zelda_pupil_files[f], f+1, len(zelda_pupil_files)))

            # read data
            clear_pupil, zelda_pupils, center = z.read_files(os.path.join(root, 'raw/'), clear_pupil_files,
                                                             zelda_pupil_files[f], dark_files,
                                                             collapse_clear=True, collapse_zelda=False)

            # analyse
            opd_cube = z.analyze(clear_pupil, zelda_pupils, wave=1.642e-6)

            fits.writeto(os.path.join(root, 'processed', zelda_pupil_files[f]+'_opd.fits'), opd_cube, overwrite=True)

            del opd_cube
    elif sequence_type == 'telescope':
        # determine common center
        clear_pupil, zelda_pupils, center = z.read_files(os.path.join(root, 'raw/'), clear_pupil_files[0],
                                                         zelda_pupil_files[0], dark_files,
                                                         collapse_clear=True, collapse_zelda=True)

        # find closest match in derotator orientation (in fact hour angle) for each ZELDA pupil image
        for idx, row in info_frames.iterrows():
            ref = info_frames_ref.loc[(info_frames_ref.ha-row.ha).idxmin(), :]

            info_frames.loc[idx, 'file_ref'] = ref.file
            info_frames.loc[idx, 'img_ref']  = ref.img

        sci = None
        for f in range(len(zelda_pupil_files)):
            print(' * {0} ({1}/{2})'.format(zelda_pupil_files[f], f+1, len(zelda_pupil_files)))

            # read ZELDA pupils
            if sci != zelda_pupil_files[f]:
                sci = zelda_pupil_files[f]

                print('  ==> reading ZELDA pupils {}'.format(sci))
                cp, zelda_pupils, c = z.read_files(os.path.join(root, 'raw/'), clear_pupil_files[0],
                                                   zelda_pupil_files[f], dark_files,
                                                   collapse_clear=True, collapse_zelda=False,
                                                   center=center)

            # read CLEAR pupils
            opd_cube = np.zeros(zelda_pupils.shape)
            ref = None
            for idx, row in info_frames.loc[info_frames.file == zelda_pupil_files[f], :].iterrows():
                file_ref = row.file_ref
                img_ref  = int(row.img_ref)
                img      = int(row.img)
                
                if ref != file_ref:
                    ref = file_ref

                    print('  ==> reading CLEAR pupils {}'.format(ref))
                    clear_pupil, zp, c = z.read_files(os.path.join(root, 'raw/'), file_ref,
                                                      zelda_pupil_files[f], dark_files,
                                                      collapse_clear=False, collapse_zelda=False,
                                                      center=center)
            
                # analyse
                opd_cube[img] = z.analyze(clear_pupil[img_ref], zelda_pupils[img], wave=1.642e-6)
            
            fits.writeto(os.path.join(root, 'processed', zelda_pupil_files[f]+'_opd.fits'), opd_cube, overwrite=True)
            del opd_cube
    elif sequence_type == 'derotator':
        pass
    else:
        raise ValueError('Unknown sequence type {}'.format(sequence_type))
    print()

    # merge all cubes    
    print('Merging cubes')
    zelda_files = info_files[info_files['type'] == 'Z']
    nframe = int(zelda_files['NDIT'].sum())

    data = fits.getdata(os.path.join(root, 'processed', zelda_pupil_files[0]+'_opd.fits'))
    dim  = data.shape[-1]    
    opd_cube = np.empty((nframe, dim, dim))
    idx = 0
    for f in range(len(zelda_pupil_files)):
        print(' * {0} ({1}/{2})'.format(zelda_pupil_files[f], f+1, len(zelda_pupil_files)))
        data = fits.getdata(os.path.join(root, 'processed', zelda_pupil_files[f]+'_opd.fits'))

        if data.ndim == 2:
            ndit = 1
        else:
            ndit = data.shape[0]
        
        opd_cube[idx:idx+ndit] = data

        idx += ndit
        del data
        
    # save
    fits.writeto(os.path.join(root, 'products', 'opd_cube.fits'), opd_cube, overwrite=True)
    

def plot(root, nimg=0):
    '''Plot individual OPD maps of a full sequence

    The function plots both the individual OPD maps of the sequence and
    the maps where the mean of the sequence has been subtracted. The
    sequence can then be combined into a movie with tools like ffmpeg,
    e.g.:

    ffmpeg -f image2 -r 10 -i opd_map_%04d.png -preset medium -crf 10 -pix_fmt yuv420p -y opd_sequence.mp4
    
    Parameters
    ----------
    root : str
        Path to the working directory

    nimg : int    
        Maximum number of images to plot. Default is 0 for all images

    '''

    print('Plot full sequence')
    
    # read info
    info_files, info_frames = read_info(root)

    # read data
    data = fits.getdata(os.path.join(root, 'products', 'opd_cube.fits'))
    ndit = data.shape[0]
    dim  = data.shape[-1]

    if nimg > 0:
        ndit = min(ndit, nimg)

    data[data == 0] = np.nan
    mean = np.mean(data, axis=0)

    # ts = np.load(path+'timestamps.npy').astype(np.float)
    info_frames['time_start'] = pd.to_datetime(info_frames['time_start'], utc=True)
    info_frames['time'] = pd.to_datetime(info_frames['time'], utc=True)
    info_frames['time_end'] = pd.to_datetime(info_frames['time_end'], utc=True)
    ts = info_frames['time'].values

    # display
    cmap0 = plt.cm.PuOr_r
    cmap1 = plt.cm.RdYlBu_r
    norm0 = colors.Normalize(vmin=-100, vmax=100)
    norm1 = colors.Normalize(vmin=-10, vmax=10)

    # final directory
    path = os.path.join(root, 'products', 'images')
    if not os.path.exists(path):
        os.mkdir(path)
    
    # loop on images
    ext = 10
    for i in range(ndit):
        if (np.mod(i, 100) == 0):
            print(' * image {0} / {1}'.format(i, ndit))
        
        cdata = data[i]

        fig = plt.figure(0, figsize=(12, 5), dpi=100)
        plt.clf()
    
        ax = fig.add_subplot(121)
        cax = ax.imshow(cdata, cmap=cmap0, norm=norm0, interpolation='none', origin=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0-ext, dim+ext)
        ax.set_ylim(0-ext, dim+ext)
        ax.set_title('img #{0:04d}'.format(i))
        ax.axis('off')
        fig.colorbar(cax, label='OPD [nm]', orientation='vertical', pad=0.05, shrink=0.93)

        # timestamp
        cts = (ts[i]-ts[0]).astype(np.float)/1e9/60
        ax.text(0, 0, 't = {0:>6.2f} min'.format(cts), horizontalalignment='left', size=14, transform=ax.transAxes)
    
        ax = fig.add_subplot(122)
        cax = ax.imshow(cdata-mean, cmap=cmap1, norm=norm1, interpolation='none', origin=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0-ext, dim+ext)
        ax.set_ylim(0-ext, dim+ext)
        ax.set_title('img #{0:04d} - mean'.format(i))
        ax.axis('off')
        fig.colorbar(cax, label='OPD [nm]', orientation='vertical', pad=0.05, shrink=0.93)

        plt.tight_layout()

        plt.subplots_adjust(wspace=0.1, left=0.02, right=0.99, bottom=0.02, top=0.98)
    
        plt.savefig(os.path.join(path, 'opd_map_{0:04d}.png'.format(i)))

    # delete data
    del data
    

def stat(root, data, pupil_mask=None, suffix=''):
    '''Compute statistics on a sequence of OPD maps

    Save the statistics in a CSV file
    
    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    pupil_mask : array    
        Binary mask to hide parts of the pupil in the OPD
        maps. Default is None

    suffix : str
        Suffix for file names
    
    '''

    print('Compute statistics')

    nimg = data.shape[0]    

    # read info
    info_files, info_frames = read_info(root)
    
    # pupil mask
    if pupil_mask is None:
        mask = (data[0] != 0)
    else:
        mask = (pupil_mask != 0)

    # compute statistics
    for i in range(nimg):
        if (i % 100) == 0:
            print(' * image {0}/{1}'.format(i, len(data)))
            
        img = data[i]
        img = img[mask]

        info_frames.loc[i, 'min']    = np.min(img)
        info_frames.loc[i, 'max']    = np.max(img)
        info_frames.loc[i, 'std']    = np.std(img)
        info_frames.loc[i, 'mean']   = np.mean(img)
        info_frames.loc[i, 'median'] = np.median(img)

    # save
    if suffix != '':
        suffix = '_'+suffix

    info_frames.to_csv(os.path.join(root, 'products', 'cube_statistics{:s}.csv'.format(suffix)))


def subtract_mean_opd(root, data, nimg=0, filename=None):
    '''Subtract a mean OPD calculated over the nimg first OPD of the
    sequence

    The function returns the data cube after subtraction of the mean OPD
    and optionally saves it on disk.
    
    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    nimg : int
        Number of images over which to average the OPD

    filename : str
        Name of the file to save result

    Returns
    -------
    data : array
        Data cube after subtraction of the mean OPD
    '''
    
    print('Subtract mean OPD over {0} images'.format(nimg))
    
    if nimg == 0:
        static_opd = data.mean(axis=0)
    else:
        static_opd = np.mean(data[0:nimg], axis=0)
        
    for i, img in enumerate(data):
        if (i % 100) == 0:
            print(' * image {0}/{1}'.format(i, len(data)))
        
        img -= static_opd

    # save
    if filename is not None:
        fits.writeto(os.path.join(root, 'products', filename+'.fits'), data, overwrite=True)
    
    return data

    
def sliding_mean(root, data, nimg=10, filename=None):
    '''Compute the sliding mean of a sequence over nimg

    The function returns the sliding mean cube and optionally saves it
    on disk.
    
    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    nimg : int    
        Number of images over which to calculate the sliding
        mean. This number should be oven. Default value is 10

    filename : str
        Name of the file to save result

    Returns
    -------
    data : array
        Sliding mean OPD cube
    '''
    
    print('Compute sliding mean over {0} images'.format(nimg))
    
    nopd = data.shape[0]
    Dpup = data.shape[-1]
    
    sliding_opd = np.empty((nopd, Dpup, Dpup))
    for i in range(nopd):
        if (i % 100) == 0:
            print(' * image {0}/{1}'.format(i, nopd))
        
        i_min = i - nimg//2
        i_max = i + nimg//2

        # edge cases
        if (i_min < 0):
            i_min = 0
            
        if (i_max >= nopd):
            i_max = nopd
            
        sliding_opd[i] = np.mean(data[i_min:i_max], axis=0)

    # save
    if filename is not None:
        fits.writeto(os.path.join(root, 'products', filename+'.fits'), sliding_opd, overwrite=True)

    return sliding_opd


def compute_psd(root, data, freq_cutoff=40, return_fft=False, pupil_mask=None, filename=None):
    '''Compute individual PSDs of a sequence of OPD maps

    The function returns PSD of individual OPD maps and optionally
    their FFT. The cubes can also be saved on disk using the base name
    provided in the filename parameter.

    For the FFT the data is saved as 2xN cube where the first axis
    represents the real part of the FFT and the second axis represents
    the imaginary part of the FFT (FITS format does not support
    complex numbers!)

    The PSDs are directly calibrated in (nm/(cycle/pupil))**2 so that
    they can be easily integrated between bounds. The normalization
    takes into account the geometry of the pupil defined by
    pupil_mask.
    
    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    freq_cutoff : float
        Cutoff frequency of the PSD in cycle/pupil. Default is 40

    return_fft : bool
        Compute and save the FFT of individual OPD maps. Default
        is False

    pupil_mask : array
        Binary mask to hide parts of the pupil in the OPD
        maps. Default is None

    filename : str
        Base name of the files to save result. The _opd and _fft
        suffixes will be added to the base name

    Returns
    -------
    psd_cube : array
        PSD cubes of the OPD sequence

    fft_cube : array (optional)
        FFT cubes of the OPD sequence
    '''

    print('Compute PSDs')
    
    nimg     = data.shape[0]    
    Dpup     = data.shape[-1]
    sampling = 2**(np.ceil(np.log(2*Dpup)/np.log(2))) / Dpup
    dim_psd  = int(2*freq_cutoff*sampling)
    
    psd_cube = np.empty((nimg, dim_psd, dim_psd))
    if return_fft:
        fft_cube = np.empty((nimg, dim_psd, dim_psd), dtype=np.complex)
    for i in range(nimg):
        if (i % 100) == 0:
            print(' * opd map {0}/{1}'.format(i, nimg))

        # OPD map
        opd = data[i]

        # remove piston
        if pupil_mask is not None:
            idx = (pupil_mask != 0)
            opd[idx] -= opd[idx].mean()

        # compute PSD
        fft_opd = ztools.compute_fft_opd(opd, mask=pupil_mask, freq_cutoff=freq_cutoff)        
        psd_cube[i] = np.abs(fft_opd)**2
        
        if return_fft:
            fft_cube[i].real = fft_opd.real
            fft_cube[i].imag = fft_opd.imag

    # save
    if filename is not None:
        fits.writeto(os.path.join(root, 'products', filename+'_psd.fits'), psd_cube, overwrite=True)
        
        if return_fft:
            fits.writeto(os.path.join(root, 'products', filename+'_fft.fits'), fft_cube, overwrite=True)

    if return_fft:
        return psd_cube, fft_cube
    else:
        return psd_cube


def integrate_psd(root, psd, freq_cutoff=40, filename=None, silent=True):
    '''Integrate the PSDs

    PSDs are integrated up to a given spatial frequency cutoff, in
    steps of 1 cycle/pupil. The function returns the integrated PSD
    value and the different bounds.

    Parameters
    ----------
    root : str
        Path to the working directory

    psd : str
        PSD cube
    
    freq_cutoff : float
        Cutoff frequency of the PSD in cycle/pupil. Default is 40
    
    filename : str
        Base name of the files to save result. The _int and _bnd
        suffixes will be added to the base name for the integrated
        values and the bounds respectively.

    silent : bool
        Print some outputs. Default is True

    Returns
    -------
    psd_sigma, freq_bounds : vectors
        Integrated PSD values and bounds for the integration
    '''

    print('Integrate PSDs')
    
    nimg = psd.shape[0]
    dim  = psd.shape[-1]
    
    nbounds = freq_cutoff
    freq_bounds = np.zeros((nbounds, 2))
    psd_sigma = np.zeros((nbounds, nimg))
    for f in range(freq_cutoff):
        freq_min = f
        freq_max = f+1
        
        if not silent: 
            print(' * bounds: {0} ==> {1}'.format(freq_min, freq_max))

        freq_bounds[f, 0] = freq_min
        freq_bounds[f, 1] = freq_max
        
        freq_min_pix = freq_min*dim/(2*freq_cutoff) 
        freq_max_pix = freq_max*dim/(2*freq_cutoff)

        if freq_min == 0:
            disc = aperture.disc(dim, freq_max_pix, diameter=False)
        else:
            disc = aperture.disc(dim, freq_max_pix, diameter=False) \
                    - aperture.disc(dim, freq_min_pix, diameter=False)
        
        for i in range(nimg):
            psd_2d = psd[i]
            psd_sigma[f, i] = np.sqrt(psd_2d[disc == 1].sum())

    # save
    if filename is not None:
        dtype = np.dtype([('BOUNDS', 'f4', freq_bounds.shape), ('PSD', 'f4', psd_sigma.shape)])
        rec = np.array([np.rec.array((freq_bounds, psd_sigma), dtype=dtype)])
        fits.writeto(os.path.join(root, 'products', filename+'_psd.fits'), rec, overwrite=True)

    return psd_sigma, freq_bounds


def psd_temporal_statistics(psd_sigma, bounds, CI=[0.99, 0.95, 0.68]):
    '''Compute temporal statistics of an integrated PSD sequence

    Parameters
    ----------
    psd_sigma : array
        Integrated PSD sequence

    bounds : array
        Bounds of the PSD integration

    CI : array
        List of confidence intervals

    Returns
    -------
    psd_sigma_med : array
        Median of the integrated PSD sequence between each bounds

    psd_sigma_lim : array    
        Limits of the integrated PSD sequence in the provide
        confidence intervals
    '''

    # confidence intervals
    CI = np.array(CI)

    # lengths
    nci = len(CI)
    nval = len(psd_sigma[0])
    nbounds = len(bounds)
    
    psd_sigma_med = np.zeros(nbounds)
    psd_sigma_lim = np.zeros((nci, 2, nbounds))
    for b in range(nbounds):
        values = np.sort(psd_sigma[b])

        cmed = np.median(values)
        psd_sigma_med[b] = cmed

        for c in range(len(CI)):
            imean = np.argmin(np.abs(values - cmed))
            cmin  = values[int(imean - nval*CI[c]/2)]
            cmax  = values[int(imean + nval*CI[c]/2)]

            psd_sigma_lim[c, 0, b] = cmin
            psd_sigma_lim[c, 1, b] = cmax

    return psd_sigma_med, psd_sigma_lim


def zernike_projection(root, data, nzernike=32, reconstruct=False, pupil_mask=None, filename=None):
    '''Project a sequence of OPD maps on Zernike polynomials

    The function returns the basis and the value of the projection
    coefficients for all OPD maps in the sequence. If reconstruct is
    True, the function also returns the reconstructed OPD maps.
    
    The projection takes into account the geometry of the pupil
    defined by pupil_mask.
    
    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    nzernike : int
        Number of Zernike modes to use for the projection

    reconstruct : bool    
        Reconstruct the OPD from the Zernike coefficients and save the
        resulting cube
    
    pupil_mask : array    
        Binary mask to hide parts of the pupil in the OPD
        maps. Default is None

    filename : str
        Base name of the files to save result. The _bas, _val and _syn
        suffixes will be added to the base name for the basis, the
        coefficients and the synthetic OPD maps respectively.

    Returns
    -------
    basis : array
        Zernike polynomials basis

    zcoeff : array
        Zernike coefficients of the projection of each OPD map on the basis

    synthetic_opd : array (optional)
        Reconstructed sequence of OPD maps

    '''
    
    nimg = data.shape[0]
    Dpup = data.shape[-1]

    # pupil mask
    if pupil_mask is None:
        mask = (data[0] != 0)
    else:
        mask = (pupil_mask != 0)
    
    # get Zernike basis
    rho, theta = aperture.coordinates(data.shape[-1], Dpup/2, cpix=True, strict=False, outside=0)
    basis = zernike.arbitrary_basis(mask, nterms=nzernike, rho=rho, theta=theta)
    basis = np.nan_to_num(basis)    

    print('Project on Zernike basis')
    
    nbasis = np.reshape(basis, (nzernike, -1))
    data   = np.reshape(data, (nimg, -1))
    mask   = mask.flatten()
    data[:, mask == 0] = 0
    zcoeff = (nbasis @ data.T) / mask.sum()
    
    # save
    if filename is not None:    
        fits.writeto(os.path.join(root, 'products', filename+'_bas.fits'), basis, overwrite=True)
        fits.writeto(os.path.join(root, 'products', filename+'_val.fits'), zcoeff, overwrite=True)

    # reconstruct the projected OPD maps
    if reconstruct:
        print('Reconstruct synthetic OPD maps')
        
        synthetic_opd = (zcoeff.T @ nbasis).reshape(nimg, Dpup, Dpup)
        
        # save
        if filename is not None:
            fits.writeto(os.path.join(root, 'products', filename+'_syn.fits'), synthetic_opd, overwrite=True)

    if reconstruct:
        return basis, zcoeff, synthetic_opd
    else:
        return basis, zcoeff
    
    
def fft_filter(root, data, freq_cutoff=40, lowpass=True, window='hann', filename=None):
    '''High-pass or low-pass filtering of a sequence of OPD maps

    Filtering is done in the Fourier space using a Hann window.
    
    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    freq_cutoff : float
        Cutoff frequency of the PSD in cycle/pupil. Default is 40
    
    lowpass : bool    
        Apply a low-pass filter or a high-pass filter. Default is
        True, i.e. apply a low-pass filter.

    window : bool
        Filtering window type. Possible valeus are Hann and rect.
        Default is Hann
    
    filename : str
        Name of the file to save result

    Returns
    -------
    data_filtered : array
        Filtered sequence of OPD maps
    '''

    nimg = data.shape[0]
    Dpup = data.shape[-1]
    
    # filtering window
    M = freq_cutoff
    xx, yy = np.meshgrid(np.arange(2*M)-M, np.arange(2*M)-M)
    rr = M + np.sqrt(xx**2 + yy**2)
    if window.lower() == 'rect':
        window = np.ones((2*M, 2*M))
    elif window.lower() == 'hann':
        window = 0.5 - 0.5*np.cos(2*np.pi*rr / (2*M-1))
    window[rr >= 2*M] = 0
    window = np.pad(window, (Dpup-2*M)//2, mode='constant', constant_values=0)

    # pupil values
    mask = (data[0] != 0)
    
    # highpass or lowpass filter
    bandpass = 'low'
    if not lowpass:
        bandpass = 'high'
        window = 1-window

    print('Apply {0}-pass filter'.format(bandpass))
        
    # filter images
    data_filtered = np.empty((nimg, Dpup, Dpup))
    for i in range(nimg):
        if (i % 100) == 0:
            print(' * opd map {0}/{1}'.format(i, nimg))
        
        opd = data[i]
        opd_fft = fft.fftshift(fft.fft2(fft.fftshift(opd)))
        opd_filtered = fft.fftshift(fft.ifft2(fft.fftshift(opd_fft*window)))
        opd_filtered = opd_filtered.real
        opd_filtered *= mask
        
        data_filtered[i] = opd_filtered

    # save
    if filename is not None:
        fits.writeto(os.path.join(root, 'products', filename+'.fits'), data_filtered, overwrite=True)

    return data_filtered
    
    
def matrix_correlation_pearson(root, data, pupil_mask=None, filename=None):
    '''Computes opd-to-opd correlation using Pearson coefficient

    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    pupil_mask : array    
        Binary mask to hide parts of the pupil in the OPD
        maps. Default is None

    filename : str
        Base name of the file to save result. The _prs suffix will be
        added to the base name.

    Returns
    -------
    matrix : array
        Pearson coefficient correlation matrix

    '''

    print('Compute OPD-to-OPD Pearson correlation coefficient')
    
    nimg = data.shape[0]    

    # pupil mask
    if pupil_mask is None:
        mask = (data[0] != 0)
    else:
        mask = (pupil_mask != 0)
    
    # compute correlation matrix
    t0 = time.time()
    
    matrix_prs = np.full((nimg, nimg), np.nan)
    for i in range(nimg):
        # time calculation
        t = time.time()
        delta_t = (t - t0)/((i+1)**2/2)/60
        time_left = (nimg**2/2 - (i+1)**2/2)*delta_t
        
        print(' * i={0}, time left={1:.2f} min'.format(i, time_left))
        for j in range(i+1):
            img0 = data[i][mask]
            img1 = data[j][mask]

            coeff, p = pearsonr(img0, img1)
            
            matrix_prs[i, j] = coeff
    
    #save
    if filename is not None:
        fits.writeto(os.path.join(root, 'products', filename+'_prs.fits'), matrix_prs, overwrite=True)

    return matrix_prs


def matrix_difference(root, data, pupil_mask=None, filename=None):
    '''Extract statistics from opd-to-opd differences

    Parameters
    ----------
    root : str
        Path to the working directory

    data : str
        OPD maps cube

    pupil_mask : array    
        Binary mask to hide parts of the pupil in the OPD
        maps. Default is None

    filename : str
        Base name of the files to save result. The _ptv and _std
        suffixes will be added to the base name for the PtV and
        standard deviation matrices respectively

    Returns
    -------
    matrix_diff_ptv, matrix_diff_std : array
        PtV and standard deviation correlation matrices

    '''
    
    print('Compute statistics on OPD-to-OPD differences')
    
    nimg = data.shape[0]    

    # pupil mask
    if pupil_mask is None:
        mask = (data[0] != 0)
    else:
        mask = (pupil_mask != 0)
    
    # compute matrices
    t0 = time.time()
    
    matrix_diff_ptv = np.full((nimg, nimg), np.nan)
    matrix_diff_std = np.full((nimg, nimg), np.nan)
    for i in range(nimg):
        # time calculation
        t = time.time()
        delta_t = (t - t0)/((i+1)**2/2)/60
        time_left = (nimg**2/2 - (i+1)**2/2)*delta_t
        
        print(' * i={0}, time left={1:.2f} min'.format(i, time_left))
        for j in range(i+1):
            img = data[i] - data[j]
            img = img[mask]
            
            matrix_diff_ptv[i, j] = img.max() - img.min()
            matrix_diff_std[i, j] = img.std()
    
    #save
    if filename is not None:
        fits.writeto(os.path.join(root, 'products', filename+'_ptv.fits'), matrix_diff_ptv, overwrite=True)
        fits.writeto(os.path.join(root, 'products', filename+'_std.fits'), matrix_diff_std, overwrite=True)

    return matrix_diff_ptv, matrix_diff_std

def array_to_numpy(shared_array, shape, dtype):
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array, dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array

def matrix_tpool_init(matrix_data_i, matrix_shape_i):
    global matrix_data, matrix_shape

    matrix_data  = matrix_data_i
    matrix_shape = matrix_shape_i

def matrix_tpool_process(diag):
    global matrix_data, matrix_shape

    matrix = array_to_numpy(matrix_data, matrix_shape, np.float)
    nimg   = matrix.shape[-1]

    mask = np.eye(nimg, k=-diag, dtype=np.bool)
    mean = matrix[mask].mean()
    std  = matrix[mask].std()

    return diag, mean, std

def matrix_process(root, matrix, ncpu=1):
    '''Process a correlation matrix

    The processing computes the average and standard deviation of the
    matrix values along all the diagonals to extract statistics at
    different time scales

    Parameters
    ----------
    root : str
        Path to the working directory

    matrix : str
        Correlation matrix to be processed

    ncpu : int
        Number of CPUs to use. Default is 1

    Returns
    -------
    vec_mean : array
        Average of matrix values along all diagonals

    vec_std : array
        Standard deviation of matrix values along all diagonals
    '''

    print('Process matrix')
    
    nimg = matrix.shape[-1]

    matrix_data  = mp.RawArray(ctypes.c_double, matrix.size)
    matrix_shape = matrix.shape
    matrix_np    = array_to_numpy(matrix_data, matrix_shape, np.float)
    matrix_np[:] = matrix

    pool = mp.Pool(processes=ncpu, initializer=matrix_tpool_init, 
                   initargs=(matrix_data, matrix_shape))
    tasks = []
    for i in range(nimg):
        tasks.append(pool.apply_async(matrix_tpool_process, args=(i, )))

    pool.close()
    pool.join()

    vec_mean = np.zeros(nimg)
    vec_std  = np.zeros(nimg)
    for task in tasks:
        idx, mean, std = task.get()
        vec_mean[idx] = mean
        vec_std[idx]  = std
    del tasks

    return vec_mean, vec_std


def subtract_internal_turbulence(root=None, turb_sliding_mean=30, method='zernike',
                                 nzern=80, filter_cutoff=40, pupil_mask=None,
                                 turbulence_residuals=False,
                                 psd_compute=True, psd_cutoff=40,
                                 ncpa_sliding_mean=10, save_intermediate=False,
                                 save_product=False, save_ncpa=True, test_mode=True):
    '''Implements the subtraction of the internal turbulence in a long
    OPD sequence

    The subtract_turbulence() method estimates the contribution of the
    internal turbulence in a sequence, subtracts it to the data and
    calculates the final quasi-static NCPA variations. The procedure
    is the following:
      1. Compute a sliding mean of the OPD sequence over a given time
         interval (turb_sliding_mean)
      2. Subtract the sliding mean to the OPD sequence to isolate the
         turbulence
      3. Project the individual turbulence images on Zernike
         polynomials (nzern)
      4. Reconstruct the synthetic turbulence based on the projection
      5. **Optional**: calculate residuals of the turbulence
         (turbulence - reconstructed_turbulence) and compute their PSD
      6. Subtract reconstructed turbulence to the original OPD sequence
      7. Compute the PSD of the final sequence without turbulence
      8. Subtract a sliding mean of ncpa_sliding_mean images to the 
         final sequence to measure the quasi-static NCPA
      9. Compute the PSD of the quasi-static NCPA
    
    Parameters
    ----------
    root : str
        Path to the working directory
    
    turb_sliding_mean : int
        Number of images over which the OPD maps will be averaged to
        compute the sliding mean. Should be even. Default is 30
        
    method : str
        Method that will be used to estimate and subtract the turbulence.
        Possible values are zernike or fft. Default is zernike

    nzern: int
        Number of Zernike modes to use for the projection of the
        turbulence. Defaut is 80.
    
    filter_cutoff : float
        Spatial frequency used for the high-pass FFT filter when 
        method='fft'. Default is 40.

    pupil_mask : array
        Mask defining the pupil.

    turbulence_residuals : bool 
        Compute the turbulence residuals and related statistics. 
        Default is False
    
    psd_compute : bool
        Perform all PSD computations. Can be disabled to save time.
        Default is True.

    psd_cutoff : float    
        Spatial frequency cutoff for the calculation of the turbulence
        residuals PSD. Default is 40

    ncpa_sliding_mean : int
        Number of images over which the OPD maps will be averaged to
        compute the sliding mean used for the final NCPA estimation.
        Should be even. Default is 10
        
    
    save_intermediate : bool
        Save all intermediate data products. Default is False

    save_product : bool
        Save the OPD after turbulence subtraction. Default is False

    save_ncpa : bool
        Save final quasi-static NCPA cube after turbulence subtraction.
        Default is False.

    test_mode : bool
        If True, limits the number of frames in the data to 100. Default is True
    '''

    log.info('Start turbulence subtraction')
    
    if method.lower() == 'zernike':
        suffix = 'method={:s}_smean={:03d}_nzern={:03d}'.format(method, turb_sliding_mean, nzern)
    elif method.lower() == 'fft':
        suffix = 'method={:s}_smean={:03d}_fcutoff={:.1f}'.format(method, turb_sliding_mean, filter_cutoff)
    else:
        raise ValueError('Unknown subtraction method {0}'.format(method))

    # root
    if root is None:
        raise ValueError('root must contain the path to the data!')
        
    # read data
    log.info('Read data')
    data = fits.getdata(root / 'products' / 'opd_cube.fits')
    if test_mode:
        data = data[0:100]

    # pupil mask
    if pupil_mask is None:
        pupil_mask = (data[0] != 0)
    else:
        # hide values outside of the pupil
        log.info('Hide values outside of the pupil')
        for i in range(len(data)):
            data[i] = data[i]*pupil_mask
    
    # sliding mean over avg_time sec
    log.info('Compute sliding mean')
    data_sliding_mean = sliding_mean(root, data, nimg=turb_sliding_mean)
    
    # subtract sliding mean to isolate turbulence
    log.info('Subtract sliding mean')
    turb = data - data_sliding_mean
    
    # free memory
    del data_sliding_mean

    if save_intermediate:
        fits.writeto(root / 'products' / 'sequence_turbulence_{:s}.fits'.format(suffix), turb, overwrite=True)
    
    # compute PSD of turbulence
    if psd_compute:
        log.info('Compute PSD of turbulence')    
        psd_cube = compute_psd(root, turb, freq_cutoff=psd_cutoff, pupil_mask=pupil_mask, return_fft=False)

        # integrate PSD of turbulence
        psd_int, psd_bnds = integrate_psd(root, psd_cube, freq_cutoff=psd_cutoff)

        # save as FITS table
        dtype = np.dtype([('BOUNDS', 'f4', psd_bnds.shape), ('PSD', 'f4', psd_int.shape)])
        rec = np.array([np.rec.array((psd_bnds, psd_int), dtype=dtype)])        
        fits.writeto(root / 'products' / 'sequence_turbulence_{:s}_psd.fits'.format(suffix), rec, overwrite=True)

        # free memory
        del psd_cube
    
    # fit turbulence with Zernikes
    if method.lower() == 'zernike':
        log.info('Fit turbulence with Zernike')
        basis, zern_coeff, turb_reconstructed = zernike_projection(root, turb, nzernike=nzern,
                                                                        reconstruct=True, pupil_mask=pupil_mask)

        # free memory
        del basis
    elif method.lower() == 'fft':
        log.info('Fit turbulence with Fourier filtering')
        # first remove some Zernike modes
        basis, zern_coeff, turb_lf = zernike_projection(root, turb, nzernike=nzern,
                                                             reconstruct=True, pupil_mask=pupil_mask)
        turb_hf = turb - turb_lf
        turb_hf_filtered = ztools.fourier_filter(turb_hf, freq_cutoff=filter_cutoff, lowpass=True,
                                                 window='rect', mask=pupil_mask)

        # reconstructed turbulence
        turb_reconstructed = turb_lf + turb_hf_filtered

        # free memory
        del basis

    if save_intermediate:
        fits.writeto(root / 'products' / 'sequence_reconstructed_turbulence_{:s}.fits'.format(suffix), turb_reconstructed, overwrite=True)
    
    # compute PSD of reconstructed turbulence
    if psd_compute:
        log.info('Compute PSD of reconstructed turbulence')
        psd_cube = compute_psd(root, turb_reconstructed, freq_cutoff=psd_cutoff, pupil_mask=pupil_mask, return_fft=False)

        # integrate PSD of residuals
        psd_int, psd_bnds = integrate_psd(root, psd_cube, freq_cutoff=psd_cutoff)

        # save as FITS table
        dtype = np.dtype([('BOUNDS', 'f4', psd_bnds.shape), ('PSD', 'f4', psd_int.shape)])
        rec = np.array([np.rec.array((psd_bnds, psd_int), dtype=dtype)])        
        fits.writeto(root / 'products' / 'sequence_reconstructed_turbulence_{:s}_psd.fits'.format(suffix), rec, overwrite=True)

        # free memory
        del psd_cube
      
    # compute turbulence residuals
    if turbulence_residuals:
        # subtract reconstructued turbulence
        log.info('Compute turbulence residuals')
        turb_residuals = turb - turb_reconstructed

        if save_intermediate:
            fits.writeto(root / 'products' / 'sequence_turbulence_residuals_{:s}.fits'.format(suffix), turb_residuals, overwrite=True)
        
        # compute PSD of residuals
        if psd_compute:
            log.info('Compute PSD of turbulence residuals')
            psd_cube = compute_psd(root, turb_residuals, freq_cutoff=psd_cutoff, pupil_mask=pupil_mask, return_fft=False)

            # free memory 
            del turb_residuals

            # integrate PSD of residuals
            psd_int, psd_bnds = integrate_psd(root, psd_cube, freq_cutoff=psd_cutoff)

            # save as FITS table
            dtype = np.dtype([('BOUNDS', 'f4', psd_bnds.shape), ('PSD', 'f4', psd_int.shape)])
            rec = np.array([np.rec.array((psd_bnds, psd_int), dtype=dtype)])        
            fits.writeto(root / 'products' / 'sequence_turbulence_residuals_{:s}_psd.fits'.format(suffix), rec, overwrite=True)

            # free memory
            del psd_cube
    
    # free memory
    del turb

    # subtract reconstructed turbulence to original data
    log.info('Subtract reconstructed turbulence to data')
    data_no_turb = data - turb_reconstructed

    # save
    if save_product:
        fits.writeto(root / 'products' / 'sequence_data_cube_no_turbulence_{:s}.fits'.format(suffix), data_no_turb, overwrite=True)

    # free memory
    del data
    del turb_reconstructed
    
    # compute PSD of the final sequence
    if psd_compute:
        log.info('Compute PSD of data without turbulence')
        psd_cube = compute_psd(root, data_no_turb, freq_cutoff=psd_cutoff, pupil_mask=pupil_mask, return_fft=False)

        # integrate PSD of residuals
        psd_int, psd_bnds = integrate_psd(root, psd_cube, freq_cutoff=psd_cutoff)

        # save as FITS table
        dtype = np.dtype([('BOUNDS', 'f4', psd_bnds.shape), ('PSD', 'f4', psd_int.shape)])
        rec = np.array([np.rec.array((psd_bnds, psd_int), dtype=dtype)])
        fits.writeto(root / 'products' / 'sequence_data_cube_no_turbulence_{:s}_psd.fits'.format(suffix), rec, overwrite=True)

        # free memory
        del psd_cube

    # NCPA estimation
    log.info('Compute final NCPA')
    ncpa_cube = subtract_mean_opd(root, data_no_turb, nimg=ncpa_sliding_mean)

    if save_ncpa:
        fits.writeto(root / 'products' / 'sequence_ncpa_cube_{:s}.fits'.format(suffix), ncpa_cube, overwrite=True)
    
    # compute PSD of the final sequence
    if psd_compute:
        log.info('Compute PSD of final NCPA')
        psd_cube = compute_psd(root, ncpa_cube, freq_cutoff=psd_cutoff, pupil_mask=pupil_mask, return_fft=False)

        # integrate PSD of residuals
        psd_int, psd_bnds = integrate_psd(root, psd_cube, freq_cutoff=psd_cutoff)

        # save as FITS table
        dtype = np.dtype([('BOUNDS', 'f4', psd_bnds.shape), ('PSD', 'f4', psd_int.shape)])
        rec = np.array([np.rec.array((psd_bnds, psd_int), dtype=dtype)])
        fits.writeto(root / 'products' / 'sequence_ncpa_cube_{:s}_psd.fits'.format(suffix), rec, overwrite=True)

        # free memory
        del psd_cube
    
    print()
    log.info('Finished!')
    print('Finished!')
    print()
