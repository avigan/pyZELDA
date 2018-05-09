# -*- coding: utf-8 -*-
'''
ZELDA monitoring module

This module is dedicated to the daily monitoring of the VLT/SPHERE
NCPA calibration done with ZELDA. This code is not directly applicable
to other sensors but could easily be modified or dupplicated for this
purpose.

arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import datetime

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.imutils as imutils

from astropy.io import fits
from astropy.time import Time
from pathlib import Path


# fixed parameters
ncpa_n_iter = 5
nzernike    = 30
freq_cutoff = 100


def import_data(path):
    '''
    Import new ZELDA monitoring data
   
    The data should be organised using a very simple architecture. If
    the root analysis directory is path/, then the raw data should be
    in path/raw/ and the results of the analysis will be stored in
    path/products/

    Parameters
    ----------
    path : str or Path
        Root path for the analysis
    '''

    # proper path
    path = Path(path)
    
    #
    # read and sort raw data
    #
    print('Read and sort new data')
    
    files = set([file.stem for file in path.glob('raw/*.fits')])

    raw_files_db = path / 'products' / 'file_info.csv'
    if raw_files_db.exists():
        info_files = pd.read_csv(raw_files_db, index_col=0)

        # detect new files
        old_files = set(info_files.index)

        files = files - old_files
    else:
        columns = ('date', 'source', 'nd_cal', 'nd_cpi', 'coro', 'filt',
                   'DIT', 'NDIT')
        info_files = pd.DataFrame(index=files, columns=columns)

    if len(files) == 0:
        print(' ==> no new data... exiting!')
        return
        
    for idx, file in enumerate(files):
        hdr = fits.getheader(path / 'raw' / '{0}.fits'.format(file))
        
        # skip already imported data
        if file in info_files.index:
            if not info_files.loc[file].isnull()['date']:
                continue
            
        print('Importing {0}'.format(file))

        # date
        time = Time(hdr['DATE-OBS']).datetime
        date = '{:04}-{:02}-{:02}'.format(time.year, time.month, time.day)

        # source
        shutter = hdr['HIERARCH ESO INS1 OPTI1 NAME']
        if shutter == 'OPEN':
            source = True
        else:
            source = False

        # create data frame
        info_files.loc[file, 'date']   = date
        info_files.loc[file, 'source'] = source
        info_files.loc[file, 'nd_cal'] = hdr['HIERARCH ESO INS4 FILT1 NAME']
        info_files.loc[file, 'nd_cpi'] = hdr['HIERARCH ESO INS4 FILT2 NAME']
        info_files.loc[file, 'coro']   = hdr['HIERARCH ESO INS4 OPTI11 NAME']
        info_files.loc[file, 'filt']   = hdr['HIERARCH ESO INS1 FILT NAME']
        info_files.loc[file, 'DIT']    = hdr['HIERARCH ESO DET SEQ1 DIT']
        info_files.loc[file, 'NDIT']   = hdr['HIERARCH ESO DET NDIT']

    # file types
    info_files.loc[np.logical_not(info_files.source), 'type'] = 'B'
    info_files.loc[info_files.source & (info_files.coro == 'ZELDA'), 'type'] = 'Z'
    info_files.loc[info_files.source & (info_files.coro == ''), 'type'] = 'R'

    # sort
    info_files.sort_index(inplace=True)

    # save
    info_files.to_csv(raw_files_db)


    #
    # compute OPD maps
    #
    print('Compute new OPD maps')
    
    # pupil
    dim        = 384
    pupil      = aperture.disc(dim, dim, diameter=True, strict=False, cpix=True)
    pupil_full = aperture.sphere_irdis_pupil(spiders=False, dead_actuator_diameter=0.025)
    
    # create sensor
    z = zelda.Sensor('SPHERE-IRDIS')
    
    # dates
    dates  = info_files.date.unique()
    
    # OPD files info
    opd_files_db = path / 'products' / 'opd_info.csv'
    if opd_files_db.exists():
        opd_info = pd.read_csv(path / 'products' / 'opd_info.csv', index_col=[0, 1],
                               header=[0], parse_dates=True)
    else:
        # create dates and iterations
        date   = datetime.datetime.strptime(dates.min(), '%Y-%m-%d')
        dstart = datetime.date(date.year, date.month, 1)
        date   = datetime.datetime.strptime(dates.max(), '%Y-%m-%d')
        dend   = datetime.date(date.year, date.month+1, 1)
        ddelta = dend - dstart
        all_dates = [pd.Timestamp(dstart+datetime.timedelta(days=x)) for x in range(ddelta.days+1) for i in range(ncpa_n_iter)]
        all_iter  = [i for j in range(ddelta.days+1) for i in range(ncpa_n_iter)]

        # create data frame
        opd_info = pd.DataFrame(index=pd.MultiIndex.from_arrays([all_dates, all_iter],
                                                                names=['date', 'iteration']),
                                columns=('dark_file', 'clear_pupil_file',
                                         'zelda_pupil_file', 'opd_map_file'))
    
    # read and analyse data
    for date in dates:        
        # get file names
        dark_file = info_files[(info_files.date == date) & (info_files.type == 'B')].index[0]
        clear_pupil_file  = info_files[(info_files.date == date) & (info_files.type == 'R')].index[0]
        zelda_pupil_files = info_files[(info_files.date == date) & (info_files.type == 'Z')].index[0:5]

        # skip dates for which we don't have all iterations
        if len(zelda_pupil_files) < 5:
            continue
        
        # skip already imported data
        if len(opd_info.loc[date]) != 0:
            if not opd_info.loc[(date, 0)].isnull()['zelda_pupil_file']:
                continue
        
        print('*')
        print('* date:  {0}'.format(date))
        print('* dark:  {0}'.format(dark_file))
        print('* clear: {0}'.format(clear_pupil_file))
        print('* zelda: {0}, ...'.format(zelda_pupil_files[0]))
        print('*')
        print()
        
        # read files
        clear_pupil, zelda_pupils, c = z.read_files(path / 'raw', clear_pupil_file,
                                                    list(zelda_pupil_files.values),
                                                    dark_file, collapse_clear=True,
                                                    collapse_zelda=False)

        # OPD map
        opd_maps = z.analyze(clear_pupil, zelda_pupils, 1.642e-6)

        # hide center and dead actuators
        for opd in opd_maps:
            opd -= opd[pupil_full != 0].mean()
            opd *= pupil

        # save
        fname = '{0}_opd.fits'.format(date)
        fits.writeto(path / 'products' / fname, opd_maps, overwrite=True)
        
        # info
        ts = pd.Timestamp(date)
        for i in range(ncpa_n_iter):
            opd_info.loc[(ts, i), 'dark_file']        = dark_file
            opd_info.loc[(ts, i), 'clear_pupil_file'] = clear_pupil_file
            opd_info.loc[(ts, i), 'zelda_pupil_file'] = zelda_pupil_files[i]
            opd_info.loc[(ts, i), 'opd_map_file']     = fname

        print()
        print()

    # save opd info
    opd_info.to_csv(opd_files_db)


    #
    # analyze OPD maps
    #
    print('Analyse new OPD maps')

    # opd info
    opd_info = pd.read_csv(path / 'products' / 'opd_info.csv', index_col=[0, 1],
                           header=[0], parse_dates=True)
    opd_info = opd_info.dropna()
    
    # read first OPD map
    data = fits.getdata(path / 'products' / opd_info['opd_map_file'][0])
    dim  = data.shape[-1]
    
    # pupil
    pupil = aperture.sphere_irdis_pupil(spiders=False, dead_actuator_diameter=0.025)
    pupil = pupil != 0
    
    # Zernike basis    
    rho, theta = aperture.coordinates(dim, dim/2, cpix=True, strict=False, outside=0)
    basis  = zernike.arbitrary_basis(pupil, nterms=nzernike, rho=rho, theta=theta)
    basis  = np.nan_to_num(basis)
    nbasis = np.reshape(basis, (nzernike, -1))
    mask   = pupil.flatten()
    
    # OPD files info
    opd_files_db = path / 'products' / 'opd_info_full.csv'
    if not opd_files_db.exists():
        # complete data frame
        cols = ['dark_file', 'clear_pupil_file', 'zelda_pupil_file', 'opd_map_file',
                 'full_std', 'no_ttf_std']
        for t in ['full_zern']:
            cols.extend(['{0}.{1}'.format(t, z) for z in range(nzernike)])
        for t in ['full_psd', 'no_ttf_psd']:
            cols.extend(['{0}.{1}'.format(t, z) for z in range(1, 100)])

        # create data frame and fill it
        opd_info_full = pd.DataFrame(index=opd_info.index, columns=cols)
        opd_info_full['dark_file']        = opd_info['dark_file']
        opd_info_full['clear_pupil_file'] = opd_info['clear_pupil_file']
        opd_info_full['zelda_pupil_file'] = opd_info['zelda_pupil_file']
        opd_info_full['opd_map_file']     = opd_info['opd_map_file']

        # write
        opd_info_full.to_csv(opd_files_db)

        del opd_info_full
        
    # read info
    opd_info_full = pd.read_csv(opd_files_db, index_col=[0, 1], header=[0], parse_dates=True)
    
    # generate discs for PSD integration
    psd_2d, psd_1d, freq = ztools.compute_psd(data[0], mask=pupil, freq_cutoff=freq_cutoff)
    dim_psd = psd_2d.shape[0]
    psd_an = np.zeros((freq_cutoff+1, dim_psd, dim_psd), dtype=np.bool)
    for f in range(1, freq_cutoff):
        freq_min = f
        freq_max = f+1

        # IRDIS
        freq_min_pix = freq_min*dim_psd/(2*freq_cutoff) 
        freq_max_pix = freq_max*dim_psd/(2*freq_cutoff)
        psd_an[f] = aperture.disc(dim_psd, freq_max_pix, diameter=False) \
                                    - aperture.disc(dim_psd, freq_min_pix, diameter=False)
    
    # OPD maps statistics
    for index in opd_info.index:
        # skip already imported data
        if index in opd_info_full.index:
            if not opd_info_full.loc[index].isnull()['full_std']:
                continue
        
        print(' * {0} - iteration {1}'.format(index[0], index[1]))

        # append data
        if not (index in opd_info_full.index):
            opd_info_full = opd_info_full.append(opd_info.loc[index])
        
        # read data only for first iteration
        if index[1] == 0:
            opd_maps = fits.getdata(path / 'products' / opd_info_full.loc[index, 'opd_map_file'])
        
        # OPD map for current iteration
        opd = opd_maps[index[1]]
                
        # projection on Zernike basis
        data = np.reshape(opd*pupil, (1, -1))
        data[:, mask == 0] = 0
        zcoeff = (nbasis @ data.T).squeeze() / mask.sum()        

        for z in range(nzernike):
            opd_info_full.loc[index, 'full_zern.{0}'.format(z)] = zcoeff[z]
        
        # remove tip, tilt and focus
        opd_full = opd.copy()
        opd -= zcoeff[1] * basis[1]
        opd -= zcoeff[2] * basis[2]
        opd -= zcoeff[3] * basis[3]
        
        # compute PSD
        opd_info_full.loc[index, 'full_std'] = opd_full[pupil].std()
        psd_2d, psd_1d, freq = ztools.compute_psd(opd_full, mask=pupil, freq_cutoff=freq_cutoff)
        for f in range(1, freq_cutoff):
            disc = psd_an[f]
            sigma = np.sqrt(psd_2d[disc].sum())
            opd_info_full.loc[index, 'full_psd.{0}'.format(f)] = sigma
        
        opd_info_full.loc[index, 'no_ttf_std'] = opd[pupil].std()
        psd_2d, psd_1d, freq = ztools.compute_psd(opd, mask=pupil, freq_cutoff=freq_cutoff)
        for f in range(1, freq_cutoff):
            disc = psd_an[f]
            sigma = np.sqrt(psd_2d[disc].sum())
            opd_info_full.loc[index, 'no_ttf_psd.{0}'.format(f)] = sigma

    # save opd info
    opd_info_full.to_csv(opd_files_db)

    
def _plot_iterations(opd_info, series, color):
    '''
    Utility function for ploting the iterations of the NCPA correction
    '''
    
    all_dates = opd_info.index.get_level_values(0).unique()
    for date in all_dates:
        data = opd_info.loc[(date, slice(None)), series]
        time = data.index.get_level_values(0) + np.timedelta64(5, 'h') * np.arange(5)
        plt.plot_date(time, data, xdate=True, linestyle='-', marker='', color=color,
                      alpha=0.6, label='')


def plot(path=None, date=None, save=False):
    '''
    Plot monitoring data, either on multiple days or at a given date

    Parameters
    ----------
    path : str or Path
        Root path for the analysis
    '''
    
    # proper path
    path = Path(path)

    # plot setup
    matplotlib.rcParams['font.size'] = 17
    color = [color['color'] for color in list(plt.rcParams['axes.prop_cycle'])]
    
    # opd info
    opd_info = pd.read_csv(path / 'products' / 'opd_info_full.csv',
                           index_col=[0, 1], parse_dates=True)    
    all_dates = opd_info.index.get_level_values(0).unique()
    
    if date is None:
        #
        # Temporal monitoring
        #

        # compute integration
        LF = ( 1,   3)
        MF = ( 4,  19)
        HF = (20,  99)

        freqs = opd_info.loc[:, slice('no_ttf_psd.{0}'.format(LF[0]), 'no_ttf_psd.{0}'.format(LF[1]))].values
        opd_info['no_ttf_LF'] = np.sqrt(np.sum(freqs**2, axis=1))

        freqs = opd_info.loc[:, slice('no_ttf_psd.{0}'.format(MF[0]), 'no_ttf_psd.{0}'.format(MF[1]))].values
        opd_info['no_ttf_MF'] = np.sqrt(np.sum(freqs**2, axis=1))

        freqs = opd_info.loc[:, slice('no_ttf_psd.{0}'.format(HF[0]), 'no_ttf_psd.{0}'.format(HF[1]))].values
        opd_info['no_ttf_HF'] = np.sqrt(np.sum(freqs**2, axis=1))

        # plot
        plt.figure(0, figsize=(25, 11))
        plt.clf()

        series = 'full_std'
        data = opd_info.loc[(slice(None), 0), series]
        plt.plot_date(data.index.get_level_values(0), data, xdate=True, linestyle=':', marker='o', color=color[0],
                      label='Total')
        # _plot_iterations(opd_info, series, color[0])

        series = 'no_ttf_std'
        data = opd_info.loc[(slice(None), 0), series]
        plt.plot_date(data.index.get_level_values(0), data, xdate=True, linestyle='-', marker='o', color=color[1],
                      label='No tip/tilt/focus')
        _plot_iterations(opd_info, series, color[1])

        series = 'no_ttf_LF'
        data = opd_info.loc[(slice(None), 0), series]
        plt.plot_date(data.index.get_level_values(0), data, xdate=True, linestyle='-', marker='o', color=color[2],
                      label=r'LF ({0}-{1} c/p)'.format(LF[0], LF[1]+1))
        _plot_iterations(opd_info, series, color[2])

        series = 'no_ttf_MF'
        data = opd_info.loc[(slice(None), 0), series]
        plt.plot_date(data.index.get_level_values(0), data, xdate=True, linestyle='-', marker='o', color=color[3],
                      label=r'MF ({0}-{1} c/p)'.format(MF[0], MF[1]+1))
        _plot_iterations(opd_info, series, color[3])

        series = 'no_ttf_HF'
        data = opd_info.loc[(slice(None), 0), series]
        plt.plot_date(data.index.get_level_values(0), data, xdate=True, linestyle='-', marker='o', color=color[4],
                      label=r'HF ({0}-{1} c/p)'.format(HF[0], HF[1]+1))
        _plot_iterations(opd_info, series, color[4])

        # specs (without beam-shift)
        plt.axhline(np.sqrt(7.9**2 + 11.6**2 + 32.6**2), color=color[1], linestyle='--', linewidth=2)
        plt.axhline( 7.9, color=color[2], linestyle='--', linewidth=2)
        plt.axhline(11.6, color=color[3], linestyle='--', linewidth=2)
        plt.axhline(32.6, color=color[4], linestyle='--', linewidth=2)

        last  = all_dates.max() + np.timedelta64(1, 'D')
        # last  = datetime.date(2018, 1, 31)
        # first = datetime.date(2018, 1, 25)
        first = last - np.timedelta64(60, 'D')
        # first = all_dates.min()

        plt.xlim(first, last)
        plt.xlabel('Date')
        months = matplotlib.dates.MonthLocator()
        plt.gca().xaxis.set_major_locator(months)
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
        days   = matplotlib.dates.DayLocator()
        plt.gca().xaxis.set_minor_locator(days)
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)

        plt.ylabel('Aberration [nm RMS]')
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=5))
        plt.ylim(0, 90)

        plt.grid(True)

        plt.title('ZELDA monitoring')

        plt.legend(loc='upper right')

        plt.tight_layout()

        if save:
            plt.savefig(str(path / 'products' / 'opd_info_full.pdf'))
    else:
        #
        # Monitoring on a specific date
        #

        plt.figure(1, figsize=(30, 11))
        plt.clf()

        # PSD
        plt.subplot(121)

        for f in range(5):
            data = opd_info.loc[(date, f), slice('no_ttf_psd.1', 'no_ttf_psd.99')].squeeze()
            plt.plot(np.arange(1, 100), data, linestyle='-', marker='', color=color[f],
                     label='Iteration {0}'.format(f))


        # specs (without beam-shift)
        plt.axhline( 7.9, xmin=1, xmax=4, color='k', linestyle='--', linewidth=2)
        plt.axhline(11.6, xmin=4, xmax=20, color='k', linestyle='--', linewidth=2)
        plt.axhline(32.6, xmin=20, xmax=100, color='k', linestyle='--', linewidth=2)

        plt.xlim(0, 100)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        plt.xlabel('Spatial frequency [cycle/pupil]')

        plt.ylabel('Aberration [(nm RMS) / (cycle/pupil)]')
        plt.gca().set_yscale('log')
        plt.ylim(1, 50)

        plt.grid(True, which='both')

        plt.title('{0} - PSD'.format(date))

        plt.legend(loc='upper right')

        # Zernike
        plt.subplot(122)

        for f in range(5):
            data = opd_info.loc[(date, f), slice('full_zern.0', 'full_zern.49')].squeeze()
            plt.bar(np.arange(len(data)), np.abs(data), color=color[f], label='Iteration {0}'.format(f))

        plt.fill_between(np.array([0, 3.5]), np.array([50, 50]), color='k', edgecolor='k',
                         alpha=0.2, hatch='///')

        plt.axhline(0, color='k', linestyle='--')

        plt.xlim(0, 40)
        plt.xlabel('Zernike polynomial')

        plt.ylim(1, 40)
        plt.ylabel('Aberration [abs. nm RMS]')

        plt.title('{0} - Zernike'.format(date))

        plt.legend(loc='upper right')

        # final
        plt.tight_layout()

        if save:
            plt.savefig(str(path / 'products' / 'opd_info_{0}.pdf'.format(date)))

