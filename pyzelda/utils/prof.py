#!/usr/bin/env python
# coding: utf-8

import numpy as np


def prof(img, ptype=1, step=1, mask=None, center=None, rmax=0, clip=True, exact=False):
    '''
    Calculate azimuthal statistics of an image

    Parameters
    ----------
    img : array
        Image on which the profiles
        
    ptype : int, optional
        Type of profiles
            1: mean (default)
            2: standard deviation
            3: variance
            4: median
            5: max
            
    mask : array, optional
        Mask for invalid values (must have the same size as image)
        
    center : {array,tupple,list}, optional
        Center of the image

    rmax : float
        Maximum radius for calculating the profile, in pixel. Default is 0 (no limit)
    
    clip : bool, optional
        Clip profile to area of image where there is a full set of data
        
    exact : bool, optional
        Performs an exact estimation of the profile. This can be very long for 
        large arrays. Default is False, which rounds the radial distance to the 
        closest 1 pixel.
    
    Returns
    -------
    prof : array
        1D profile vector
        
    rad : array
        Separation vector, in pixel

    History
    -------
    2016-07-08 - Arthur Vigan
        Improved algorithm    
    
    2015-10-05 - Arthur Vigan
        First version based on IDL equivalent
    '''
    
    # array dimensions
    dimx = img.shape[1]
    dimy = img.shape[0]

    # center
    if (center is None):
        center = (dimx // 2, dimy // 2)

    # masking
    if mask is not None:
        # check size
        if mask.shape != img.shape:
            raise ValueError('Image and mask don''t have the same size. Returning.')

        img[mask == 0] = np.nan
        
    # intermediate cartesian arrays
    x = np.arange(dimx, dtype=np.int64) - center[0]
    y = np.arange(dimy, dtype=np.int64) - center[1]
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    # rounds for faster calculation
    if exact is not True:
        rr = np.round(rr, decimals=0)
    
    # find unique radial values
    uniq = np.unique(rr, return_inverse=True, return_counts=True)
    r_uniq_val = uniq[0]
    r_uniq_inv = uniq[1]
    r_uniq_cnt = uniq[2]

    # number of elements
    if clip is True:
        extr  = np.abs(np.array((x[0], x[-1], y[0], y[-1])))
        r_max = extr.min()
        i_max = int(r_uniq_val[r_uniq_val <= r_max].size)
    else:
        r_max = r_uniq_val.max()
        i_max = r_uniq_val.size

    # limit extension of profile
    if (rmax > 0):
        r_max = rmax
        i_max = int(r_uniq_val[r_uniq_val <= r_max].size)
        
    t_max = r_uniq_cnt[0:i_max].max()

    # intermediate polar array
    polar = np.empty((i_max, t_max), dtype=img.dtype)
    polar.fill(np.nan)
    
    img_flat = img.ravel()
    for r in range(i_max):
        cnt = r_uniq_cnt[r]
        val = img_flat[r_uniq_inv == r]
        polar[r, 0:cnt] = val
            
    # calculate profile
    rad  = r_uniq_val[0:i_max]
    
    if step == 1:
        # fast statistics if step=1
        if ptype == 1:
            prof = np.nanmean(polar, axis=1)
        elif ptype == 2:
            prof = np.nanstd(polar, axis=1, ddof=1)
        elif ptype == 3:
            prof = np.nanvar(polar, axis=1)
        elif ptype == 4:
            prof = np.nanmedian(polar, axis=1)
        elif ptype == 5:
            prof = np.nanmax(polar, axis=1)
        else:
            raise ValueError('Unknown statistics ptype = {0}'.format(ptype))
    else:
        # slower if we need step > 1
        prof = np.zeros(i_max, dtype=img.dtype)
        for r in range(i_max):
            idx = ((rad[r]-step/2) <= rad) & (rad <= (rad[r]+step/2))
            val = polar[idx, :]
            
            if ptype == 1:
                prof[r] = np.nanmean(val)
            elif ptype == 2:
                prof[r] = np.nanstd(val)
            elif ptype == 3:
                prof[r] = np.nanvar(val)
            elif ptype == 4:
                prof[r] = np.nanmedian(val)
            elif ptype == 5:
                prof[r] = np.nanmax(val)
            else:
                raise ValueError('Unknown statistics ptype = {0}'.format(ptype))

    return prof, rad


def mean(img, **kwargs):
    '''
    Calculate the azimuthal mean of an image
    
    Parameters
    ----------
    See documentation for profile()
    '''

    return prof(img, ptype=1, **kwargs)


def std(img, **kwargs):
    '''
    Calculate the azimuthal standard deviation of an image
    
    Parameters
    ----------
    See documentation for profile()
    '''

    return prof(img, ptype=2, **kwargs)


def var(img, **kwargs):
    '''
    Calculate the azimuthal variance of an image
    
    Parameters
    ----------
    See documentation for profile()
    '''

    return prof(img, ptype=3, **kwargs)


def median(img, **kwargs):
    '''
    Calculate the azimuthal median of an image
    
    Parameters
    ----------
    See documentation for profile()
    '''

    return prof(img, ptype=4, **kwargs)


def max(img, **kwargs):
    '''
    Calculate the azimuthal maximum of an image
    
    Parameters
    ----------
    See documentation for profile()
    '''

    return prof(img, ptype=5, **kwargs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dimx = 512
    dimy = 512
    center = (dimx//2, dimy//2)
    
    x = np.arange(dimx, dtype=np.float64) - center[0]
    y = np.arange(dimy, dtype=np.float64) - center[1]
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    p, r = mean(rr, clip=True, center=(120, 200))

    plt.figure()
    plt.plot(r, p)
    
    
    
    
    
    
