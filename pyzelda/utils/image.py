# -*- coding: utf-8 -*-

'''
Image utility functions for pyZELDA
'''

import numpy as np

from astropy.convolution import convolve, Box2DKernel
from scipy import optimize


def sigma_filter(img, box=5, nsigma=3., iterate=False, maxIters=20, _iters=0):
    '''
    Performs sigma-clipping over an image

    Parameters
    ----------
    img : array
        The input image
    
    box : int, optional
        Box size for the sigma-clipping. Default is 5 pixel
    
    nsigma : float, optional
        Sigma value. Default if 3.
    
    iterate : bool, optional
        Controls if the filtering is iterative. Default is False
    
    maxIters : int, optional
        Maximum number of iterations. Default is 20
    
    _iters : int (internal)
        Internal counter to keep track during iterative sigma-clipping
        
    Returns
    -------
    return_value : array
        Input image with clipped values
    '''
    
    box2 = box**2
    
    kernel = Box2DKernel(box)
    mean = (convolve(img, kernel)*box2 - img) / (box2-1)
    
    imdev = (img - mean)**2
    fact = nsigma**2 / (box2-2)
    imvar = fact*(convolve(imdev, kernel)*box2 - imdev)
    
    wok = np.nonzero(imdev < imvar)
    nok = wok[0].size
    
    npix = img.size
    
    if (nok == npix):
        return img
       
    if (nok > 0):
        mean[wok] = img[wok]
    
    if (iterate is True):
        _iters = _iters+1
        if (_iters >= maxIters):
            return mean
        return sigma_filter(mean, box=box, nsigma=nsigma, iterate=True, _iters=_iters)
    
    return mean


def distance_from_center(c, x, y):
    '''
    Distance of each 2D points from the center (xc, yc)

    Parameters
    ----------
    c : array_like
        Coordinates of the center

    x,y : array_like
        Arrays with the x,y coordinates
    '''
    xc = c[0]
    yc = c[1]
    
    Ri = np.sqrt((x-xc)**2 + (y-yc)**2)
    
    return Ri - Ri.mean()


def least_square_circle(x, y):
    '''
    Least-square determination of the center of a circle

    Parameters
    ----------
    x,y : array_like
        Arrays with the x,y coordinates of the points on/inside the circle
    '''
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    
    center, ier = optimize.leastsq(distance_from_center, center_estimate, args=(x, y))

    # results
    xc, yc = center
    Ri     = np.sqrt((x-xc)**2 + (y-yc)**2)
    R      = Ri.mean()    
    residu = np.sum((Ri - R)**2)
    
    return xc, yc, R, residu

