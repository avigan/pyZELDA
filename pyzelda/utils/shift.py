# -*- coding: utf-8 -*-

'''
Image shift module for pyZELDA
'''

import collections
import numpy as np
import scipy.ndimage.interpolation as interp
import numpy.fft as fft


def shift_fft(array, shift_val):
    Ndim  = array.ndim
    dtype = array.dtype.kind
    
    if (dtype != 'f'):
        raise ValueError('Array must be float')
    
    shifted = array
    if (Ndim == 1):
        Nx = array.shape[0]
        
        x_ramp = np.arange(Nx, dtype=array.dtype) - Nx//2
        
        tilt = (2*np.pi/Nx) * (shift_val[0]*x_ramp)
        
        cplx_tilt = np.cos(tilt) + 1j*np.sin(tilt)
        cplx_tilt = fft.fftshift(cplx_tilt)
        narray    = fft.fft(fft.ifft(array) * cplx_tilt)
        shifted   = narray.real
    elif (Ndim == 2):
        Nx = array.shape[0]
        Ny = array.shape[1]
        
        x_ramp = np.outer(np.full(Nx, 1.), np.arange(Ny, dtype=array.dtype)) - Nx//2
        y_ramp = np.outer(np.arange(Nx, dtype=array.dtype), np.full(Ny, 1.)) - Ny//2
        
        tilt = (2*np.pi/Nx) * (shift_val[0]*x_ramp+shift_val[1]*y_ramp)
        
        cplx_tilt = np.cos(tilt) + 1j*np.sin(tilt)        
        cplx_tilt = fft.fftshift(cplx_tilt)
        
        narray    = fft.fft2(fft.ifft2(array) * cplx_tilt)
        shifted   = narray.real
    else:
        raise ValueError('This function can shift only 1D or 2D arrays')
    
    return shifted


def shift_interp(array, shift_val, mode='constant', cval=0.):
    shifted = interp.shift(array, np.flip(shift_val, 0), order=3, mode=mode, cval=cval)
        
    return shifted


def shift_roll(array, shift_val):
    Ndim  = array.ndim

    if (Ndim == 1):
        shifted = np.roll(array, shift_val[0])
    elif (Ndim == 2):
        shifted = np.roll(np.roll(array, shift_val[0], axis=1), shift_val[1], axis=0)
    else:
        raise ValueError('This function can shift only 1D or 2D arrays')
        
    return shifted


def shift(array, value, method='fft', mode='constant', cval=0.):
    '''
    Shift a 1D or 2D input array.
    
    The array can be shift either using FFT or using interpolation. Note that 
    if the shifting value is an integer, the function uses numpy roll procedure
    to shift the array.
    
    Parameters
    ----------
    array : array
        The array to be shift
    
    value : float or sequence
        The shift along the axes. If a float, shift_val is the same for each axis. 
        If a sequence, value should contain one value for each axis.
    
    method : str, optional
        Method for shifting the array, ('fft', 'interp', 'roll'). Default is 'fft'
    
    mode : str
        Points outside the boundaries of the input are filled according 
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        This value is ignored if method='fft'. Default is 'constant'.
    
    cval : float, optional
        Value used for points outside the boundaries of the input if 
        mode='constant'. Default is 0.0
        
    Returns
    -------
    shift : array
        The shifted array
    '''

    # array dimensions
    Ndim  = array.ndim
    if (Ndim != 1) and (Ndim != 2):
        raise ValueError('This function can shift only 1D or 2D arrays')

    # check that shift value is fine
    if isinstance(value, collections.Iterable):
        if (len(value) != Ndim):
            raise ValueError("Number of dimensions in array and shift don't match")
    elif isinstance(value, (int, float)):
        value = tuple([value for i in range(Ndim)])
    else:
        raise ValueError("Shift value of type '{0}' is not allowed".format(type(shift).__name__))
    value = np.array(value)

    # check if shift values are int and automatically change method in case they are
    if (value.dtype.kind == 'i'):
        method = 'roll'

    # shift with appropriate function                
    method = method.lower()
    if (method == 'fft'):
        shifted = shift_fft(array, value)
    elif (method == 'interp'):
        shifted = shift_interp(array, value, mode=mode, cval=cval)
    elif (method == 'roll'):
        value = np.round(value).astype(int)
        shifted = shift_roll(array, value)
    else:
        raise ValueError("Unknown shift method '{0}'".format(method))

    return shifted
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aperture as aper
    
    from astropy.io import fits

    c = (75, 200)
    s = (80.2, 30.7)

    # images
    img = aper.disc(300, 50, center=c, diameter=True)
    # img = img[100:, :]
    img_roll   = shift(img, value=s, method='roll')
    img_interp = shift(img, value=s, method='interp')
    img_fft    = shift(img, value=s, method='fft')

    # plot
    fig = plt.figure(1, figsize=(12, 12))
    plt.clf()

    ax = fig.add_subplot(221)
    ax.plot(c[0], c[1], marker='+')
    ax.imshow(img)
    ax.set_title('Original')

    ax = fig.add_subplot(222)
    ax.imshow(img_roll)
    ax.plot(c[0]+s[0], c[1]+s[1], marker='+')
    ax.set_title('Roll')

    ax = fig.add_subplot(223)
    ax.imshow(img_interp)
    ax.plot(c[0]+s[0], c[1]+s[1], marker='+')
    ax.set_title('Interp')

    ax = fig.add_subplot(224)
    ax.imshow(img_fft)
    ax.plot(c[0]+s[0], c[1]+s[1], marker='+')
    ax.set_title('FFT')

    plt.tight_layout()
