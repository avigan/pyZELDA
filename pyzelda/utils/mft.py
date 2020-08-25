# -*- coding: utf-8 -*-

'''
Matrix Fourier transform module.

Based on the formalism detailed in:

    Fast computation of Lyot-style coronagraph propagation
    Soummer, Pueyo, Sivaramakrishnan and Vanderbei
    2007, Optics Express, 15, 15935
'''

import numpy as np


def _mft(array, Na, Nb, m, inverse=False, cpix=False):
    '''
    Performs the matrix Fourier transform of an array.
    
    Based on the formalism detailed in:
    
        Fast computation of Lyot-style coronagraph propagation
        Soummer, Pueyo, Sivaramakrishnan and Vanderbei
        2007, Optics Express, 15, 15935

    Parameters
    ----------
    array : array
        The input array
    
    Na : int
        Number of pixels in direct space
    
    Nb : int
        Number of pixels in Fourier space
    
    m : float
        Number of lambda/D elements in Fourier space
    
    inverse : bool, optional
        Control if the direct or inverse transform is performed. Default is `False`
    
    cpix : bool, optional
        Center the MFT on one pixel, instead of between 4 pixels. Default is 'False'
        
    Returns
    -------
    return_value : array
        MFT of the input array
    '''
        
    if (inverse is True):
        sign = 1
    else:
        sign = -1
    
    # For pixel centering
    offsetA = 0
    offsetB = 0
    
    # If centering between 4 pixels
    if not(cpix):
        offsetA = .5/Na
        offsetB = .5/Nb
    
    coeff = m / (Na * Nb)
    
    x = np.linspace(-0.5, 0.5, Na, endpoint=False, dtype=np.double) + offsetA
    y = x
    
    u = m * (np.linspace(-0.5, 0.5, Nb, endpoint=False, dtype=np.double) + offsetB)
    v = u
    
    A1 = np.exp(sign*2j*np.pi*np.outer(u, x))
    A3 = np.exp(sign*2j*np.pi*np.outer(v, y))
    
    B = coeff * A1.dot(array).dot(A3.T)

    return B


def mft(array, Na, Nb, m, cpix=False):
    '''
    Performs the matrix Fourier transform of an array.
    
    Based on the formalism detailed in:
    
        Fast computation of Lyot-style coronagraph propagation
        Soummer, Pueyo, Sivaramakrishnan and Vanderbei
        2007, Optics Express, 15, 15935

    Parameters
    ----------
    array : array
        The input array
    
    Na : int
        Number of pixels in direct space
    
    Nb : int
        Number of pixels in Fourier space
    
    m : float
        Number of lambda/D elements in Fourier space
    
    cpix : bool, optional
        Center the MFT on one pixel, instead of between 4 pixels. Default is 'False'
         
    Returns
    -------
    return_value : array
        MFT of the input array
    '''
    
    return _mft(array, Na, Nb, m, inverse=False, cpix=cpix)


def imft(array, Na, Nb, m, cpix=True):
    '''
    Performs the inverse matrix Fourier transform of an array.
    
    Based on the formalism detailed in:
    
        Fast computation of Lyot-style coronagraph propagation
        Soummer, Pueyo, Sivaramakrishnan and Vanderbei
        2007, Optics Express, 15, 15935

    Parameters
    ----------
    array : array
        The input array
    
    Na : int
        Number of pixels in direct space
    
    Nb : int
        Number of pixels in Fourier space
    
    m : float
        Number of lambda/D elements in Fourier space
        
    Returns
    -------
    return_value : array
        MFT of the input array
    '''
    
    return _mft(array, Na, Nb, m, inverse=True, cpix=cpix)


if __name__ == "__main__":
    from disc import disc
    import matplotlib.pyplot as plt

    d = disc(200, 100)
    a = mft(d, 200, 400, 10)
    
    plt.imshow(np.abs(a)**2)
