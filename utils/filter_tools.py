import torch
import numpy as np
from scipy import signal as signal

def dmaxflat(N,d):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF dmaxflat.m
    FROM THE Nonsubsampled Contourlet Toolbox.   -- Stefan Loock, Dec 2016.
    returns 2-D diamond maxflat filters of order 'N'
    the filters are nonseparable and 'd' is the (0,0) coefficient, being 1 or 0
    depending on use.
    by Arthur L. da Cunha, University of Illinois Urbana-Champaign
    Aug 2004

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """
    if (N > 7) or (N < 1):
        raise ValueError('Error: N must be in {1,2,...,7}')

    if N == 4:
        h = np.array([[0, -5, 0, -3, 0], [-5, 0, 52, 0, 34],
                        [0, 52, 0, -276, 0], [-3, 0, -276, 0, 1454],
                        [0, 34, 0, 1454, 0]])/np.power(2,12)
        h = np.append(h, np.fliplr(h[:,0:-1]),1)
        h = np.append(h, np.flipud(h[0:-1,:]),0)
        h[4,4] = d
    else:
        raise ValueError('Not implemented!')
    return h

def mctrans(b,t):
    """
    This is a translation of the original Matlab implementation of mctrans.m
    from the Nonsubsampled Contourlet Toolbox by Arthur L. da Cunha.
    MCTRANS McClellan transformation
        H = mctrans(B,T)
    produces the 2-D FIR filter H that corresponds to the 1-D FIR filter B
    using the transform T.
    Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    # if mod(n,2) != 0 -> error
    n = (b.size-1)//2

    b = np.fft.fftshift(b[::-1]) #inverse fftshift
    b = b[::-1]
    a = np.zeros(n+1)
    a[0] = b[0]
    a[1:n+1] = 2*b[1:n+1]

    inset = np.floor((np.asarray(t.shape)-1)/2)
    inset = inset.astype(int)
    # Use Chebyshev polynomials to compute h
    P0 = 1
    P1 = t;
    h = a[1]*P1;
    rows = int(inset[0]+1)
    cols = int(inset[1]+1)
    h[rows-1,cols-1] = h[rows-1,cols-1]+a[0]*P0;
    for i in range(3,n+2):
        P2 = 2*signal.convolve2d(t, P1)
        rows = (rows + inset[0]).astype(int)
        cols = (cols + inset[1]).astype(int)
        if i == 3:
            P2[rows-1,cols-1] = P2[rows-1,cols-1] - P0
        else:
            P2[rows[0]-1:rows[-1],cols[0]-1:cols[-1]] = P2[rows[0]-1:rows[-1],
                                                        cols[0]-1:cols[-1]] - P0
        rows = inset[0] + np.arange(np.asarray(P1.shape)[0])+1
        rows = rows.astype(int)
        cols = inset[1] + np.arange(np.asarray(P1.shape)[1])+1
        cols = cols.astype(int)
        hh = h;
        h = a[i-1]*P2
        h[rows[0]-1:rows[-1], cols[0]-1:cols[-1]] = h[rows[0]-1:rows[-1],
                                                        cols[0]-1:cols[-1]] + hh
        P0 = P1;
        P1 = P2;
    h = np.rot90(h,2)
    return h

def modulate2(x, type, center=np.array([0, 0])):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF
    modulate2.m FROM THE Nonsubsampled Contourlet Toolbox.
    MODULATE2	2D modulation
            y = modulate2(x, type, [center])
    With TYPE = {'r', 'c' or 'b'} for modulate along the row, or column or
    both directions.
    CENTER secify the origin of modulation as floor(size(x)/2)+1+center
    (default is [0, 0])
    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """
    size = np.asarray(x.shape)
    if x.ndim == 1:
        if np.array_equal(center, [0, 0]):
            center = 0
    origin = np.floor(size/2)+1+center
    n1 = np.arange(size[0])-origin[0]+1
    if x.ndim == 2:
        n2 = np.arange(size[1])-origin[1]+1
    else:
        n2 = n1
    if type == 'r':
        m1 = np.power(-1,n1)
        if x.ndim == 1:
            y = x*m1
        else:
            y = x * np.transpose(np.tile(m1, (size[1], 1)))
    elif type == 'c':
        m2 = np.power(-1,n2)
        if x.ndim == 1:
            y = x*m2
        else:
            y = x * np.tile(m2, np.array([size[0], 1]))
    elif type == 'b':
        m1 = np.power(-1,n1)
        m2 = np.power(-1,n2)
        m = np.outer(m1, m2)
        if x.ndim == 1:
            y = x * m1
        else:
            y = x * m
    return y

def dfilters(fname, type):
    """
    generate directional 2D filters

    input: 
        fname: filter names, default: 'dmaxflat' (maximally flat 2D fan filter)
        type: 'd' or 'r' for decomposition or reconstruction filters

    output:
        h0, h1: diamond filter pair (lowpass and highpass)
    
    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """

    if fname == 'dmaxflat4':
        M1 = 1/np.sqrt(2)
        M2 = M1
        k1 = 1-np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25*k2*k3, 0.5*k2, 1+0.5*k2*k3])*M1
        h = np.append(h, h[-2::-1])
        g = np.array([-0.125*k1*k2*k3, 0.25*k1*k2,
                    -0.5*k1-0.5*k3-0.375*k1*k2*k3, 1+0.5*k1*k2])*M2
        g = np.append(g, h[-2::-1])

        B = dmaxflat(4,0)
        h0 = mctrans(h,B)
        g0 = mctrans(g,B)

        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    else:
        raise ValueError("Filter type not implemented!")

    return h0, h1

def dshear(inputArray, k, axis):
    """
    Computes the discretized shearing operator for a given inputArray, shear
    number k and axis.
    This version is adapted such that the MATLAB indexing can be used here in the
    Python version.

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """
    if k==0:
        return inputArray
    rows = np.asarray(inputArray.shape)[0]
    cols = np.asarray(inputArray.shape)[1]

    shearedArray = torch.zeros((rows, cols), dtype=inputArray.dtype, device=inputArray.device)

    if axis == 0:
        for col in range(cols):
            shearedArray[:,col] = torch.roll(inputArray[:,col], int(k * np.floor(cols/2-col)))
    else:
        for row in range(rows):
            shearedArray[row,:] = torch.roll(inputArray[row,:], int(k * np.floor(rows/2-row)))
    return shearedArray

def padArray(array, newSize):
    """
    Implements the padding of an array as performed by the Matlab variant. 

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """
    if np.isscalar(newSize):
        #padSizes = np.zeros((1,newSize))
        # check if array is a vector...
        currSize = array.size
        paddedArray = torch.zeros(newSize, dtype=array.dtype, device=array.device)
        sizeDiff = newSize - currSize
        idxModifier = 0
        if sizeDiff < 0:
            raise ValueError("Error: newSize is smaller than actual array size.")
        if sizeDiff == 0:
            print("Warning: newSize is equal to padding size.")
        if sizeDiff % 2 == 0:
            padSizes = sizeDiff//2
        else:
            padSizes = int(np.ceil(sizeDiff/2))
            if currSize % 2 == 0:
                # index 1...k+1
                idxModifier = 1
            else:
                # index 0...k
                idxModifier = 0
        print(padSizes)
        paddedArray[padSizes-idxModifier:padSizes+currSize-idxModifier] = array

    else:
        padSizes = torch.zeros(newSize.size, dtype=array.dtype, device=array.device)
        paddedArray = torch.zeros((newSize[0], newSize[1]), dtype=array.dtype, device=array.device)
        idxModifier = np.array([0, 0])
        currSize = np.asarray(array.shape)
        if array.ndim == 1:
            currSize = np.array([len(array), 0])
        for k in range(newSize.size):
            sizeDiff = newSize[k] - currSize[k]
            if sizeDiff < 0:
                raise ValueError("Error: newSize is smaller than actual array size in dimension " + str(k) + ".")
            if sizeDiff == 0:
                print("Warning: newSize is equal to padding size in dimension " + str(k) + ".")
            if sizeDiff % 2 == 0:
                padSizes[k] = sizeDiff//2
            else:
                padSizes[k] = np.ceil(sizeDiff/2)
                if currSize[k] % 2 == 0:
                    # index 1...k+1
                    idxModifier[k] = 1
                else:
                    # index 0...k
                    idxModifier[k] = 0
        padSizes = padSizes.int()

        # if array is 1D but paddedArray is 2D we simply put the array (as a
        # row array in the middle of the new empty array). this seems to be
        # the behavior of the ShearLab routine from matlab.
        if array.ndim == 1:
            paddedArray[padSizes[1], padSizes[0]:padSizes[0]+currSize[0]+idxModifier[0]] = array
        else:
            paddedArray[padSizes[0]-idxModifier[0]:padSizes[0]+currSize[0]-idxModifier[0],
                    padSizes[1]:padSizes[1]+currSize[1]+idxModifier[1]] = array
    return paddedArray

def upsample_np(array, dims, nZeros):
    """
    Performs an upsampling by a number of nZeros along the dimenion(s) dims
    for a given array.

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """
    assert dims == 0 or dims == 1

    if array.ndim == 1:
        sz = len(array)
        idx = range(1,sz)
        arrayUpsampled = np.insert(array, idx, 0)
    else:
        sz = np.asarray(array.shape)
        if dims == 0:
            arrayUpsampled = np.zeros(((sz[0]-1)*(nZeros+1)+1, sz[1]))
            for col in range(sz[0]):
                arrayUpsampled[col*(nZeros)+col,:] = array[col,:]
        if dims == 1:
            arrayUpsampled = np.zeros((sz[0], ((sz[1]-1)*(nZeros+1)+1)))
            for row in range(sz[1]):
                arrayUpsampled[:,row*(nZeros)+row] = array[:,row]
    return torch.Tensor(arrayUpsampled)

def upsample(array, dims, nZeros):
    """
    Performs an upsampling by a number of nZeros along the dimenion(s) dims
    for a given array.

    Taken from pyshearlab (https://github.com/stefanloock/pyshearlab).
    """
    assert dims == 0 or dims == 1

    if array.ndim == 1:
        sz = len(array)
        array_zero = torch.zeros((sz-1), dtype=array.dtype, device=array.device)
        arrayUpsampled = torch.empty((2*sz - 1), dtype=array.dtype, device=array.device)
        arrayUpsampled[0::2] = array 
        arrayUpsampled[1:-1:2] = array_zero
    else:
        sz = np.asarray(array.shape)
        # behaves like in matlab: dims == 1 and dims == 2 instead of 0 and 1.
        if dims == 0:
            arrayUpsampled = torch.zeros(((sz[0]-1)*(nZeros+1)+1, sz[1]), dtype=array.dtype, device=array.device)
            for col in range(sz[0]):
                arrayUpsampled[col*(nZeros)+col,:] = array[col,:]
        if dims == 1:
            arrayUpsampled = torch.zeros((sz[0], ((sz[1]-1)*(nZeros+1)+1)), dtype=array.dtype, device=array.device)
            for row in range(sz[1]):
                arrayUpsampled[:,row*(nZeros)+row] = array[:,row]
    return arrayUpsampled
