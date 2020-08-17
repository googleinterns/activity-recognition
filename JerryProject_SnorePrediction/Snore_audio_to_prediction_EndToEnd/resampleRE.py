# This file is modified from python library "resampy"
# Used for downsample wav file from 44.1kHz to 16kHz
# However, cannot be utilized directly in Android Chaquopy
# for it used stored .npz file and incompatible compilor numba.
# Modified so now can be used in Android Chaquopy.

# Deprecated for too long execution time.

import numba

import scipy.signal
import numpy as np
import six
import kaiser_best

FILTER_FUNCTIONS = ['sinc_window']

__all__ = ['get_filter'] + FILTER_FUNCTIONS


def resample(x, sr_orig, sr_new, axis=-1, filter='kaiser_best', **kwargs):
    '''Resample a signal x from sr_orig to sr_new along a given axis.
    Parameters
    ----------
    x : np.ndarray, dtype=np.float*
        The input signal(s) to resample.
    sr_orig : int > 0
        The sampling rate of x
    sr_new : int > 0
        The target sampling rate of the output signal(s)
    axis : int
        The target axis along which to resample `x`
    filter : optional, str or callable
        The resampling filter to use.
        By default, uses the `kaiser_best` (pre-computed filter).
    kwargs
        additional keyword arguments provided to the specified filter
    Returns
    -------
    y : np.ndarray
        `x` resampled to `sr_new`
    Raises
    ------
    ValueError
        if `sr_orig` or `sr_new` is not positive
    TypeError
        if the input signal `x` has an unsupported data type.
    Examples
    --------
    '''

    print(x)

    print("Jerry resampleRE.py: start resample()")
    if sr_orig <= 0:
        raise ValueError('Invalid sample rate: sr_orig={}'.format(sr_orig))

    if sr_new <= 0:
        raise ValueError('Invalid sample rate: sr_new={}'.format(sr_new))

    sample_ratio = float(sr_new) / sr_orig

    # Set up the output shape
    shape = list(x.shape)
    shape[axis] = int(shape[axis] * sample_ratio)

    if shape[axis] < 1:
        raise ValueError('Input signal length={} is too small to '
                         'resample from {}->{}'.format(x.shape[axis], sr_orig, sr_new))

    # Preserve contiguity of input (if it exists)
    # If not, revert to C-contiguity by default
    if x.flags['F_CONTIGUOUS']:
        order = 'F'
    else:
        order = 'C'

    y = np.zeros(shape, dtype=x.dtype, order=order)

    print("Jerry resampleRE.py: before get_filter(), kaiser_best")

    interp_win, precision = np.array(kaiser_best.half_window), kaiser_best.precision

    if sample_ratio < 1:
        interp_win *= sample_ratio

    interp_delta = np.zeros_like(interp_win)
    interp_delta[:-1] = np.diff(interp_win)
    # Construct 2d views of the data with the resampling axis on the first dimension
    print()
    print(x)
    print(x.shape)
    x_2d = x.swapaxes(0, axis).reshape((x.shape[axis], -1))
    y_2d = y.swapaxes(0, axis).reshape((y.shape[axis], -1))
    print(x_2d)
    print(x_2d.shape)
    print()

    print("Jerry resampleRE.py: before resample_f()")
    resample_f(x_2d, y_2d, sample_ratio, interp_win, interp_delta, precision)

    print("Jerry resampleRE.py: finish resample()")

    return y


def sinc_window(num_zeros=64, precision=9, window=None, rolloff=0.945):
    '''Construct a windowed sinc interpolation filter
    Parameters
    ----------
    num_zeros : int > 0
        The number of zero-crossings to retain in the sinc filter
    precision : int > 0
        The number of filter coefficients to retain for each zero-crossing
    window : callable
        The window function.  By default, uses Blackman-Harris.
    rolloff : float > 0
        The roll-off frequency (as a fraction of nyquist)
    Returns
    -------
    interp_window: np.ndarray [shape=(num_zeros * num_table + 1)]
        The interpolation window (right-hand side)
    num_bits: int
        The number of bits of precision to use in the filter table
    rolloff : float > 0
        The roll-off frequency of the filter, as a fraction of Nyquist
    Raises
    ------
    TypeError
        if `window` is not callable or `None`
    ValueError
        if `num_zeros < 1`, `precision < 1`,
        or `rolloff` is outside the range `(0, 1]`.
    Examples
    '''

    if window is None:
        window = scipy.signal.blackmanharris
    elif not six.callable(window):
        raise TypeError('window must be callable, not type(window)={}'.format(type(window)))

    if not 0 < rolloff <= 1:
        raise ValueError('Invalid roll-off: rolloff={}'.format(rolloff))

    if num_zeros < 1:
        raise ValueError('Invalid num_zeros: num_zeros={}'.format(num_zeros))

    if precision < 0:
        raise ValueError('Invalid precision: precision={}'.format(precision))

    # Generate the right-wing of the sinc
    num_bits = 2**precision
    n = num_bits * num_zeros
    sinc_win = rolloff * np.sinc(rolloff * np.linspace(0, num_zeros, num=n + 1,
                                                       endpoint=True))

    # Build the window function and cut off the left half
    taper = window(2 * n + 1)[n:]

    interp_win = (taper * sinc_win)

    return interp_win, num_bits, rolloff


@numba.jit(nopython=True, nogil=True)
def resample_f(x, y, sample_ratio, interp_win, interp_delta, num_table):

    print("Jerry resampleRE.py: start resample_f")

    scale = min(1.0, sample_ratio)
    time_increment = 1./sample_ratio
    index_step = int(scale * num_table)
    time_register = 0.0

    n = 0
    frac = 0.0
    index_frac = 0.0
    offset = 0
    eta = 0.0
    weight = 0.0

    nwin = interp_win.shape[0]
    n_orig = x.shape[0]
    n_out = y.shape[0]
    n_channels = y.shape[1]

    for t in range(n_out):
        # n_out = 156911
        # Grab the top bits as an index to the input buffer
        n = int(time_register)

        # Grab the fractional component of the time index
        frac = scale * (time_register - n)

        # Offset into the filter
        index_frac = frac * num_table
        offset = int(index_frac)

        # Interpolation factor
        eta = index_frac - offset

        # Compute the left wing of the filter response
        i_max = min(n + 1, (nwin - offset) // index_step)
        for i in range(i_max):

            weight = (interp_win[offset + i * index_step] + eta * interp_delta[offset + i * index_step])
            for j in range(n_channels):
                y[t, j] += weight * x[n - i, j]

        # Invert P
        frac = scale - frac

        # Offset into the filter
        index_frac = frac * num_table
        offset = int(index_frac)

        # Interpolation factor
        eta = index_frac - offset

        # Compute the right wing of the filter response
        k_max = min(n_orig - n - 1, (nwin - offset)//index_step)
        for k in range(k_max):
            weight = (interp_win[offset + k * index_step] + eta * interp_delta[offset + k * index_step])
            for j in range(n_channels):
                y[t, j] += weight * x[n + k + 1, j]

        # Increment the time register
        time_register += time_increment