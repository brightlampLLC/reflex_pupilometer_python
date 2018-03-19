# ----------------------------------------------------------------------------------------------------------------------
#   Name: Pupil_Dilation_Registered_Images
#   Purpose: Generates FFTW objects
#
#   !/usr/bin/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 18 22:14:12 2018
#   @author: brettmeyers
#   @author: jonholt
# ----------------------------------------------------------------------------------------------------------------------

import pyfftw
import multiprocessing

# These are the variables set in the main function that need to be added into this file
# First usage
xResample = imgProp['size'][1]
yResample = imgProp['size'][0]

# Second usage
cropWin = np.round(np.median(WinDims[frng, :], axis=0)).astype(int)

# Generate FFTW objects - https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
#   Parameters:
#       input_array - Return the input array that is associated with the FFTW instance.
#       output_array - Return the output array that is associated with the FFTW instance.
#       axes – Return the axes for the planned FFT in canonical form, as a tuple of positive integers.
#       direction – Return the planned FFT direction. Either ‘FFTW_FORWARD’ or ‘FFTW_BACKWARD’.
#       flags – Return which flags were used to construct the FFTW object.
#       threads – Tells the wrapper how many threads to use when invoking FFTW, with a default of 1.
#       planning_timelimit - Indicates the maximum number of seconds it should spend planning the FFT.
#   Example:
#       pyfftw.FFTW(input_array,
#                  output_array,
#                   axes=(-1, ),
#                   direction='FFTW_FORWARD',
#                   flags=('FFTW_MEASURE', ),
#                   threads=1,
#                   planning_timelimit=None)

# First usage
fft01FullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                axes=(-2, -1),
                                direction='FFTW_FORWARD',
                                flags=('FFTW_MEASURE', ),
                                threads=multiprocessing.cpu_count(),
                                planning_timelimit=None)

fft02FullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                axes=(-2, -1),
                                direction='FFTW_FORWARD',
                                flags=('FFTW_MEASURE', ),
                                threads=multiprocessing.cpu_count(),
                                planning_timelimit=None)

ifftFullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                               pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                               axes=(-2, -1),
                               direction='FFTW_BACKWARD',
                               flags=('FFTW_MEASURE', ),
                               threads=multiprocessing.cpu_count(),
                               planning_timelimit=None)

# Second usage
fft01FMCObj = pyfftw.FFTW(pyfftw.empty_aligned((cropWin[2], cropWin[3]), dtype='complex128'),
                              pyfftw.empty_aligned((cropWin[2], cropWin[3]), dtype='complex128'),
                              axes=(-2, -1),
                              direction='FFTW_FORWARD',
                              flags=('FFTW_MEASURE', ),
                              threads=multiprocessing.cpu_count(),
                              planning_timelimit=None)

fft02FMCObj = pyfftw.FFTW(pyfftw.empty_aligned((cropWin[2], cropWin[3]), dtype='complex128'),
                          pyfftw.empty_aligned((cropWin[2], cropWin[3]), dtype='complex128'),
                          axes=(-2, -1),
                          direction='FFTW_FORWARD',
                          flags=('FFTW_MEASURE', ),
                          threads=multiprocessing.cpu_count(),
                          planning_timelimit=None)

ifftFMCObj = pyfftw.FFTW(pyfftw.empty_aligned((cropWin[2], cropWin[3]), dtype='complex128'),
                         pyfftw.empty_aligned((cropWin[2], cropWin[3]), dtype='complex128'),
                         axes=(-2, -1),
                         direction='FFTW_BACKWARD',
                         flags=('FFTW_MEASURE', ),
                         threads=multiprocessing.cpu_count(),
                         planning_timelimit=None)