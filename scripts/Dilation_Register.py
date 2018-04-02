# ----------------------------------------------------------------------------------------------------------------------
#   Name: Dilation_Register
#   Purpose: Registers the dilation of the pupil in the video
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 18 22:14:12 2018
#   @author: brettmeyers
#   @author: jonholt
# ----------------------------------------------------------------------------------------------------------------------

import sys
import cv2
import pyfftw
import imageio
import numpy as np
import scipy as sp
import multiprocessing
import pylab
import numba
import pandas
import matplotlib.pyplot as plt

from numpy import log, multiply, sqrt, conj
from scripts.SubPixel2D import subPixel2D

# Import functions from new Coordinate Transform
from scripts.Coordinate_Transform import LPGrid, tformLP


def spatialRegister(i, frame01, frame02, Win2D, MaxRad, errthresh, iterthresh, dispX, dispY, scldisp):
    # initialize iterations & error
    errval = float(100)
    iteration = 0
    TRev = np.eye(3, dtype=float)
    TFor = np.eye(3, dtype=float)
    xResample = frame01.shape[1]
    yResample = frame01.shape[0]

    # Construct Log Polar Grid
    imgLPX, imgLPY = LPGrid(xResample, yResample, 1, MaxRad, np.max([xResample, yResample]), np.min([xResample, yResample]))

    # Generate meshgrid
    imgX, imgY = np.meshgrid(np.linspace(-xResample / 2 + 0.5, xResample / 2 - 0.5, xResample),
                             np.linspace(-yResample / 2 + 0.5, yResample / 2 - 0.5, yResample))

    # Generate FFTW objects - https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html
    #   Parameters:
    #       input_array - Return the input array that is associated with the FFTW instance.
    #       output_array - Return the output array that is associated with the FFTW instance.
    #       axes – Return the axes for the planned FFT in canonical form, as a tuple of positive integers.
    #       direction – Return the planned FFT direction. Either ‘FFTW_FORWARD’ or ‘FFTW_BACKWARD’.
    #       flags – Return which flags were used to construct the FFTW object.
    #       threads – Tells the wrapper how many threads to use when invoking FFTW, with a default of 1.
    #       planning_timelimit - Indicates the maximum number of seconds it should spend planning the FFT.
    # Example:
    #   pyfftw.FFTW(input_array,
    #               output_array,
    #               axes=(-1, ),
    #               direction='FFTW_FORWARD',
    #               flags=('FFTW_MEASURE', ),
    #               threads=1,
    #               planning_timelimit=None)
    fft01FullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                    pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                    axes=(-2, -1),
                                    direction='FFTW_FORWARD',
                                    flags=('FFTW_MEASURE',),
                                    threads=multiprocessing.cpu_count(),
                                    planning_timelimit=None)

    fft02FullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                    pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                    axes=(-2, -1),
                                    direction='FFTW_FORWARD',
                                    flags=('FFTW_MEASURE',),
                                    threads=multiprocessing.cpu_count(),
                                    planning_timelimit=None)

    ifftFullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                   pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
                                   axes=(-2, -1),
                                   direction='FFTW_BACKWARD',
                                   flags=('FFTW_MEASURE',),
                                   threads=multiprocessing.cpu_count(),
                                   planning_timelimit=None)

    while all((errval > errthresh, iteration < iterthresh)):
        # Reconstruct images based on transform matrices
        fr01 = cv2.warpAffine(frame01.astype(float),
                              TFor[0:2, :],
                              (xResample, yResample),
                              cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS).astype(float)
        fr01 = np.nan_to_num(fr01)
        fr01 -= fr01.mean(axis=(0, 1))
        fr01 = multiply(fr01, Win2D)
        fr02 = cv2.warpAffine(frame02.astype(float),
                              TRev[0:2, :],
                              (xResample, yResample),
                              cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS).astype(float)
        fr02 = np.nan_to_num(fr02)
        fr02 -= fr02.mean(axis=(0, 1))
        fr02 = multiply(fr02, Win2D)

        # Calculate FFTs for image pair
        FFT01 = sp.fftpack.fftshift(fft01FullImageObj(fr01))
        FFT02 = sp.fftpack.fftshift(fft02FullImageObj(fr02))

        # Log Polar Unwrap
        lp01 = (tformLP(imgX, imgY, imgLPX, imgLPY, imgLPX.shape[1], imgLPX.shape[0], np.abs(FFT01).astype(float)))
        lp02 = (tformLP(imgX, imgY, imgLPX, imgLPY, imgLPX.shape[1], imgLPX.shape[0], np.abs(FFT02).astype(float)))

        # Run FMT on FFTs
        FMT01 = multiply(Win2D, lp01).astype(float)
        FMT02 = multiply(Win2D, lp02).astype(float)

        # Calculate FFTs of FMTs
        FMC01 = fft01FullImageObj(FMT01)
        FMC02 = fft02FullImageObj(FMT02)

        # Run Translation Subfunc
        trnsdisp = subPixel2D(abs(sp.fftpack.fftshift(ifftFullImageObj(multiply(FFT01, conj(FFT02))))))

        # Store new displacement
        dispX[i] += trnsdisp[1]
        dispY[i] += trnsdisp[0]

        # Run Scaling Subfunc
        fmcdisp = subPixel2D(abs(sp.fftpack.fftshift(ifftFullImageObj(multiply(FMC01, conj(FMC02))))))

        # Store Scale from FMC algorithm
        scldisp[i] += fmcdisp[1]

        # Update Warping Matrix
        TRev[0, 0] = np.sqrt(1 / pow(MaxRad, -scldisp[i] / frame01.shape[1]))
        TRev[1, 1] = np.sqrt(1 / pow(MaxRad, -scldisp[i] / frame01.shape[1]))
        TRev[0, 2] = (1 - TRev[0, 0]) * frame01.shape[1] / 2 - dispX[i] / 2
        TRev[1, 2] = (1 - TRev[1, 1]) * frame01.shape[0] / 2 - dispY[i] / 2

        TFor[0, 0] = np.sqrt(1 * pow(MaxRad, -scldisp[i] / frame01.shape[1]))
        TFor[1, 1] = np.sqrt(1 * pow(MaxRad, -scldisp[i] / frame01.shape[1]))
        TFor[0, 2] = (1 - TFor[0, 0]) * frame01.shape[1] / 2 + dispX[i] / 2
        TFor[1, 2] = (1 - TFor[1, 1]) * frame01.shape[0] / 2 + dispY[i] / 2

        # Update iteration & error value
        errval = max([sqrt(trnsdisp[1]**2 + trnsdisp[0]**2), abs(fmcdisp[1])])
        iteration += 1

    print("Registering frame %03i, Iter %03i, DispX %03.2f, DispY %03.2f, Scale %03.3f, Error %03.3f"
              % (i, iteration, np.float(dispX[i]), np.float(dispY[i]),
                 pow(MaxRad, -scldisp[i] / frame01.shape[1]), errval))

    return scldisp, dispX, dispY, fr01, fr02
