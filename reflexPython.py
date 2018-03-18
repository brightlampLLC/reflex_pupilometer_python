#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:32:54 2018

@author: brettmeyers
@editor: jonholt
"""
    ###############################################################################
    ###################  IMPORT RELEVANT LIBRARIES   ##############################
    ###############################################################################
DEBUG = True
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
import traceback

# from numba import jit
# from pdb import set_trace as keyboard
from numpy import where, array, mod, log, exp, stack, multiply, arange, sqrt, zeros, conj
from scipy.signal import hanning
from pdb import set_trace as keyboard
from pandas import rolling_median

# Load Eye Cascade
eye_cascade = cv2.CascadeClassifier('/Users/JonHolt/Desktop/BL_Stuff/haarcascade_eye.xml')

    ###############################################################################
    ###################  TRACEBACKS FOR DEBUGGING    ##############################
    ###############################################################################


def trace(verbose, log_file=""):
    exc_info = sys.exc_info()[2]
    format_tb = traceback.format_tb(exc_info)[0]
    errors = "Py Errors:\nTb Info:\n {} \n Error Info:\n {}: {} \n ".format(format_tb,
                          str(sys.exc_info()[0]),
                          str(sys.exc_info()[1]))

    if log_file:
        o = open(log_file, 'a')
        o.write(errors + "\n\n\n")
        o.close()

    if verbose:
        print(errors)
    return errors

pyfftw.interfaces.cache.enable()
#@numba.jit(numba.float64(numba.float64), nopython=False, nogil=True,\
#           cache=True, forceobj=False, locals={})


def subPixel2D(plane):
    try:
        # Find location of peak in correlation
        peakLocs = where(plane == plane.max())
        subPixOffset = array([0, 0])

        if mod(plane.shape[0], 2) != 0:
            subPixOffset[0] -= 0.5

        if mod(plane.shape[1], 2) != 0:
            subPixOffset[1] -= 0.5

        disp = array([(plane.shape[0] / 2) - peakLocs[0][0] + subPixOffset[0],
                        (plane.shape[1] / 2) - peakLocs[1][0] + subPixOffset[1]])

        if all([peakLocs[0][0] <= plane.shape[0] - 1, peakLocs[1][0] <= plane.shape[1] - 1,
            peakLocs[0][0] >= 2, peakLocs[1][0] >= 2]):
            disp[1] -= (log(plane[peakLocs[0][0] - 0, peakLocs[1][0]-1]) - log(plane[peakLocs[0][0] + 0,
                        peakLocs[1][0] + 1])) / (2 * (log(plane[peakLocs[0][0] - 0,
                        peakLocs[1][0] - 1]) + log(plane[peakLocs[0][0] + 0,
                        peakLocs[1][0] + 1]) - 2 * log(plane[peakLocs[0][0] - 0, peakLocs[1][0] - 0])))

            disp[0] -= (log(plane[peakLocs[0][0] - 1, peakLocs[1][0] - 0]) - log(plane[peakLocs[0][0] + 1,
                        peakLocs[1][0] + 0])) / (2 * (log(plane[peakLocs[0][0] - 1,
                        peakLocs[1][0] - 0]) + log(plane[peakLocs[0][0] + 1,
                        peakLocs[1][0] + 0]) - 2 * log(plane[peakLocs[0][0] - 0, peakLocs[1][0] - 0])))
        return disp
    except:
        trace(True)
        if DEBUG:
            input("Press enter to continue...")

# Build sp.signal.hanning Filter
#@jit


def hanningWindow(insize):
    try:
        hannWinX = zeros((1, insize[1]))
        hannWinY = zeros((insize[0], 1))
        hannWinX[0, :] = hanning(insize[1], sym=True)
        hannWinY[:, 0] = hanning(insize[0], sym=True)
        hannWin2D = hannWinY.dot(hannWinX)
        return hannWin2D
    except:
        trace(True)
        if DEBUG:
            input("Press enter to continue...")


def hampel(x,k, t0=3):
    try:
        '''adapted from hampel function in R package pracma
        x= 1-d numpy array of numbers to be filtered
        k= number of items in window/2 (# forward and backward wanted to capture in median filter)
        t0= number of standard deviations to use; 3 is default
        '''
        n = len(x)
        y = x #y is the corrected series
        L = 1.4826
        for i in range((k + 1), (n - k)):
            if np.isnan(x[(i - k):(i + k + 1)]).all():
                continue
            x0 = np.nanmedian(x[(i - k):(i + k + 1)])
            S0 = L * np.nanmedian(np.abs(x[(i - k):(i + k + 1)] - x0))
            if (np.abs(x[i] - x0) > t0 * S0):
                y[i] = x0
        return y
    except:
        trace(True)
        if DEBUG:
            input("Press enter to continue...")

#@numba.jit(numba.types.UniTuple(numba.float64[:],3)(numba.int64,numba.uint8[:,:],\
#           numba.float64,numba.float64,numba.complex128[:,:],numba.complex128[:,:],\
#           numba.float64[:],numba.float64[:],numba.float64[:]), nopython=False, nogil=True,\
#           cache=True, forceobj=False, locals={})


def spatialRegister(i, frame01, frame02, Win2D, MaxRad, errthresh, iterthresh, dispX, dispY, scldisp):
    try:
        # initialize iterations & error
        errval = float(100)
        iteration = 0
        TRev = np.eye(3, dtype=float)
        TFor = np.eye(3, dtype=float)

        while all((errval > errthresh, iteration < iterthresh)):
            # Reconstruct images based on transform matrices
            fr01 = cv2.warpAffine(frame01.astype(float), TFor[0:2, :], (xResample, yResample),\
                                  cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS).astype(float)
            fr01 = np.nan_to_num(fr01)
            fr01 -= fr01.mean(axis=(0, 1))
            fr01 = multiply(fr01, Win2D)
            fr02 = cv2.warpAffine(frame02.astype(float), TRev[0:2, :], (xResample, yResample),\
                                  cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS).astype(float)
            fr02 = np.nan_to_num(fr02)
            fr02 -= fr02.mean(axis=(0, 1))
            fr02 = multiply(fr02, Win2D)

            # Calculate FFTs for image pair
            FFT01 = sp.fftpack.fftshift(fft01FullImageObj(fr01))
            FFT02 = sp.fftpack.fftshift(fft02FullImageObj(fr02))

            # Run FMT on FFTs
            FMT01 = multiply(Win2D, cv2.logPolar(abs(FFT01).astype(float), (FFT01.shape[1]/2, FFT01.shape[0]/2),\
                   FFT01.shape[1]/log(MaxRad), flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)).astype(float)
            FMT02 = multiply(Win2D, cv2.logPolar(abs(FFT02).astype(float), (FFT02.shape[1]/2, FFT02.shape[0]/2),\
                   FFT02.shape[1]/log(MaxRad), flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)).astype(float)

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

        print("Registering frame %03i, Iter %03i, DispX %03.2f, DispY %03.2f, Scale %03.3f, Error %03.3f"\
                  % (i, iteration, np.float(dispX[i]), np.float(dispY[i]),\
                     pow(MaxRad, -scldisp[i] / frame01.shape[1]), errval))

        return scldisp, dispX, dispY, fr01, fr02
    except:
        trace(True)
        if DEBUG:
            input("Press enter to continue...")


# def dilationEstimator(i, subreg01, subreg02, win2D, maxrad, errthresh, iterthresh, dispx, dispy, sclPix):
#     try:
#         # initialize iterations & error
#         errval = float(100)
#         iteration = 0
#         Trev = np.eye(3, dtype=float)
#         Tfor = np.eye(3, dtype=float)
#
#         while all((errval > errthresh, iteration < iterthresh)):
#             # Reconstruct images based on transform matrices
#             sr01 = cv2.warpAffine(subreg01.astype(float), Tfor[0:2, :], (subreg01.shape[1], subreg01.shape[0]),\
#                                   cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS).astype(float)
#             sr01 = np.nan_to_num(sr01)
#             sr01 -= sr01.mean(axis=(0, 1))
#             sr01 = multiply(sr01, win2D)
#             sr02 = cv2.warpAffine(subreg02.astype(float), Trev[0:2, :], (subreg01.shape[1], subreg01.shape[0]),\
#                                   cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS).astype(float)
#             sr02 = np.nan_to_num(sr02)
#             sr02 -= sr02.mean(axis=(0, 1))
#             sr02 = multiply(sr02, win2D)
#
#             # Calculate FFTs for image pair
#             fft01 = sp.fftpack.fftshift(fft01FMCObj(sr01))
#             fft02 = sp.fftpack.fftshift(fft02FMCObj(sr02))
#
#             # Run FMT on FFTs
#             fmt01 = multiply(win2D, cv2.logPolar(abs(fft01).astype(float), (fft01.shape[1] / 2, fft01.shape[0] / 2),\
#                    fft01.shape[1] / log(maxrad), flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)).astype(float)
#             fmt02 = multiply(win2D, cv2.logPolar(abs(fft02).astype(float), (fft02.shape[1] / 2, fft02.shape[0] / 2),\
#                    fft02.shape[1] / log(maxrad), flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)).astype(float)
#
#             # Calculate FFTs of FMTs
#             fmc01 = fft01FMCObj(fmt01)
#             fmc02 = fft02FMCObj(fmt02)
#
#             # Run Translation Subfunc
#             trnsdisp = subPixel2D(abs(sp.fftpack.fftshift(ifftFMCObj(multiply(fft01, conj(fft02))))))
#
#             # Store new displacement
#             dispx[i] += trnsdisp[1]
#             dispy[i] += trnsdisp[0]
#
#             # Run Scaling Subfunc
#             fmcdisp = subPixel2D(abs(sp.fftpack.fftshift(ifftFMCObj(multiply(fmc01, conj(fmc02))))))
#
#             # Store Scale from FMC algorithm
#             sclPix[i] += fmcdisp[1]
#
#             # Update Warping Matrix
#             Trev[0, 0] = np.sqrt(1 / pow(maxrad, -sclPix[i] / subreg01.shape[1]))
#             Trev[1, 1] = np.sqrt(1 / pow(maxrad, -sclPix[i] / subreg01.shape[1]))
#             Trev[0, 2] = (1 - Trev[0, 0]) * subreg01.shape[1] / 2 - dispx[i] / 2
#             Trev[1, 2] = (1 - Trev[1, 1]) * subreg01.shape[0] / 2 - dispy[i] / 2
#
#             Tfor[0, 0] = np.sqrt(1 * pow(maxrad, -sclPix[i] / subreg02.shape[1]))
#             Tfor[1, 1] = np.sqrt(1 * pow(maxrad, -sclPix[i] / subreg02.shape[1]))
#             Tfor[0, 2] = (1 - Tfor[0, 0])*subreg02.shape[1] / 2 + dispx[i] / 2
#             Tfor[1, 2] = (1 - Tfor[1, 1])*subreg02.shape[0] / 2 + dispy[i] / 2
#
#             # Update iteration & error value
#             errval = max([sqrt(trnsdisp[1]**2 + trnsdisp[0]**2), abs(fmcdisp[1])])
#             iteration += 1
#         print("Registering frame %03i, Iter %03i, DispX %03.2f, DispY %03.2f, Scale %03.3f, Error %03.3f"\
#                   % (i, iteration, np.float(dispx[i]), np.float(dispy[i]), pow(maxrad, -sclPix[i]/subreg01.shape[1]), errval))
#         return sclPix, dispx, dispy, sr01, sr02
#     except:
#         trace(True)
#         if DEBUG:
#             input("Press enter to continue...")

    ###############################################################################
    ##################  LOAD VIDEO FROM CURRENT PWD  ##############################
    ###############################################################################


if __name__ == "__main__":

    # Set Image Directory
    filename = '/Users/JonHolt/Desktop/BL_Stuff/testVideo_1-1_compression.mp4'

    # filename = '/Users/brettmeyers/Desktop/from_S7/2018-01-06 15:32:44.mp4'
    # "Read" the video
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get image properties for memory allocation purposes
    imgProp = vid.get_meta_data(index=None)

    # Set Frame Range to Evaluate
    frng = arange(0, imgProp['nframes'], 1)
    rmrng = zeros([imgProp['nframes'], 1])
    fps = vid.get_meta_data()['fps']

    # Identify over-saturated frames, build frame series
    for i in frng:
        rmrng[i] = vid.get_data(i).mean()/255
    frng = np.delete(frng, (np.r_[0:np.round(0.8 * vid.get_meta_data()['fps']),
                            where((rmrng >= 0.70))[0][-2],
                            where((rmrng >= 0.70))[0][-1],
                            where((rmrng >= 0.70))[0][-1],
                            np.round(5.8 * vid.get_meta_data()['fps']):imgProp['nframes']]))

    ###############################################################################
    ##################    RUN IMAGE REGISTRATION     ##############################
    ###############################################################################

    # Image & FMC Parameters
    xResample = imgProp['size'][1]
    yResample = imgProp['size'][0]
    fmcMinRad = 1
    fmcMaxRad = np.min([yResample, xResample]) / 2
    fmcNoOfRings = xResample
    fmcNoOfWedges = yResample
    Win2D = hanningWindow([yResample, xResample])

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

    # Initialize displacement & scaling vectors
    dispX = zeros([imgProp['nframes'], 1])
    dispY = zeros([imgProp['nframes'], 1])
    scldisp = zeros([imgProp['nframes'], 1])
    iterthresh = float(25)
    errthresh = float(1E-1)

    for i in frng:
        scldisp, dispX, dispY, fr01, fr02 = spatialRegister(i, vid.get_data(frng[0])[:, :, 0].reshape(yResample, xResample),
                                                            vid.get_data(i)[:, :, 0].reshape(yResample, xResample),
                                                            Win2D, fmcMaxRad, errthresh, iterthresh, dispX, dispY, scldisp)

    ###############################################################################
    #############     DETECT EYE IN REGISTERED IMAGES     #########################
    ###############################################################################

    # Initialize window size & window center vectors
    WinDims = zeros([imgProp['nframes'], 4], dtype=int)

    # Run Haar Cascade classifier to find image center & window size per frame
    # Build Transform
    T1 = np.eye(3, dtype=float)

    for i in frng:
        # Build Transform
        T1[0, 0] = pow(fmcMaxRad, -scldisp[i] / xResample)
        T1[1, 1] = pow(fmcMaxRad, -scldisp[i] / xResample)
        T1[0, 2] = (1 - T1[0, 0]) * xResample / 2 + dispX[i]
        T1[1, 2] = (1 - T1[1, 1]) * yResample / 2 + dispY[i]

        # Transform current frame to reference position, convert to 8U
        curframe = cv2.warpAffine(vid.get_data(i).reshape(yResample, xResample, 3),
                                  T1[0:2, :],
                                  (xResample, yResample),
                                  cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        curframe[np.where(curframe == 0)] = 255
        blurframe = cv2.bitwise_not(cv2.GaussianBlur(curframe.max(axis=-1), (109, 109), 11))
        peakLocs = where(blurframe == blurframe.max())

        # Run Haar Cascade classifier - https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
        # DOES NOT WORK WITH ARTIFICIAL DATA (i.e. FAKE EYES)
        #   Parameters:
        #       cascade – Haar classifier cascade
        #       image – Matrix of the type CV_8U containing an image where objects are detected.
        #       objects – Vector of rectangles where each rectangle contains the detected object.
        #       scaleFactor – Specifies how much the image size is reduced at each image scale.
        #       minNeighbors – Specifies how many neighbors each candidate rectangle should have to retain it.
        #       flags – Parameter with the same meaning for an old cascade. Not used for a new cascade.
        #       minSize – Minimum possible object size. Objects smaller than that are ignored.
        #       maxSize – Maximum possible object size. Objects larger than that are ignored.
        # Example:
        #   cv2.CascadeClassifier.detectMultiScale(image[,
        #                                            scaleFactor[,
        #                                            minNeighbors[,
        #                                            flags[,
        #                                            minSize[,
        #                                            maxSize]]]]])
        eyes = eye_cascade.detectMultiScale(curframe,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            flags=0,
                                            minSize=(int(np.min([yResample, xResample]) / 4),
                                                     int(np.min([yResample, xResample]) / 4)))

        # If eye is found, use Gaussian fitting function to find pupil center
        if len(eyes) != 0:
            # Store center locations and window sizes
            WinDims[i, 0] += peakLocs[0].mean()
            WinDims[i, 1] += peakLocs[1].mean()
            WinDims[i, 2] += (eyes[0, 3]) / 2
            WinDims[i, 3] += (eyes[0, 2]) / 2

        # print("Detecting eye in Frame %03i, Y Center %03.2f, X Center %03.2f, Width %03i, Height %03i"
        #           % (i, peakLocs[0].mean(), peakLocs[1].mean(), eyes[0, 3] / 2, eyes[0, 2] / 2))

        # Only use when skipping the haar classifier - FOR ARTIFICIAL DATA
        print("Detecting eye in Frame %03i, Y Center %03.2f, X Center %03.2f, Width %03i, Height %03i"
                  % (i, peakLocs[0].mean(), peakLocs[1].mean(), 64, 64))

    ###############################################################################
    #############     PUPIL DILATION OF REGISTERED IMAGES     #####################
    ###############################################################################
    try:
        cropWin = np.round(np.median(WinDims[frng, :], axis=0)).astype(int)
        timeVector = arange(0, imgProp['nframes'], 1) / vid.get_meta_data()['fps']
        timeStep = np.gradient(frng) / vid.get_meta_data()['fps']
        frmStep = np.gradient(frng)

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

        win2D = hanningWindow([cropWin[2], cropWin[3]])
        Tforward = np.eye(3, dtype=float)
        Treverse = np.eye(3, dtype=float)
        fmcmaxrad = np.min([cropWin[2], cropWin[3]]) / 2

        # Load Reference Image
        fr01 = zeros([cropWin[2], cropWin[3]])
        fr02 = zeros([cropWin[2], cropWin[3]])
        iterthresh = float(100)
        errthresh = float(1E-2)
        dispx = zeros([imgProp['nframes'], 1])
        dispy = zeros([imgProp['nframes'], 1])
        sclPix = zeros([imgProp['nframes'], 1])

        for i in arange(1, len(frng)-2, 1):
            # Build forward transform
            Tforward[0, 0] = pow(fmcMaxRad, -scldisp[frng[i + 1]] / xResample)
            Tforward[1, 1] = pow(fmcMaxRad, -scldisp[frng[i + 1]] / xResample)
            Tforward[0, 2] = (1 - Tforward[0, 0])*xResample / 2 + dispX[frng[i + 1]]
            Tforward[1, 2] = (1 - Tforward[1, 1])*yResample / 2 + dispY[frng[i + 1]]

            # Build reverse transform
            Treverse[0, 0] = pow(fmcMaxRad, -scldisp[frng[i - 1]] / xResample)
            Treverse[1, 1] = pow(fmcMaxRad, -scldisp[frng[i - 1]] / xResample)
            Treverse[0, 2] = (1 - Treverse[0, 0])*xResample / 2 + dispX[frng[i - 1]]
            Treverse[1, 2] = (1 - Treverse[1, 1])*yResample / 2 + dispY[frng[i - 1]]

            # Transform reference & current frame to reference position, convert to 8U
            current = cv2.warpAffine(vid.get_data(frng[i+1]).reshape(yResample, xResample, 3),
                    Tforward[0:2, :], (xResample, yResample), cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
            reference = cv2.warpAffine(vid.get_data(frng[i-1]).reshape(yResample, xResample, 3),
                    Treverse[0:2, :], (xResample, yResample), cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

            # Crop regions using window elements identified in previous step
            curROI = current[(cropWin[0] - cropWin[2] / 2):(cropWin[0] + cropWin[2] / 2),
                                    (cropWin[1] - cropWin[3] / 2):(cropWin[1]+cropWin[3] / 2), :]
            refROI = reference[(cropWin[0] - cropWin[2]/2):(cropWin[0] + cropWin[2] / 2),
                                    (cropWin[1] - cropWin[3] / 2):(cropWin[1]+cropWin[3] / 2), :]

            # Convert to HSV & Histogram Equalize
            curROI = cv2.cvtColor(curROI, cv2.COLOR_RGB2Lab)
        #    curROI[:,:,0] = clahe.apply(curROI[:, :, 0])
        #    curROI[:,:,0] = cv2.equalizeHist(curROI[:, :, 0])
        #    curROI  = cv2.cvtColor(curROI, cv2.COLOR_HSV2RGB)
            curroi = cv2.bitwise_not(cv2.equalizeHist(curROI[:, :, 0])).astype(float)
            refROI = cv2.cvtColor(refROI, cv2.COLOR_RGB2Lab)
        #    refROI[:,:,0] = clahe.apply(refROI[:,:,0])
        #    refROI[:,:,0] = cv2.equalizeHist(refROI[:,:,0])
        #    refROI  = cv2.cvtColor(refROI,cv2.COLOR_HSV2RGB)
            refroi = cv2.bitwise_not(cv2.equalizeHist(refROI[:, :, 0])).astype(float)

            for i in frng:
                sclPix, dispX, dispY, fr01, fr02 = spatialRegister(i, curroi, refroi, Win2D, fmcMaxRad, errthresh,
                            iterthresh, dispX, dispY, scldisp)
    except:
        trace(True)
        if DEBUG:
            input("Press enter to continue...")

    #    errval      = float(100)
    #    iteration   = 0
    #    Tf = np.eye(3, dtype=float)
    #    while all((errval > errthresh, iteration < iterthresh)):
    #        # Load Current Image and Warp first image
    #        fr01        = refroi
    #        fr01       -= fr01.mean(axis=(0,1))
    #        fr01        = multiply(fr01,win2D)
    #        fr02        = cv2.warpAffine(curroi,Tf[0:2,:],(curroi.shape[1],curroi.shape[0]),\
    #                              cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS).astype(float)
    #        fr02       -= fr02.mean(axis=(0,1))
    #        fr02        = multiply(fr02,win2D)
    #
    #        FFT01       = sp.fftpack.fftshift(fft01FMCObj(fr01))
    #        FMC01       = multiply(win2D,cv2.logPolar(abs(FFT01),(FFT01.shape[1]/2,FFT01.shape[0]/2),\
    #               FFT01.shape[1]/log(FFT01.shape[1]),flags = cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS))
    #        FMT01       = fft01FMCObj(FMC01)
    #        FMT01       = sp.fftpack.fftshift(FMT01)
    #        FFT02       = sp.fftpack.fftshift(fft02FMCObj(fr02))
    #        FMC02       = multiply(win2D,cv2.logPolar(abs(FFT02),(FFT02.shape[1]/2,FFT02.shape[0]/2),\
    #               FFT02.shape[1]/log(FFT02.shape[1]),flags = cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS))
    #        FMT02       = fft02FMCObj(FMC02)
    #        FMT02       = sp.fftpack.fftshift(FMT02)
    #        # Run Scaling Subfunc
    #        fmcdisp     = subPixel2D(abs(sp.fftpack.fftshift(ifftFMCObj(multiply(FMT01,conj(FMT02))))))
    #        # Store Scale from FMC algorithm
    #        sclPix[frng[i]]   -= fmcdisp[1]
    #        # Update iteration & error value
    #        errval      = abs(fmcdisp[1])
    #        # Update Warping Matrix
    #        Tf[0,0]     = pow(FFT02.shape[1],-sclPix[frng[i]]/(FFT02.shape[1]))
    #        Tf[1,1]     = pow(FFT02.shape[1],-sclPix[frng[i]]/(FFT02.shape[1]))
    #        iteration  += 1
    #    print("Dilation of frame %03i, Iter %03i, Scale %03.3f, Error %03.5f" % \
    #          (frng[i],iteration,pow(FFT01.shape[1],-sclPix[frng[i]]/FFT01.shape[1]),fmcdisp[1]))

    #    fig = plt.figure(1)
    #    plt.imshow(refroi, cmap='Greens', interpolation='nearest')
    #    plt.imshow(curroi, cmap='Purples', alpha=.4, interpolation='nearest')
    #    plt.show()
    #    plt.pause(1E-5)

    #    sclPix = dilationEstimator(frng[i],sclPix,curroi,refroi,win2D,errthresh,iterthresh)
    #    sclPix[frng[i]] = sclPix[frng[i]]/(2*frmStep[i])
    #    #%%
    ## Filter data for erroneous measurements
    #sclPixfilt = hampel(sclPix[frng], 4, 3)
    ## Plot Time Series response
    #fig = pylab.figure(5)
    #pylab.plot(timeVector[frng[1::]],pow(2*cropWin[2],sp.integrate.cumtrapz(sclPixfilt[:,0])/(2*cropWin[2])))
    #pylab.show()
    ## Write Time Series to File
    #np.savetxt('data.txt', np.array([arange(0,len(frng[1::]),1),timeVector[frng[1::]],\
    #    pow(2*cropWin[2],sp.integrate.cumtrapz(sclPixfilt[:,0])/(2*cropWin[2]))]).T,\
    #    fmt='%03i  %1.5f  %1.5f', delimiter=' ', newline='\n', header='   Time[s]  Dilation Ratio')
    #    #%%
    ## Crop regions using window elements identified in previous step
    #curROI  = current[(cropWin[0]-cropWin[2]):(cropWin[0]+cropWin[2]),\
    #                        (cropWin[1]-cropWin[3]):(cropWin[1]+cropWin[3]),:]
    #curROI  = cv2.cvtColor(curROI,cv2.COLOR_RGB2Lab)
    #clahe   = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(12,12))
    #curROI[:,:,0] = clahe.apply(curROI[:,:,0])
    #curROI[:,:,0] = cv2.equalizeHist(curROI[:,:,0])
    #fig = pylab.figure(6)
    #fig.suptitle('HSV Equalize', fontsize=20)
    #pylab.imshow(curROI[:,:,0])
    #pylab.show()
    ##%%
    #fps = vid.get_meta_data()['fps']
    #writer = imageio.get_writer('/Users/brettmeyers/Desktop/registered.mp4', fps=fps)
    ## Show/Write Registered Video
    #for i in frng:
    #    #    # Build Transform
    #    T1 = np.eye(3, dtype=float)
    #    T1[0,0]     = pow(fmcMaxRad,-scldisp[i]/xResample)
    #    T1[1,1]     = pow(fmcMaxRad,-scldisp[i]/xResample)
    #    T1[0,2]     = (1-T1[0,0])*xResample/2 - np.float(dispX[i])
    #    T1[1,2]     = (1-T1[1,1])*yResample/2 - np.float(dispY[i])
    #    curframe = cv2.warpAffine(vid.get_data(i).reshape(yResample,xResample,3),\
    #                              T1[0:2,:],(xResample,yResample),cv2.INTER_LINEAR)
    #    writer.append_data(curframe)
    #    fig = pylab.figure(5)
    #    fig.suptitle('image #{}'.format(i), fontsize=20)
    #    pylab.imshow(curframe)
    #    pylab.show()
    #    pylab.pause(1E-2)
    #    pylab.clf()
    #writer.close()
    #
    ##%%
    #img01  = zeros([yResample,xResample])
    #img02  = zeros([yResample,xResample])
    #img01[831:1088,412:668] = 255
    #img02[839:1099,404:664] = 255
    #
    #scldisp, dispX, dispY, fr01, fr02 = spatialRegister(0,np.uint8(img02),\
    #                np.uint8(img01),Win2D,1E-3, \
    #                25,dispX,dispY,scldisp)
    ##%%
    #img01  = zeros([cropWin[2],cropWin[3]])
    #img02  = zeros([cropWin[2],cropWin[3]])
    #img01[137:201,137:201] = 255
    #img02[131:200,139:207] = 255
    #
    #sclPix, dispx, dispy, fr01, fr02 = dilationEstimator(0,img02,img01,win2D,fmcmaxrad,errthresh, \
    #                    iterthresh,dispx,dispy,sclPix)