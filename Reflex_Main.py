# ----------------------------------------------------------------------------------------------------------------------
#   Name: Reflex_Main
#   Purpose: Main program for 'Reflex' algorithm
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 18 22:14:12 2018
#   @author: brettmeyers
#   @author: jonholt
# ----------------------------------------------------------------------------------------------------------------------

# functions to use for importing / exporting values
# From filename import func_name
# import filename
# import filename as func
# func.coolfun(arg)
# import settings_file as set
# Local_var = set.config_var

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

# from numba import jit
# from pdb import set_trace as keyboard
from numpy import where, array, mod, log, exp, stack, multiply, arange, sqrt, zeros, conj
from pdb import set_trace as keyboard
from pandas import rolling_median

# Importing subdirectory "scripts" files into main function
# from reflex_pupilometer_python.scripts.SubPixel2D import subPixel2D
# from reflex_pupilometer_python.scripts.Hampel import hampel
from reflex_pupilometer_python.scripts.Hanning_Window import hanningWindow
from reflex_pupilometer_python.scripts.Dilation_Register import spatialRegister

# Load Eye Cascade
eye_cascade = cv2.CascadeClassifier('/Users/JonHolt/Desktop/BL_Stuff/haarcascade_eye.xml')

pyfftw.interfaces.cache.enable()

    ###############################################################################
    ##################  LOAD VIDEO FROM CURRENT PWD  ##############################
    ###############################################################################


if __name__ == "__main__":

    # Set Image Directory
    filename = '/Users/JonHolt/Desktop/BL_Stuff/testVideo_1-2_compression.mp4'

    # filename = '/Users/brettmeyers/Desktop/from_S7/2018-01-06 15:32:44.mp4'
    # "Read" the video
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Get image properties for memory allocation purposes
    imgProp = vid.get_meta_data(index=None)

    # Set Frame Range to Evaluate
    frng = arange(0, imgProp['nframes'], 1)
    rmrng = zeros([imgProp['nframes'], 1])
    fps = vid.get_meta_data()['fps']

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
    # fft01FullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
    #                                 pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
    #                                 axes=(-2, -1),
    #                                 direction='FFTW_FORWARD',
    #                                 flags=('FFTW_MEASURE', ),
    #                                 threads=multiprocessing.cpu_count(),
    #                                 planning_timelimit=None)
    #
    # fft02FullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
    #                                 pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
    #                                 axes=(-2, -1),
    #                                 direction='FFTW_FORWARD',
    #                                 flags=('FFTW_MEASURE', ),
    #                                 threads=multiprocessing.cpu_count(),
    #                                 planning_timelimit=None)
    #
    # ifftFullImageObj = pyfftw.FFTW(pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
    #                                pyfftw.empty_aligned((yResample, xResample), dtype='complex128'),
    #                                axes=(-2, -1),
    #                                direction='FFTW_BACKWARD',
    #                                flags=('FFTW_MEASURE', ),
    #                                threads=multiprocessing.cpu_count(),
    #                                planning_timelimit=None)

    # Initialize displacement & scaling vectors
    dispX = zeros([imgProp['nframes'], 1])
    dispY = zeros([imgProp['nframes'], 1])
    scldisp = zeros([imgProp['nframes'], 1])
    iterthresh = float(25)
    errthresh = float(1E-1)

    # for i in frng:
    #     scldisp, dispX, dispY, fr01, fr02 = spatialRegister(i, vid.get_data(frng[0])[:, :, 0].reshape(yResample, xResample),
    #                                                         vid.get_data(i)[:, :, 0].reshape(yResample, xResample),
    #                                                         Win2D, fmcMaxRad, errthresh, iterthresh, dispX, dispY, scldisp)

    for i in frng:
        scldisp, dispX, dispY, fr01, fr02 = spatialRegister(i, np.max(vid.get_data(frng[0]).reshape(yResample, xResample, 3), 2),
                                                            np.max(vid.get_data(i).reshape(yResample, xResample, 3), 2),
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
        if len(eyes) == 0:
            # Store center locations and window sizes
            WinDims[i, 0] += peakLocs[0].mean()
            WinDims[i, 1] += peakLocs[1].mean()
            # WinDims[i, 2] += (eyes[0, 3]) / 2
            # WinDims[i, 3] += (eyes[0, 2]) / 2
            WinDims[i, 2] += 256
            WinDims[i, 3] += 256
        # Only use when skipping the haar classifier - FOR ARTIFICIAL DATA
        print("Detecting eye in Frame %03i, Y Center %03.2f, X Center %03.2f, Width %03i, Height %03i"
                  % (i, peakLocs[0].mean(), peakLocs[1].mean(), 256, 256))


    ###############################################################################
    #############     PUPIL DILATION OF REGISTERED IMAGES     #####################
    ###############################################################################
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

    # For single frame testing
    # for i in arange(11, 12, 1):

    for i in arange(1, len(frng) - 2, 1):
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
        current = cv2.warpAffine(vid.get_data(frng[i + 1]).reshape(yResample, xResample, 3),
                Tforward[0:2, :], (xResample, yResample), cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        reference = cv2.warpAffine(vid.get_data(frng[i - 1]).reshape(yResample, xResample, 3),
                Treverse[0:2, :], (xResample, yResample), cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

        # For ARTIFICAL eyes
        # HeightRange = np.arange((cropWin[0] - cropWin[2]), (cropWin[0] + cropWin[2]), 1).astype(int)
        # WidthRange = np.arange((cropWin[1] - cropWin[3]), (cropWin[1] + cropWin[3]), 1).astype(int)

        # For NON-ARTIFICIAL eyes
        HeightRange = np.arange((cropWin[0] - cropWin[2] / 2), (cropWin[0] + cropWin[2] / 2), 1).astype(int)
        WidthRange = np.arange((cropWin[1] - cropWin[3] / 2), (cropWin[1] + cropWin[3] / 2), 1).astype(int)

        curROI = current[HeightRange, :, :]
        curROI = curROI[:, WidthRange, :]

        refROI = reference[HeightRange, :, :]
        refROI = refROI[:, WidthRange, :]

        # Convert to HSV & Histogram Equalize
        curROI = cv2.cvtColor(curROI, cv2.COLOR_RGB2HSV)
        curroi = cv2.bitwise_not(cv2.equalizeHist(curROI[:, :, 2])).astype(float)
        refROI = cv2.cvtColor(refROI, cv2.COLOR_RGB2HSV)
        refroi = cv2.bitwise_not(cv2.equalizeHist(refROI[:, :, 2])).astype(float)

        sclPix, dispx, dispy, fr01, fr02 = spatialRegister(i, curroi, refroi, win2D, fmcmaxrad, errthresh,
                                                           iterthresh, dispx, dispy, sclPix)

    sclPix