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
import json

# from numba import jit
# from pdb import set_trace as keyboard
from numpy import where, array, mod, log, exp, stack, multiply, arange, sqrt, zeros, conj
from pdb import set_trace as keyboard
from pandas import rolling_median

# Importing subdirectory "scripts" files into main function
# from reflex_pupilometer_python.scripts.SubPixel2D import subPixel2D
# from reflex_pupilometer_python.scripts.Hampel import hampel
from scripts.Hanning_Window import hanningWindow
from scripts.Dilation_Register import spatialRegister
from scripts.Haar_Cascade import detect_eye


# Load Eye Cascade
cascade = cv2.CascadeClassifier('C:\\Users\Jonathan\Desktop\BL_Stuff\haarcascade_eye.xml')

pyfftw.interfaces.cache.enable()

###############################################################################
##################         MAIN FUNCTION         ##############################
###############################################################################


if __name__ == "__main__":

    # Set Image Directory
    filename = 'C:\\Users\Jonathan\Desktop\BL_Stuff\Video_1-2_compression.mp4'

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

    # Initialize displacement & scaling vectors
    dispX = zeros([imgProp['nframes'], 1])
    dispY = zeros([imgProp['nframes'], 1])
    scldisp = zeros([imgProp['nframes'], 1])
    iterthresh = float(25)
    errthresh = float(1E-1)

    for i in frng:
        scldisp, dispX, dispY, fr01, fr02 = spatialRegister(i, np.max(vid.get_data(frng[0]).reshape(yResample, xResample, 3), 2),
                                                            np.max(vid.get_data(i).reshape(yResample, xResample, 3), 2),
                                                            Win2D, fmcMaxRad, errthresh, iterthresh, dispX, dispY, scldisp)

    ###############################################################################
    #############     DETECT EYE IN REGISTERED IMAGES     #########################
    ###############################################################################

    # Initialize window size & window center vectors
    WinDims = zeros([imgProp['nframes'], 4], dtype=int)

    # Build Transform
    T1 = np.eye(3, dtype=float)

    # Run Haar Cascade classifier to find image center & window size per frame
    detect_eye(cascade, T1, WinDims, vid, frng, fmcMaxRad, xResample, yResample, dispX, dispY, scldisp)

    ###############################################################################
    #############     PUPIL DILATION OF REGISTERED IMAGES     #####################
    ###############################################################################

    cropWin = np.round(np.median(WinDims[frng, :], axis=0)).astype(int)
    timeVector = arange(0, imgProp['nframes'], 1) / vid.get_meta_data()['fps']
    timeStep = np.gradient(frng) / vid.get_meta_data()['fps']
    frmStep = np.gradient(frng)

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
    # for i in arange("desired frame", "next frame", 1):

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
                                 Tforward[0:2, :],
                                 (xResample, yResample),
                                 cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        reference = cv2.warpAffine(vid.get_data(frng[i - 1]).reshape(yResample, xResample, 3),
                                   Treverse[0:2, :],
                                   (xResample, yResample),
                                   cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        # For NON-ARTIFICIAL eyes
        HeightRange = np.arange((cropWin[0] - cropWin[2] / 2), (cropWin[0] + cropWin[2] / 2), 1).astype(int)
        WidthRange = np.arange((cropWin[1] - cropWin[3] / 2), (cropWin[1] + cropWin[3] / 2), 1).astype(int)

        curROI = current[HeightRange, :, :]
        curROI = curROI[:, WidthRange, :]

        refROI = reference[HeightRange, :, :]
        refROI = refROI[:, WidthRange, :]

        # Convert to HSV & Histogram Equalize
        curROI = cv2.cvtColor(curROI, cv2.COLOR_RGB2HSV)
        curroi = cv2.bitwise_not(cv2.equalizeHist(curROI[:, :, 2]))
        refROI = cv2.cvtColor(refROI, cv2.COLOR_RGB2HSV)
        refroi = cv2.bitwise_not(cv2.equalizeHist(refROI[:, :, 2]))

        sclPix, dispx, dispy, fr01, fr02 = spatialRegister(i, refroi, curroi, win2D, fmcmaxrad, errthresh,
                                                           iterthresh, dispx, dispy, sclPix)

    ###############################################################################
    ###################        OUTPUT PARAMETERS        ###########################
    ###############################################################################

    dilationRatio = pow(fmcmaxrad / 1, (np.cumsum(np.multiply(-sclPix[:, 0] / 2, frmStep)) / cropWin[3]))
    dilationVelocity = sclPix / 2
    constrictInd = np.where(dilationRatio == np.min(dilationRatio))
    maxConstrictTime = timeVector[constrictInd][0]
    magAcceleration = abs(np.gradient(sclPix[0: constrictInd[0][0]][:, 0]))
    pks = sp.signal.find_peaks_cwt(magAcceleration, np.arange(1, 5))
    quant50ind = np.where(magAcceleration[pks] >= np.median(magAcceleration[pks]))
    onsetInd = np.where(magAcceleration[pks[quant50ind]] == np.max(magAcceleration[pks[quant50ind]]))
    onsetInd = pks[quant50ind[0][onsetInd[0][0]]]
    onsetTime = timeVector[onsetInd]
    recoveryInd = np.where(dilationRatio[constrictInd[0][0]::] >= 0.75 * abs(1 - dilationRatio[constrictInd[0][0]])
                           + dilationRatio[constrictInd[0][0]])
    recoveryInd = constrictInd[0][0] + recoveryInd[0][0]
    recoveryTime = timeVector[recoveryInd]
    averageConstriction = np.trapz(sclPix[onsetInd:constrictInd[0][0]], axis=0)[0] / (constrictInd[0][0] - onsetInd)
    averageDilation = np.trapz(sclPix[constrictInd[0][0]:recoveryInd], axis=0)[0] / (constrictInd[0][0] - recoveryInd)

    # DEBUG POINT
    # sclPix

parameters = json.dumps({'Time to React': str(onsetTime),
                         'Time to Recover': str(recoveryTime),
                         'Dilation Magnitude': str(np.min(dilationRatio)),
                         'Time to Minimum Constriction': str(constrictInd[0][0]),
                         'Dilation Speed': str(averageDilation)},
                        sort_keys=True, indent=4, separators=(',', ': '))

with open(r'C:\\Users\Jonathan\Desktop\BL_Stuff\ReflexOutput.json', 'w') as outfile:
    outfile.write(parameters)
