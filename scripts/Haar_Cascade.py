# ----------------------------------------------------------------------------------------------------------------------
#   Name: Haar Cascade
#   Purpose: Finds image center & window size per frame
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 21 22:14:12 2018
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

from numpy import where, zeros


def detect_eye(cascade, T1, WinDims, vid, frng, fmcMaxRad, xResample, yResample, dispX, dispY, scldisp):
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
                                  cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
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
        eyes = cascade.detectMultiScale(curframe,
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

    return eyes
