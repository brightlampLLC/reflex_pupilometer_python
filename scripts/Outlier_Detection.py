# ----------------------------------------------------------------------------------------------------------------------
#   Name: Outlier_Detection
#   Purpose: Detects and removes outliers from timeseries data
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 21 23:59:43 2018
#   @author: brettmeyers
#   @author: jonholt
# ----------------------------------------------------------------------------------------------------------------------

import sys
import os
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

from scipy import interpolate


def velocityThreshold(A, threshold):
    B = A
    B[abs(A) >= threshold] = np.nan
    return B


def movingMAD(A, iterations, windowSize, threshold):
    windowRadius = np.floor(windowSize / 2).astype(int)
    for i in np.arange(0, len(A), 1):
        if A[i] != np.nan:
            leftInd = i - windowRadius
            rightInd = i + windowRadius
            if leftInd > 0:
                leftVals = A[-leftInd::]
                rightVals = A[0:rightInd]
                vals = np.concatenate((leftVals, rightVals), axis=0)
            elif rightInd > len(A):
                leftVals = A[leftInd::]
                rightVals = A[0:(rightInd - len(A) - 1)]
                vals = np.concatenate((leftVals, rightVals), axis=0)
            else:
                vals = A[leftInd:rightInd]
            block = np.setdiff1d(vals, np.array(A[i]))
            if block.size:
                zScore = abs(A[i] - np.nanmedian(block)) / np.nanmedian(np.abs(block-np.nanmedian(block))) + 0.1
                if zScore > threshold:
                    A[i] = np.nan
    return A


def velocityInterpolate(A):
    # interpolate to fill nan values
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good[0]], A[good[0], 0], bounds_error=False)
    B = np.where(np.isfinite(A[:, 0]), A[:, 0], f(inds))
    return B