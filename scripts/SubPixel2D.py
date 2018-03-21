# ----------------------------------------------------------------------------------------------------------------------
#   Name: SubPixel2D
#   Purpose: Finds location of peaks in the correlation
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

from numpy import where, array, mod, log


def subPixel2D(plane):
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