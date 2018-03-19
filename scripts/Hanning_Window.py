# ----------------------------------------------------------------------------------------------------------------------
#   Name: Hanning_Window
#   Purpose: Signal processing to smooth values
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 18 22:14:12 2018
#   @author: brettmeyers
#   @author: jonholt
# ----------------------------------------------------------------------------------------------------------------------
from numpy import zeros
from scipy.signal import hanning


def hanningWindow(insize):
    hannWinX = zeros((1, insize[1]))
    hannWinY = zeros((insize[0], 1))
    hannWinX[0, :] = hanning(insize[1], sym=True)
    hannWinY[:, 0] = hanning(insize[0], sym=True)
    hannWin2D = hannWinY.dot(hannWinX)
    return hannWin2D