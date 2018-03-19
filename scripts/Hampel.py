# ----------------------------------------------------------------------------------------------------------------------
#   Name: Hanning_Window
#   Purpose: Signal processing to smooth values
#            adapted from hampel function in R package pracma
#            x= 1-d numpy array of numbers to be filtered
#            k= number of items in window/2 (# forward and backward wanted to capture in median filter)
#            t0= number of standard deviations to use; 3 is default
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 18 22:14:12 2018
#   @author: brettmeyers
#   @author: jonholt
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


def hampel(x,k, t0=3):
    n = len(x)
    # y is the corrected series
    y = x
    L = 1.4826
    for i in range((k + 1), (n - k)):
        if np.isnan(x[(i - k):(i + k + 1)]).all():
            continue
        x0 = np.nanmedian(x[(i - k):(i + k + 1)])
        S0 = L * np.nanmedian(np.abs(x[(i - k):(i + k + 1)] - x0))
        if np.abs(x[i] - x0) > t0 * S0:
            y[i] = x0
    return y