#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:00:39 2017
@author: brettmeyers
"""
import numpy as np
from scipy import signal
from numba import jit
# Build Gaussian Filter
@jit
def gaussWindow(insize,sigma):
    gaussWinX       = np.zeros((1,insize[1]))
    gaussWinX[0,:]  = signal.gaussian(insize[1],std = sigma*insize[1],sym=True)
    gaussWinY       = np.zeros((insize[0],1))
    gaussWinY[:,0]  = signal.gaussian(insize[0],std = sigma*insize[0],sym=True)
    gaussWin2D      = gaussWinY.dot(gaussWinX)
    return gaussWin2D

# Build Hanning Filter
@jit
def hanningWindow(insize):
    hannWinX        = np.zeros((1,insize[1]))
    hannWinY        = np.zeros((insize[0],1))
    hannWinX[0,:]   = signal.hanning(insize[1], sym=True)
    hannWinY[:,0]   = signal.hanning(insize[0], sym=True)
    hannWin2D       = hannWinY.dot(hannWinX)
    return hannWin2D