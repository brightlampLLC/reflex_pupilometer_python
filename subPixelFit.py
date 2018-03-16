#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:10:16 2017
@author: brettmeyers
"""
from numpy import where, array, mod, log
import numba
# Code adopeted from prana
@numba.jit(numba.float64(numba.float64), nopython=False, nogil=True,\
           cache=True, forceobj=False, locals={})
def subPixel2D(plane):
    # Find location of peak in correlation    
    peakLocs        = where(plane == plane.max())
    subPixOffset    = array([0,0])
    if mod(plane.shape[0],2) !=0:
        subPixOffset[0]  -= 0.5
    if mod(plane.shape[1],2) !=0:
        subPixOffset[1]  -= 0.5
    disp = array([peakLocs[0][0]-(plane.shape[0]/2)-subPixOffset[0], \
                     peakLocs[1][0]-(plane.shape[1]/2)-subPixOffset[1]])
    if all([peakLocs[0][0] <= plane.shape[0]-1,peakLocs[1][0] <= plane.shape[1]-1, \
        peakLocs[0][0] >= 2,peakLocs[1][0] >= 2]):
        disp[1]    += ( log(plane[peakLocs[0][0]-0,peakLocs[1][0]-1]) - \
            log(plane[peakLocs[0][0]+0,peakLocs[1][0]+1]) )/( 2* \
                  ( log(plane[peakLocs[0][0]-0,peakLocs[1][0]-1]) + \
                   log(plane[peakLocs[0][0]+0,peakLocs[1][0]+1]) - 2 * \
                         log(plane[peakLocs[0][0]-0,peakLocs[1][0]-0]) ))

        disp[0]    += ( log(plane[peakLocs[0][0]-1,peakLocs[1][0]-0]) - \
            log(plane[peakLocs[0][0]+1,peakLocs[1][0]+0]) )/( 2* \
                  ( log(plane[peakLocs[0][0]-1,peakLocs[1][0]-0]) + \
                   log(plane[peakLocs[0][0]+1,peakLocs[1][0]+0]) - 2 * \
                         log(plane[peakLocs[0][0]-0,peakLocs[1][0]-0]) ))       
    return disp