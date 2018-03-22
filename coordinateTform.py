#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:24:50 2017
@author: brettmeyers
"""
import cv2
import numpy as np
import scipy as sp
from pdb import set_trace as keyboard
# Cartesian to Polar Grid transform
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
# Polar to Cartesian Grid transform
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
# Log Polar Grid
def LPGrid(nx,ny,rmin,rmax,nwedges,nring):
    # Zero Center for LP Transform
    xZero = (nx+0.5)/2
    yZero = (ny+0.5)/2
    # Log R coordinate space
    rv = np.exp( np.linspace( np.log(rmin), np.log(rmax), nring))
    # Angular coordinate vector
    thv = np.linspace(0, 2*np.pi * (1 - 1 / nwedges), nwedges)
    # Log-polar grid
    r, th = np.meshgrid(rv, thv)
    # Convert log polar grid to cartesian.
    x, y = pol2cart(r, th)
    # Shift the center of the log-polar grid to the zero-frequency
    x = x + xZero
    y = y + yZero
    return (x,y)
# Transform image based on warping
def tformImg(x,y,M,nX,nY,img):
    # Build vector of zeros (Z)
    X = np.ones([nX*nY,1])
    Y = np.ones([nX*nY,1])
    Z = np.ones([nX*nY,1])
    # Flatten arrays
    X[:,0] = x.flatten('C')
    Y[:,0] = y.flatten('C')
    # Build Nx3 matrix [X,Y,Z]
    inputPoints = np.column_stack((X,Y,Z)).T
    interpPoints= np.linalg.lstsq(M, inputPoints)[0] 
    interpImg = np.empty_like(x)
    X = np.reshape(interpPoints[0,:],(nY,nX))
    Y = np.reshape(interpPoints[1,:],(nY,nX))
    sp.ndimage.interpolation.map_coordinates(img, [Y+nY/2-0.5,X+nX/2-0.5],\
                                             output=interpImg, order = 1,\
                                             mode='constant', cval=0.0,\
                                             prefilter=True)
    return interpImg 
# Log Polar Transform
def tformLP(x,y,xLP,yLP,nX,nY,img):
    interpImg = np.empty_like(xLP)
    sp.ndimage.interpolation.map_coordinates(img, [yLP,xLP],\
                                             output=interpImg, order = 1,\
                                             mode='constant', cval=0.0,\
                                             prefilter=True)
    return interpImg