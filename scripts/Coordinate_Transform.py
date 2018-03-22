# ----------------------------------------------------------------------------------------------------------------------
#   Name: Coordinate_Transform
#   Purpose: Creates coordination transform - replaces "cv.logpolar" call in Dilation_Register
#
#   !/usr/scripts/env python3
#   -*- coding: utf-8 -*-
#
#   Created on Tue Mar 21 23:59:43 2018
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


def cart2pol(x, y):
    # Cartesian to Polar Grid transform
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi


def pol2cart(rho, phi):
    # Polar to Cartesian Grid transform
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


def LPGrid(nx, ny, rmin, rmax, nwedges, nring):
    # Log Polar Grid
    # Zero Center for LP Transform
    xZero = (nx + 0.5) / 2
    yZero = (ny + 0.5) / 2

    # Log R coordinate space
    rv = np.exp( np.linspace( np.log(rmin), np.log(rmax), nring))

    # Angular coordinate vector
    thv = np.linspace(0, 2 * np.pi * (1 - 1 / nwedges), nwedges)

    # Log-polar grid
    r, th = np.meshgrid(rv, thv)

    # Convert log polar grid to cartesian.
    x, y = pol2cart(r, th)

    # Shift the center of the log-polar grid to the zero-frequency
    x = x + xZero
    y = y + yZero

    return x, y


def tformImg(x, y, M, nX, nY, img):
    # Transform image based on warping
    # Build vector of zeros (Z)
    X = np.ones([nX * nY, 1])
    Y = np.ones([nX * nY, 1])
    Z = np.ones([nX * nY, 1])

    # Flatten arrays
    X[:, 0] = x.flatten('C')
    Y[:, 0] = y.flatten('C')

    # Build Nx3 matrix [X,Y,Z]
    inputPoints = np.column_stack((X, Y, Z)).T
    interpPoints = np.linalg.lstsq(M, inputPoints)[0]
    interpImg = np.empty_like(x)
    X = np.reshape(interpPoints[0, :], (nY, nX))
    Y = np.reshape(interpPoints[1, :], (nY, nX))
    sp.ndimage.interpolation.map_coordinates(img, [Y + nY / 2 - 0.5, X + nX / 2 - 0.5],
                                             output=interpImg,
                                             order=1,
                                             mode='constant',
                                             cval=0.0,
                                             prefilter=True)

    return interpImg


def tformLP(x, y, xLP, yLP, nX, nY, img):
    # Log Polar Transform
    interpImg = np.empty_like(xLP)
    sp.ndimage.interpolation.map_coordinates(img, [yLP, xLP],
                                             output=interpImg, order=1,
                                             mode='constant', cval=0.0,
                                             prefilter=True)

    return interpImg
