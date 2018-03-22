#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:10:08 2017

@author: brettmeyers
"""
import cv2
import numpy as np
import scipy as sp
from subPixelFit import subPix2D
from apodWindows import gaussWindow, hanningWindow
from coordinateTform import tformLP, LPGrid
from pdb import set_trace as keyboard
def SCC(fr01,fr02):
    Win2D       = hanningWindow(fr02.shape)
    ccSpectral  = np.zeros([fr01.shape[0],fr01.shape[1]])
    if len(fr01.shape) == 2:
        # Take FFTs of the images
        FFT01   = sp.fftpack.fft2(np.multiply(Win2D,\
                                              fr01-np.mean(fr01.flatten())))
        FFT02   = sp.fftpack.fft2(np.multiply(Win2D,\
                                              fr02-np.mean(fr02.flatten())))
        # Cross-correlate
        ccSpectral = ccSpectral + np.multiply(FFT01,np.conj(FFT02))
    else:
        for i in np.arange(0,fr01.shape[2],1):
            # Take FFTs of the images
            FFT01 = sp.fftpack.fft2(np.multiply(Win2D,\
                                fr01[:,:,i]-np.mean(fr01[:,:,i].flatten())))
            FFT02 = sp.fftpack.fft2(np.multiply(Win2D,\
                                fr02[:,:,i]-np.mean(fr02[:,:,i].flatten())))
            # Cross-correlate
            ccSpectral = ccSpectral + np.multiply(FFT01,np.conj(FFT02))
    # Tranform back to spatial domain, perform shift
    ccSpatial   = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(ccSpectral)))
    # Find location of peak in correlation
    peakLocs    = np.where(ccSpatial == np.max(ccSpatial.flatten('C')))
    # Perform subpixel fitting
    disp        = subPix2D(ccSpatial,peakLocs,np.array([0,0]))
    # Store temporary displacements
    tdx         = disp[1]
    tdy         = disp[0]
    return tdx, tdy
def FMC(fr01,fr02,MinRad,MaxRad,NoOfWedges,NoOfRings,sdx):
    nX = fr01.shape[1]
    nY = fr01.shape[0]
    # Construct Log Polar Grid
    imgLPX, imgLPY  = LPGrid(nX,nY,MinRad,MaxRad,NoOfWedges,NoOfRings)
    # Generate meshgrid 
    imgX, imgY  = np.meshgrid(np.linspace(-nX/2+0.5, nX/2-0.5, nX), \
                             np.linspace(-nY/2+0.5, nY/2-0.5, nY))          
    WinLP       = hanningWindow(imgLPX.shape)
    Win2D       = hanningWindow(fr02.shape)
    ccSpectral  = np.zeros([imgLPX.shape[0],imgLPX.shape[1]])
    if len(fr01.shape) == 2:
        # Take magnitude of FFT of images
        mFFT01 = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(\
                            np.multiply(Win2D,fr01-np.mean(fr01.flatten())))))
        mFFT02 = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(\
                            np.multiply(Win2D,fr02-np.mean(fr02.flatten())))))
        # Log Polar Unwrap
        lp01 = np.abs(tformLP(imgX,imgY,imgLPX,imgLPY,\
                              imgLPX.shape[1],imgLPX.shape[0],mFFT01))
        lp02 = np.abs(tformLP(imgX,imgY,imgLPX,imgLPY,\
                              imgLPX.shape[1],imgLPX.shape[0],mFFT02))
        # Take FFTs of the images
        FFT01 = sp.fftpack.fft2(np.multiply(WinLP,lp01))
        FFT02 = sp.fftpack.fft2(np.multiply(WinLP,lp02))
        # Cross-correlate
        ccSpectral = np.multiply(FFT01,np.conj(FFT02))
    else:
        for i in np.arange(0,fr01.shape[2],1):
            # Take magnitude of FFT of images
            mFFT01 = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(\
            np.multiply(Win2D,fr01[:,:,i]-np.mean(fr01[:,:,i].flatten())))))
            mFFT02 = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(\
            np.multiply(Win2D,fr02[:,:,i]-np.mean(fr02[:,:,i].flatten())))))
            # Log Polar Unwrap
            lp01 = np.abs(tformLP(imgX,imgY,imgLPX,imgLPY,\
                                  imgLPX.shape[1],imgLPX.shape[0],mFFT01))
            lp02 = np.abs(tformLP(imgX,imgY,imgLPX,imgLPY,\
                                  imgLPX.shape[1],imgLPX.shape[0],mFFT02))
            # Take FFTs of the images
            FFT01 = sp.fftpack.fft2(np.multiply(WinLP,lp01))
            FFT02 = sp.fftpack.fft2(np.multiply(WinLP,lp02))
            # Cross-correlate
            ccSpectral = ccSpectral + np.multiply(FFT01,np.conj(FFT02))
    # Tranform back to spatial domain, perform shift
    ccSpatial = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(ccSpectral)))
    # Find location of peak in correlation
    peakLocs = np.where(ccSpatial == np.max(ccSpatial.flatten('C')))
    offsets = np.array([0,0])
    if np.mod(ccSpatial.shape[0],2) !=0:
        offsets[0] = -0.5
    if np.mod(ccSpatial.shape[1],2) !=0:
        offsets[1] = -0.5
    # Perform subpixel fitting
    disp = subPix2D(ccSpatial,peakLocs,offsets)
    # Update Scale Value
    sdx     -= disp[1]
    stemp    = disp[1]
    scale    = np.exp(1*np.log(MaxRad/MinRad)*sdx/(nX-1))
    return scale, sdx, stemp
def LPC(fr01,fr02,MinRad,MaxRad,NoOfWedges,NoOfRings,sdx):
    nX = fr01.shape[1]
    nY = fr01.shape[0]
    # Construct Log Polar Grid
    imgLPX, imgLPY  = LPGrid(nX,nY,MinRad,MaxRad,NoOfWedges,NoOfRings)
    # Generate meshgrid 
    imgX, imgY  = np.meshgrid(np.linspace(-nX/2+0.5, nX/2-0.5, nX), \
                             np.linspace(-nY/2+0.5, nY/2-0.5, nY))          
    WinLP       = hanningWindow(imgLPX.shape)
    Win2D       = hanningWindow(fr02.shape)
#    Win2D       = gaussWindow(fr02.shape,0.2)
    ccSpectral  = np.zeros([imgLPX.shape[0],imgLPX.shape[1]])
#    ccSpectral  = np.zeros(fr02.shape[0:2])
    if len(fr01.shape) == 2:
        # Log Polar Unwrap
        lp01 = tformLP(imgX,imgY,imgLPX,imgLPY,\
                imgLPX.shape[1],imgLPX.shape[0],np.multiply(Win2D,\
                    fr01-np.mean(fr01.flatten())))
        lp02 = tformLP(imgX,imgY,imgLPX,imgLPY,\
                imgLPX.shape[1],imgLPX.shape[0],np.multiply(Win2D,\
                    fr02-np.mean(fr02.flatten())))
#        lp01    = cv2.logPolar(fr01-np.mean(fr01.flatten()),\
#                               (nY/2,nX/2),nX/np.log(MaxRad),cv2.INTER_LINEAR)
#        lp02    = cv2.logPolar(fr02-np.mean(fr02.flatten()),\
#                               (nY/2,nX/2),nX/np.log(MaxRad),cv2.INTER_LINEAR)
        WinLP   = hanningWindow(lp01.shape)
        # Take FFTs of the images
        FFT01 = sp.fftpack.fft2(np.multiply(WinLP,lp01))
        FFT02 = sp.fftpack.fft2(np.multiply(WinLP,lp02))
        # Cross-correlate
        ccSpectral = np.multiply(FFT01,np.conj(FFT02))
    else:
        for i in np.arange(0,fr01.shape[2],1):
            # Log Polar Unwrap
            lp01 = np.abs(tformLP(imgX,imgY,imgLPX,imgLPY,imgLPX.shape[1],\
            imgLPX.shape[0],np.multiply(Win2D,fr01[:,:,i]-\
                        np.mean(fr01[:,:,i].flatten()))))
            lp02 = np.abs(tformLP(imgX,imgY,imgLPX,imgLPY,imgLPX.shape[1],\
            imgLPX.shape[0],np.multiply(Win2D,fr02[:,:,i]-\
                        np.mean(fr02[:,:,i].flatten()))))
#            lp01    = cv2.logPolar(np.multiply(Win2D,fr01[:,:,i]-\
#                        np.mean(fr01[:,:,i].flatten())),\
#                        (nY/2,nX/2),nX/np.log(MaxRad),cv2.INTER_LINEAR)
#            lp02    = cv2.logPolar(np.multiply(Win2D,fr02[:,:,i]-\
#                        np.mean(fr02[:,:,i].flatten())),\
#                        (nY/2,nX/2),nX/np.log(MaxRad),cv2.INTER_LINEAR)
            WinLP   = hanningWindow(lp01.shape)
            # Take FFTs of the images
            FFT01 = sp.fftpack.fft2(np.multiply(WinLP,lp01))
            FFT02 = sp.fftpack.fft2(np.multiply(WinLP,lp02))
            # Cross-correlate
            ccSpectral = ccSpectral + np.multiply(FFT01,np.conj(FFT02))
    # Tranform back to spatial domain, perform shift
    ccSpatial = np.abs(sp.fftpack.fftshift(sp.fftpack.fft2(ccSpectral)))
    # Find location of peak in correlation
    peakLocs = np.where(ccSpatial == np.max(ccSpatial.flatten('C')))
    offsets = np.array([0,0]).astype(float)
    if np.mod(ccSpatial.shape[0],2) !=0:
        offsets[0] = -0.5
    if np.mod(ccSpatial.shape[1],2) !=0:
        offsets[1] = -0.5
    # Perform subpixel fitting
    disp = subPix2D(ccSpatial,peakLocs,offsets)
    # Update Scale Value
    sdx     -= disp[1]
    stemp    = disp[1]
    scale    = np.exp(1*np.log(MaxRad/MinRad)*sdx/(nX-1))
#    keyboard()
    return scale, sdx, stemp
# Log Polar Unwrap
#lp01    = cv2.logPolar(np.abs(mFFT01),(nY/2,nX/2),nX/np.log(MaxRad),cv2.INTER_LINEAR)
#lp02    = cv2.logPolar(np.abs(mFFT02),(nY/2,nX/2),nX/np.log(MaxRad),cv2.INTER_LINEAR)
#WinLP   = hanningWindow(lp01.shape)