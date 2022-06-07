#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: Erik Osinga, heavily borrowed from Jort Boxelaar HALO-FDCA
'''

from astropy.coordinates import SkyCoord
import logging
import os
from datetime import datetime
import astropy.units as u
import numpy as np
import argparse

import FDCA


####

from astropy.io import fits
import matplotlib.pyplot as plt

"""


Plan: 

1. First just try to get the MCMC working on the test image. 
Everything in pixel coordinates. 

/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest.fits
 
 -. Then think about the initial guesses: ##### DONE. CURVE_FIT. 

 -. Test what happens to NaNs ##### DONE. THEY ARE IGNORED.

 -. Then regrid data to beam size? # 
            # Convert input x0 y0 also to these re-gridded coords?


            # MAYBE FIRST TRY WHAT HAPPENS TO CURVE_FIT IF I MODEL THE REGRID INSIDE 
            # THE CURVE_FIT MODEL
            # THEN X0 AND Y0 SHOULD BE 400,600 
            # AND THEN COMPARE WHAT HAPPENS WHEN I INPUT REGRIDDED COORDS.

 -. Then think about masking # TODO: BEST TO MAKE A COPY AND SET THESE VALUES ALSO TO NAN

                             # MAYBE BEST:
                             ###  Set any values in the mask to np.nan
                             ### Save a new mask that is 1 wherever the value is np.nan
                             ### then also rotate and regrid this new mask
                             ### and input into MCMC only the values that are not masked.



 -. Then convert the output to physical coordinates.

 -. Then implement arguments

 -. Then implement the logging.

 -. Then implement rotation wrt the beam. 

 -. Finally implement other models than circle

"""


def check_datashape(data):
    """ Check shape of HDU to see if it is 2 axis or 4 axis. Return 2D array."""
    if len(data.shape) == 4:
        if data.shape[0] == 1:
            # Then data is of the form (1,1,xpix,ypix)
            return data[0,0]
        elif data.shape[-1] == 1:
            # Then data is of the form (xpix,ypix,1,1)
            return data[:,:,0,0]
    if len(data.shape) == 2:
        # data is already 2 dimensional
        return data
    raise ValueError(f"Data shape {data.shape} not recognised.")

def image_info(header, imagerms):
    bmaj      = header['BMAJ']*u.deg
    bmin      = header['BMIN']*u.deg
    bpa       = header['BPA']*u.deg
    pix_size  = abs(header['CDELT2'])*u.deg # square pixels
    beammaj_sigma        = bmaj/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
    beammin_sigma        = bmin/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
    pix_area  = abs(header['CDELT1']*header['CDELT2'])*u.deg*u.deg
    beam_area = 2.*np.pi*1.0*beammaj_sigma*beammin_sigma
    beam2pix  = beam_area/pix_area

    image_info = {
        "bmaj":              bmaj,
        "bmin":              bmin,
        "bpa":               bpa,
        "pix_size":          pix_size,
        "beam_area":         beam_area,
        "beam2pix":          beam2pix,
        "xsize":             header['NAXIS1'],
        "ysize":             header['NAXIS2'],
        "max_radius":        header['NAXIS1']*2,
        "imagerms":          imagerms, # in Jy/beam at the moment

        # print("TODO")
        # "pix2kpc":           pix2kpc,
        # "mask":              obj.mask,
        # "sigma":             obj.mcmc_noise,
        # "margin":            margin,
        # "_func_":            obj._func_mcmc,
        # "image_mask":        obj.image_mask,
        # "binned_image_mask": obj.binned_image_mask,
        # "mask_treshold":     obj.mask_treshold,
        # "max_radius":        obj.max_radius,
        # "params":            obj.params,
        # "paramNames":        obj.paramNames,
        # "gamma_prior":       obj.gamma_prior,
        }
    return image_info

if __name__ == '__main__':
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest.fits'
    image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_NaN.fits'
    image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_noise.fits'
    image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_noise_noNaN.fits'

    # image = './halotest_noise_noNaN.fits'

    with fits.open(image) as hdul:
        data = hdul[0].data
        header = hdul[0].header


    imagerms = 0.01 # Jy/beam

    data = check_datashape(data)
    iminfo = image_info(header, imagerms)

    # I0, x0, y0, r_e  (#Jy/beam, pixel, pixel, pixel)
    p0 = [1, 660, 423, 20] ## USER INITIAL GUESS: TODO

    bounds = ([0.0,0.0,0.0,0.0]
             ,[np.inf, data.shape[0], data.shape[1], data.shape[0]*3]
             )

    # x and y image coords in a grid. 
    x     = np.arange(0, data.shape[1])
    y     = np.arange(0, data.shape[0])
    coord = np.meshgrid(x,y)

    fitter = FDCA.mcmc_eo.MCMCfitter(data, iminfo, mask=False, regrid=True)

    if False:
        # Get first guess from curve_fit
        pinit = fitter.pre_mcmc_fit(p0, bounds)
        print(f"Initial guess from curve_fit: {pinit}")
    else:
        pinit = np.array([1.0,600,400,40])
        print(f"Initial guess for MCMC: {pinit}")

    self=fitter


    sampler = fitter.runMCMC(pinit, walkers=8, steps=100)
    fitter.print_bestfitparams()

    FDCA.mcmc_eo.plotMCMC(fitter.samples, pinit)

    # import sys
    # sys.exit("\nFinished")

