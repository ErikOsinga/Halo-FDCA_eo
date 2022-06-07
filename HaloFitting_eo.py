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
from astropy.wcs import WCS
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

 -. Then regrid data to beam size? ##### DONE. 
            # DATA regridding now happens inside the model, so the x0 and y0 and r_e
            # are in pixel coords. 

            ### TODO: Convert Jy/beam to Jy/arcsec2 ???


 -. Test what happens to NaNs ##### DONE. THEY ARE IGNORED

 -. Then think about masking # DONE 

                             ###  Set any values in the mask to np.nan
                             ### Save a new mask that is 1 wherever the value is np.nan
                             ### then also rotate and regrid this new mask
                             ### and input into MCMC only the values that are not masked.
 -. Then save the chain

 -. Then convert the output to physical coordinates.

 -. Then implement arguments

 -. Then implement the logging.

 -. Then implement rotation wrt the beam. 

 -. Finally implement other models than circle

"""




if __name__ == '__main__':
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest.fits'
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_NaN.fits'
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_noise_noNaN.fits'

    # image = '/home/osingae/Documents/phd/Abell2256/images/Abell2256_23MHz.int.restored.zoomhalo_mask.fits'

    image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_noise.fits'
    mask = '/home/osingae/Documents/phd/Abell2256/halo_unrelated_sources_kamlesh.reg'
    regrid=True
    output_dir = './output/'

    # I0, x0, y0, r_e  (#Jy/beam, pixel, pixel, pixel)
    p0 = [1, 660, 423, 20] ## USER INITIAL GUESS: TODO MAKE ARGUMENT 

    rms = 0.1 # 0.1 Jy/beam

    # Initialise the fitter
    fitter = FDCA.mcmc_eo.MCMCfitter(image, rms
        , mask=mask, regrid=regrid, output_dir=output_dir)

    if True:
        # Get first guess from curve_fit
        pinit = fitter.pre_mcmc_fit(p0)
        print(f"Initial guess from curve_fit: {pinit}")
    else:
        pinit = np.array([1.0,600,400,40])
        print(f"Initial guess for MCMC: {pinit}")

    self=fitter

    sampler = fitter.runMCMC(pinit, walkers=8, steps=10)
    fitter.print_bestfitparams()

    savefig = fitter.plotdir + image.split('/')[-1].replace('.fits','.pdf')
    FDCA.mcmc_eo.plotMCMC(fitter.samples, pinit, savefig=savefig)

    # import sys
    # sys.exit("\nFinished")

