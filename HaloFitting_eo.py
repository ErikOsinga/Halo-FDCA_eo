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

from astropy.io import fits
import matplotlib.pyplot as plt
FDCA.utils.paper_fig_params()

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
 -. Then save the chain # DONE

 -. Then convert the output to physical coordinates. ## DONE

 -. Then calculate chi2

 -. Then implement arguments

 -. Then implement the logging.

 -. Then implement rotation wrt the beam. 

 -. Finally implement other models than circle

"""




if __name__ == '__main__':
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest.fits'
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_NaN.fits'
    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_noise_noNaN.fits'


    # image = '/home/osingae/Documents/phd/Abell2256/test_halofitting/halotest_noise.fits'
    # image = '/home/osingae/Documents/phd/Abell2256/images/Abell2256_23MHz.int.restored.zoomhalo_mask.fits'
    
    image = '/home/osingae/Documents/phd/Abell2256/images/Abell2256_23MHz.int.restored.zoom.fits'
    mask = '/home/osingae/Documents/phd/Abell2256/halo_unrelated_sources_kamlesh.reg'
    maskoutside = '/home/osingae/Documents/phd/Abell2256/Halo_main_LBA_region_Kamlesh.reg'

    regrid=True
    output_dir = './output/'
    run_MCMC = False
    curvefit = run_MCMC
    redshift = 0.058 # for unit conversions

    # I0, x0, y0, r_e  (#Jy/beam, pixel, pixel, pixel)
    # p0 = [1, 660, 423, 20] ## USER INITIAL GUESS: TODO MAKE ARGUMENT 
    
    # TODO: make arguments
    p0 = [0.1, 230, 260, 100] ## USER INITIAL GUESS: TODO MAKE ARGUMENT 

    rms = 8.5e-3 # 8.5 mJy/beam

    # Initialise the fitter
    fitter = FDCA.mcmc_eo.MCMCfitter(image, rms
        , mask=mask, regrid=regrid, output_dir=output_dir
        , maskoutside=maskoutside, redshift=redshift)

    if curvefit:
        # Get first guess from curve_fit
        pinit = fitter.pre_mcmc_fit(p0)
        print(f"Initial guess from curve_fit: {pinit}")
    else:
        pinit = np.array(p0)
        if run_MCMC: print(f"User initial guess for MCMC: {pinit}")

    self=fitter ## For testing purposes

    if run_MCMC:
        sampler = fitter.runMCMC(pinit, walkers=8, steps=100)
    else:
        sampler = fitter.loadMCMC()

    fitter.print_bestfitparams(fitter.percentiles)
    savefig = fitter.plotdir + image.split('/')[-1].replace('.fits','.pdf')

    # Corner plot and sampler chain plot
    FDCA.mcmc_eo.plotMCMC(fitter.samples, pinit, savefig=savefig)

    fitter.plot_data_model_residual(savefig=savefig)

    # Convert params to RA,DEC and r_e in kpc. Also calculate total flux
    fitter.convert_units()
    # fitter.print_bestfitparams(fitter.percentiles_units)

    # Calculate chi2