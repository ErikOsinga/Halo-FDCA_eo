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

 -. Test what happens to NaNs ##### DONE. THEY ARE IGNORED

 -. Then think about masking # DONE 
                             ###  Set any values in the mask to np.nan
                             ### Save a new mask that is 1 wherever the value is np.nan
                             ### then also rotate and regrid this new mask
                             ### and input into MCMC only the values that are not masked.
 -. Then save the chain # DONE

 -. Then convert the output to physical coordinates. ## DONE

 -. Then calculate chi2 ## DONE, IMPROVED FROM JORTS CASE I THINK

 -. Then implement arguments  ### DONE

 -. Then implement the logging.

 -. Then implement rotation wrt the beam. ## DONE

 -. Finally implement other models than circle ## BIG TODO.

"""

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean type expected.')

def newargparse():
    parser = argparse.ArgumentParser(description='Halo-FDCA (basic version!): An automated MCMC fitter for radio halos in galaxy clusters. (methods from Boxelaar et al. 2021)')

    # Required arguments
    parser.add_argument('image',          help='(str) FITS image location.', type=str)
    parser.add_argument('-rms',           help="RMS in FITS image. In Jy/beam.", type=float, required=True)
    parser.add_argument('-p0',            help="User initial guess for parameter I_0. In Jy/beam.", type=float, required=True)
    # Note that X,Y in ds9 coordinates indeed corresponds in the code to p1,p2 (x0,y0)
    parser.add_argument('-p1',            help="User initial guess for parameter x_0. In PIXEL coordinates.", default = None, type=float, required=True)
    parser.add_argument('-p2',            help="User initial guess for parameter y_0. In PIXEL coordinates.", default = None, type=float, required=True)
    parser.add_argument('-p3',            help="User initial guess for parameter r_e. In PIXEL coordinates.", default = None, type=float, required=True)
    parser.add_argument('-redshift',      help='(float) cluster redshift', type=float, required=True)
    # Optional arguments
    parser.add_argument('-output_dir',    help='(str) Path to output. Default: ./output/', default='./output/', type=str)
    parser.add_argument('-mask',          help='(str) Mask file location (.reg). Mask sources INSIDE these regions. Default: None', default=None, type=str)
    parser.add_argument('-maskoutside',   help='(str) Mask file location (.reg). Mask everything OUTSIDE this region. Default: None', default=None, type=str)
    parser.add_argument('-rms_region',     help='(str) Region file location (.reg). Used to calculate the RMS. Necessary if user wants chi2 value when regrid=True. Default: None', default=None, type=str)
    parser.add_argument('-regrid',        help='(bool) Whether to regrid and rotate image to 1 beam = 1 pix. Default: True',default=True, type=str2bool)
    parser.add_argument('-run_MCMC',      help='(bool) Whether to run a MCMC routine or skip it to go straight to processing. can be done if a runned sample already exists in the output path. Default: True',default=True, type=str2bool)
    parser.add_argument('-curvefit',      help='(bool) Whether to do a simple fit to get an initial guess for MCMC or just use the user input guess. Default: True',default=True, type=str2bool)

    args = parser.parse_args()

    args.puser = [args.p0, args.p1, args.p2, args.p3]

    return args


if __name__ == '__main__':

    args = newargparse()

    # Initialise the fitter
    fitter = FDCA.mcmc_eo.MCMCfitter(args.image, args.rms
        , mask=args.mask, regrid=args.regrid, output_dir=args.output_dir
        , maskoutside=args.maskoutside, redshift=args.redshift
        , rms_region=args.rms_region)

    ########## FITTING ##########
    if args.curvefit: # Get first guess from curve_fit
        pinit = fitter.pre_mcmc_fit(args.puser)
        print(f"Initial guess from curve_fit: {pinit}")
    else: # Use user initial guess
        pinit = np.array(args.puser)
        if args.run_MCMC: print(f"User initial guess for MCMC: {pinit}")
    if args.run_MCMC:
        print ("\n Running MCMC")
        sampler = fitter.runMCMC(pinit, walkers=8, steps=100)
    else:
        print ("\n Loading previous results")
        sampler = fitter.loadMCMC()
    ########## FITTING ##########

    self=fitter ## For testing purposes

    ########## PROCESSING ##########
    print("Best fit params in image units")
    fitter.print_bestfitparams(fitter.percentiles)
    # Automatically save figures in the output directory using the image name.
    savefig = fitter.plotdir + args.image.split('/')[-1].replace('.fits','.pdf')
    # Corner plot and sampler chain plot
    FDCA.mcmc_eo.plotMCMC(fitter.samples, pinit, savefig=savefig) 
    # Data-model-residual plot
    fitter.plot_data_model_residual(savefig=savefig)
    ########## PROCESSING ##########


    ########## CONVERT TO PHYSICAL UNITS ##########
    # Convert params to RA,DEC and r_e in kpc.
    fitter.convert_units()
    # print params also in useful units
    print("Best fit params in physical units")
    fitter.print_bestfitparams(fitter.percentiles_units)
    # calculate total flux
    totalflux = fitter.totalflux(d=2.6)
    ########## CONVERT TO PHYSICAL UNITS ##########
