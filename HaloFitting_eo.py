#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: Erik Osinga, heavily borrowed from Jort Boxelaar HALO-FDCA


TODO: 

 -. Implement the logging.

 -. Implement other models than circle ### BIG TODO.

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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean type expected.')

def newargparse(bashfile=None):
    parser = argparse.ArgumentParser(description='Halo-FDCA (basic version!): An automated MCMC fitter for radio halos in galaxy clusters. (methods from Boxelaar et al. 2021)')

    # Required arguments
    parser.add_argument('image',          help='(str) FITS image location.', type=str)
    parser.add_argument('-rms',           help="(float) RMS in FITS image. In Jy/beam.", type=float, required=True)
    parser.add_argument('-p0',            help="(float) User initial guess for parameter I_0. In Jy/beam.", type=float, required=True)
    # Note that X,Y in ds9 coordinates indeed corresponds in the code to p1,p2 (x0,y0)
    parser.add_argument('-p1',            help="(float) User initial guess for parameter x_0. In PIXEL coordinates.", default = None, type=float, required=True)
    parser.add_argument('-p2',            help="(float) User initial guess for parameter y_0. In PIXEL coordinates.", default = None, type=float, required=True)
    parser.add_argument('-p3',            help="(float) User initial guess for parameter r_e. In PIXEL coordinates.", default = None, type=float, required=True)
    parser.add_argument('-redshift',      help='(float) cluster redshift', type=float, required=True)
    # Optional arguments
    parser.add_argument('-nwalkers',      help='(int) Amount of walkers. Default 200',default=200, type=int)
    parser.add_argument('-steps',         help='(int) Amount of steps.  Default=1000',default=1000, type=int)
    parser.add_argument('-output_dir',    help='(str) Path to output. Default: ./output/', default='./output/', type=str)
    parser.add_argument('-mask',          help='(str) Mask file location (.reg). Mask sources INSIDE these regions. Default: None', default=None, type=str)
    parser.add_argument('-maskoutside',   help='(str) Mask file location (.reg). Mask everything OUTSIDE this region. Default: None', default=None, type=str)
    parser.add_argument('-rms_region',     help='(str) Region file location (.reg). Used to calculate the RMS. Necessary if user wants chi2 value when regrid=True. Default: None', default=None, type=str)
    parser.add_argument('-regrid',        help='(bool) Whether to regrid and rotate image to 1 beam = 1 pix. Default: True',default=True, type=str2bool)
    parser.add_argument('-run_MCMC',      help='(bool) Whether to run a MCMC routine or skip it to go straight to processing. can be done if a runned sample already exists in the output path. Default: True',default=True, type=str2bool)
    parser.add_argument('-curvefit',      help='(bool) Whether to do a simple fit to get an initial guess for MCMC or just use the user input guess. Default: True',default=True, type=str2bool)
    parser.add_argument('-d_int',         help='(float) Fraction of r_e far to integrate the model up to. Default=2.6',default=2.6, type=float)
    parser.add_argument('-d_int_kpc',     help='(float) Radius to integrate the model up to in kpc. Default=None',default=None, type=float)
    parser.add_argument('-presentation',  help='(bool) Plot transparent figures. Default: False',default=False, type=str2bool)
    parser.add_argument('-saveradial',    help='(bool) Whether to save the radial profile. Default: False',default=False, type=str2bool)

    if bashfile is None:
        args = parser.parse_args()
    else:
        print(f"WARNING: LOADING PARAMETERS FROM FILE {bashfile}")
        argsfromfile = FDCA.utils.argsfromfile(bashfile)
        args = parser.parse_args(argsfromfile.split(' '))

    args.puser = [args.p0, args.p1, args.p2, args.p3]
    return args


if __name__ == '__main__':

    args = newargparse()

    #### If user wants to call this from ipython uncomment these two lines
    # bashfile = 'plot23.sh'
    # bashfile = 'plot46.sh'
    # bashfile = 'plot144.sh'
    # bashfile = 'plot144_kamlesh.sh'
    # args = newargparse(bashfile)

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
        sampler = fitter.runMCMC(pinit, walkers=args.nwalkers, steps=args.steps)
    else:
        print ("\n Loading previous results")
        sampler = fitter.loadMCMC()
    ########## FITTING ##########

    ########## PROCESSING ##########
    print("Best fit params in image units")
    fitter.print_bestfitparams(fitter.percentiles)
    # Automatically save figures in the output directory using the image name.
    savefig = fitter.plotdir + args.image.split('/')[-1].replace('.fits','.pdf')
    # Corner plot and sampler chain plot
    FDCA.mcmc_eo.plotMCMC(fitter.samples, pinit, savefig=savefig) 
    # Data-model-residual plot
    fitter.plot_data_model_residual(savefig=savefig, presentation=args.presentation)
    ########## PROCESSING ##########

    ########## CONVERT TO PHYSICAL UNITS ##########
    # Convert params to RA,DEC and r_e in kpc.
    fitter.convert_units()
    # print params also in useful units
    print("Best fit params in physical units")
    fitter.print_bestfitparams(fitter.percentiles_units)
    # calculate total flux
    totalflux = fitter.totalflux(args.d_int, args.d_int_kpc)
    ########## CONVERT TO PHYSICAL UNITS ##########

    ########## ADDITIONAL PLOTTING DIAGNOSTICS ##########
    if args.saveradial:
        saveradial = fitter.output_dir + args.image.split('/')[-1].replace('.fits','_dataprofile.npy')
    else:
        saveradial = None

    fitter.plot_1D(d=3.0, d_int_kpc=None, savefig=savefig, saveradial=saveradial)


    # FDCA.plotting.compare_annuli():

    ########## ADDITIONAL PLOTTING DIAGNOSTICS ##########