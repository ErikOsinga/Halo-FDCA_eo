#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: Erik Osinga, heavily borrowed from Jort Boxelaar HALO-FDCA
'''

from __future__ import division
import sys
import os
import logging
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy import ndimage
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from skimage.measure import block_reduce
from skimage.transform import rescale
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import emcee
import corner


class MCMCfitter(object):
    """MCMC fitter class"""
    def __init__(self, data, iminfo, regrid=True, mask=False):
        self.data = data
        self.iminfo = iminfo
        self.regrid = regrid # whether to regrid to 1 pixel approx 1 beam
        self.mask = mask

        self.wherefinite = np.isfinite(data)

        ## TODO: in case of mask make more NaN values
        self.data_noNaN = data[self.wherefinite] # is 1D array

        self.dim = 4 # Circle model
        self.labels = ['I0', 'x0', 'y0', 'r_e']

        # x-y grid of pixels before re-gridding
        self.coords = np.meshgrid(np.arange(0, data.shape[1]),np.arange(0, data.shape[0]))

        if mask:
            print("TODO")

    def convolve_with_gaussian(self, data):
        """
        data   -- 2D array
        """
        # convert FWHM of beam to standard deviations of gaussians
        sigma1 = (self.iminfo['bmaj']/self.iminfo['pix_size'])/np.sqrt(8*np.log(2.))
        sigma2 = (self.iminfo['bmin']/self.iminfo['pix_size'])/np.sqrt(8*np.log(2.))
        kernel = Gaussian2DKernel(sigma1, sigma2, self.iminfo['bpa'])
        astropy_conv = convolve(data,kernel,boundary='extend',normalize_kernel=True)
        return astropy_conv

    def circle_model(self, theta):
        """Return circle model as 1D array"""
        x,y = self.coords
        G   = ((x-theta['x0'])**2+(y-theta['y0'])**2)/theta['r1']**2
        Ir  = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
        Ir = self.convolve_with_gaussian(Ir)

        if self.regrid: 
            # Have to regrid the model as well as the data
            Ir = regridding(Ir, self.iminfo)
            
            # ("TODO:: case of NaNs", dont flatten but use mask)
            Ir = Ir.flatten()
        
        else: # No regridding
            # Flatten and remove where data is NaN for a fair comparison
            Ir = Ir[self.wherefinite]
        return Ir

    def pre_mcmc_func(self, obj, *theta):
        """Simply returns the convolved circle model"""
        theta = {'I0':theta[0], 'x0':theta[1],'y0':theta[2],'r1':theta[3]}
        # Default params
        theta['k_exp'] = 0; theta['off'] = 0

        model = self.circle_model(theta)

        if self.mask:
            print("TODO")
            return model[obj.image_mask.ravel() == 0]

        return model

    def pre_mcmc_fit(self, p0, bounds):

        ### TODO: want to rebin already here or no? 


        if self.mask:
            print("TODO")
            ## something like data = data[self.image_mask.ravel() == 0]
        
        # Dont have to check finite because I already remove NaNs
        # Call pre_mcmc_func with f(self, *params) and make it match data_noNaN
        opt, pcov = curve_fit(self.pre_mcmc_func, self, self.data_noNaN
                            ,p0=p0, bounds=bounds, check_finite=False) 

        return opt # initial guess I0,x0,y0,r_e in Jy/beam and pixel coords

    def runMCMC(self, theta_guess, walkers=12, steps=400, burninfrac=0.25):
        """
        Start MCMC with initial guess theta_guess
        """

        ## TODO Maybe define this in the main class?
        self.walkers = walkers
        self.steps = steps

        if self.regrid:
            # rotate and regrid the data such that 1 pix approx 1 beam

            ## TODO: case of NANS 
            self.data_rebin = regridding(self.data, self.iminfo)
            
            ## TODO: case of NaNS use the rebinned mask
            self.data_mcmc = self.data_rebin.flatten()

        else:
            # Simply use all datapoints except those that are NaN
            self.data_mcmc = self.data_noNaN

        pos = [theta_guess*(1.+1.e-3*np.random.randn(self.dim)) for i in range(self.walkers)]

        num_CPU = cpu_count()
        with Pool(num_CPU) as pool:
            ### TODO: pool
            sampler = emcee.EnsembleSampler(self.walkers, self.dim, lnprob, pool=pool,
                            args=[self.data_mcmc, self.iminfo, self.circle_model])
            sampler.run_mcmc(pos, self.steps, progress=True)

        self.sampler_chain = sampler.chain
        self.burntime = int(burninfrac * steps)
        # nwalkers by nsteps-burnin by number of parameters: (12,300,4)
        self.samples = self.sampler_chain[:,self.burntime:,:]#.reshape((-1,self.dim))

        percentiles = np.ones((self.samples.shape[-1],3)) #(4,3) # 4 params, 3 percentiles
        for i in range(self.samples.shape[-1]):
            percentiles[i,:] = np.percentile(self.samples[:, :, i], [16, 50, 84])
        self.percentiles = percentiles

        return sampler

    def print_bestfitparams(self):
        """
        After runMCMC() call this to print best fit params
        """
        for i in range(self.dim):
            low, mid, up = self.percentiles[i]
            print(f"{self.labels[i]} = {mid:.1f}^+{(up-mid):.1f}_-{(mid-low):.1f}")


def lnprob(theta, data, info, modelf):
    """
    Log posterior, inputting a vector of parameters and data
    """    

    # TODO: make this better?
    theta = {'I0':theta[0], 'x0':theta[1],'y0':theta[2],'r1':theta[3]}
    # Default params
    theta['k_exp'] = 0; theta['off'] = 0
    ## TODO: rotation
    theta['ang'] = 0
    theta['r2'] = 0

    lp = lnprior(theta, info)
    if not np.isfinite(lp):
        return -np.inf
    return lnL(theta, data, info, modelf) + lp

def lnL(theta, data, info, modelf):
    """
    Log likelihood, inputting a vector of parameters and data
    """   

    ## TODO: rotation
    # kwargs = {"rotate" : True}
    # raw_model = info['_func_'](info,coord,theta,**kwargs)*u.Jy
    # model = set_model_to_use(info, raw_model)

    model = modelf(theta)
    return -np.sum( ((data-model)**2.)/(2*info['imagerms']**2.)\
                        + np.log(np.sqrt(2*np.pi)*info['imagerms']) )

def lnprior(theta, info):
    prior = -np.inf
    if (theta['I0'] > 0) and (-0.4 < theta['k_exp'] < 19):
        if (0 <= theta['x0'] < info['xsize']) and (0 <= theta['y0'] < info['ysize']):
            if 0 < theta['r1'] < info['max_radius']:
                if -np.pi/4. < theta['ang'] < 5*np.pi/4.:
                    prior = 0.0
                if not (0 <= theta['r2'] <= theta['r1']):
                    prior = -np.inf

    if prior != -np.inf:
        radii = np.array([theta['r1']])
    return prior

def plotMCMC(samples, pinit):
    """
    Plot MCMC chain and initial guesses
    """
    labels = ['I0', 'x0', 'y0', 'r_e']
    dim = samples.shape[-1]

    ########## CORNER PLOT ###########
    fig = corner.corner(samples.reshape((-1,dim)),labels=labels, quantiles=[0.160, 0.5, 0.840],
                        truths=pinit, # plot blue line at inital value
                        show_titles=True, title_fmt='.5f')
    
    # if savefig is not None: plt.savefig(savefig.replace('.pdf','_corner.pdf'))
    plt.show()
    plt.close()

    ########## PLOT WALKERS ###########
    fig, axes = plt.subplots(ncols=1, nrows=dim, sharex=True)
    axes[0].set_title('Number of walkers: '+str(len(samples)))
    for axi in axes.flat:
        axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        fig.set_size_inches(2*10,15)

    for i in range(dim):
        axes[i].plot(samples[:, :, i].transpose(),
                                        color='black', alpha=0.3)
        axes[i].set_ylabel('param '+str(i+1), fontsize=15)
        plt.tick_params(labelsize=15)

    # if savefig is not None: plt.savefig(savefig.replace('.pdf','_walkers.pdf'))
    plt.show()
    plt.close()

def rotate_image(data, iminfo):
    """Rotate image
    iminfo -- dictionary -- Should have beam position angle 'bpa' 
    """

    ### Jort puts any np.nan to zero before rotation. 
    ### because otherwise whole image is nan. 

    ### TODO: in case of NaN make sure we put also values to 0

    img_rot = ndimage.rotate(data, -iminfo['bpa'].value, reshape=False)
    return img_rot

def regrid_to_beamsize(data, iminfo, accuracy=100.):
    """Regrid data to 1 pixel = 1 beam
    iminfo -- dictionary -- Made by def image_info() in HaloFitting_eo.py
    """    
    y_scale = np.sqrt(iminfo['beam_area']*iminfo['bmin']/iminfo['bmaj']).value
    x_scale = (iminfo['beam_area']/y_scale).value
    new_pix_size = np.array((y_scale,x_scale))
    
    accuracy = int(1./accuracy*100)

    scale = np.round(accuracy*new_pix_size/iminfo['pix_size']).astype(np.int64).value

    pseudo_size = (accuracy*np.array(data.shape) ).astype(np.int64)    
    orig_scale = (np.array(pseudo_size)/np.array(data.shape)).astype(np.int64)
    elements   = np.prod(np.array(orig_scale,dtype='float64'))

    if accuracy is 1:
        pseudo_array = np.copy(data)
    else:
        pseudo_array = np.zeros((pseudo_size))
        for j in range(data.shape[0]):
            for i in range(data.shape[1]):
                pseudo_array[orig_scale[1]*i:orig_scale[1]*(i+1),
                             orig_scale[0]*j:orig_scale[0]*(j+1)] = data[i,j]/elements

    # subsampled array where approx 1 pixel is 1 beam.
    f= block_reduce(pseudo_array, block_size=tuple(scale), func=np.sum, cval=0)
    return f

def regridding(data, iminfo):
    """Rotate image
    iminfo -- dictionary -- Should have beam position angle 'bpa' 
                            and 

    RETURNS
    IMAGE THAT IS ROTATED AND REGRIDDED TO 1 PIXEL IS 1 BEAM
    """    
    data_rot = rotate_image(data, iminfo)
    regrid   = regrid_to_beamsize(data_rot, iminfo)
    return regrid