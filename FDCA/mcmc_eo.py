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
import matplotlib.colors as mplc

from skimage.measure import block_reduce
from skimage.transform import rescale
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import emcee
import corner
from regions import Regions

from . import utils
utils.paper_fig_params()


class MCMCfitter(object):
    """MCMC fitter class

    image  -- str   -- location of fits image
    rms    -- float -- rms in Jy/beam
    regrid -- bool  -- whether to regrid to 1pix=1beam
    mask   -- str   -- mask file to remove sources
    """
    def __init__(self, image, rms, regrid=True, mask=None, output_dir='./output/', maskoutside=None, redshift=0.058):
        self.image = image
        self.regrid = regrid # whether to regrid to 1 pixel approx 1 beam
        self.mask = mask
        self.maskoutside = maskoutside
        self.output_dir = output_dir
        self.redshift = redshift
        self.dim = 4 # Circle model
        self.labels = ['$I_0$', '$x_0$', '$y_0$', '$r_e$']

        # Bookkeeping of settings and directories
        self.check_settings()

        # Load the image file, WCS information and other handy conversions
        # Note pickling WCS gives WARNING: https://github.com/astropy/astropy/issues/12834
        self.data_original, self.wcs1, self.iminfo  = self.loadfitsim(image, rms)
    
        # Bookkeeping, store the data to use as self.data_mcmc 
        # and the mask as self.wherefinite
        self.set_data_to_use()

    def check_settings(self):
        """Check input settings"""
        check_createdir(self.output_dir)

        self.plotdir = self.output_dir + 'Plots/'
        check_createdir(self.plotdir)

        self.imname = self.image.split('/')[-1]
        print(f"Using image {self.imname}")

        ## TODO: log

    def loadfitsim(self, image, rms):
        """Load fits image and return useful info"""
        with fits.open(image) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs1 = wcs.WCS(header)

        # Dealing with the fact that .fits files can have 2 or 4 axes
        data = check_datashape(data)
        # Loading useful info from header
        iminfo = image_info(header, rms)

        return data, wcs1, iminfo

    def set_data_to_use(self):
        """
        Bookkeeping. 

        1. Masks the data inside the regions defined if mask file is given

        2. Keeps track of the case with and without regridding.

        
        if self.regrid then:
            All model calculations are done wrt the original grid, which is then
            reprojected and regridded in the same way as the data and compared to that
            regridded data. 
        """

        # Mask the data by setting NaN inside the regions defined by the user
        # and return a MASK array that is False where NaNs occur
        data, self.wherefinite_original, data_mask0 = self.mask_data(self.data_original)
        # data is still the original 2D shape, data_mask0 is data with 0 instead of NaNs

        ## TODO: maskoutside can also be defined quite easily
        
        # x-y grid of pixels before re-gridding. Models will be computed on this grid
        self.coords = np.meshgrid(np.arange(0, data.shape[1]),np.arange(0, data.shape[0]))

        if self.regrid:
            # We need to regrid also the mask
            # In this case don't take the sum but just take the max value for pooling
            self.wherefinite = regridding(self.wherefinite_original, self.iminfo, func=np.max)

            # regridded version of the data (cant have NaNs)
            self.data_rebin = regridding(data_mask0, self.iminfo)

            # # Regridded version of the data with NaNs, to test the mask
            # self.data_rebin_masked = np.copy(self.data_rebin)
            # self.data_rebin_masked[~self.wherefinite] = np.nan

            # 1D array of the regridded data outside the mask (removed NaNs)
            self.data_mcmc = self.data_rebin[self.wherefinite]

        else: ## No reprojecing and regridding
            # 1D array of the data outside the mask (removed NaNs)
            self.data_mcmc = data[self.wherefinite_original] 
            self.wherefinite = self.wherefinite_original

        return

    def mask_data(self, data):
        """
        Mask the data if self.mask is not None by setting it to NaN

        Then return a mask array that is True wherever the data is NaN
        """

        data_masked = np.copy(data)
        
        w = self.wcs1
        while w.naxis > 2: # in case the wcs is 4 axes
            w = w.dropaxis(-1)
        
        if self.mask is not None:
            print (f"Masking regions {self.mask}")
            r = Regions.read(self.mask)
            # Mask every region by setting it to NaN
            for i in range(len(r)):
                rmask = r[i].to_pixel(w).to_mask()
                # True where inside region, 0 where not, same shape as data
                rmask = np.array(rmask.to_image(data.shape),dtype='bool')
                # Mask INSIDE the region
                data_masked[rmask] = np.nan  

        if self.maskoutside is not None:
            print (f"Masking outside the region {self.maskoutside}")
            # Also make data outside the regions self.maskoutside.
            r = Regions.read(self.maskoutside)
            if len(r) > 1:
                raise ValueError(f"Mask outside should contain only 1 region. Found {len(r)} regions")            
            rmask = r[0].to_pixel(w).to_mask()
            # True where inside region, 0 where not, same shape as data
            rmask = np.array(rmask.to_image(data.shape),dtype='bool')
            # Mask OUTSIDE the region
            data_masked[~rmask] = np.nan

        wherefinite = np.isfinite(data_masked)

        # We also need a copy with NaNs to zero, because we have to rotate and regrid
        data_mask0 = np.copy(data_masked)
        data_mask0[~wherefinite] = 0

        return data_masked, wherefinite, data_mask0

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

    def circle_model(self, theta, returnfull=False):
        """Return circle model as 1D array"""
        x,y = self.coords
        G   = ((x-theta['x0'])**2+(y-theta['y0'])**2)/theta['r1']**2
        Ir  = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
        Ir = self.convolve_with_gaussian(Ir)
        if returnfull:
            # Return full model as 2D array for plotting
            return Ir

        if self.regrid: 
            # Have to regrid the model and compare to regridded data
            Ir = regridding(Ir, self.iminfo)

        # Remove model points where we don't have data. Either because its masked or NaN                    
        Ir = Ir[self.wherefinite]
        return Ir

    def pre_mcmc_func(self, obj, *theta):
        """Simply returns the convolved circle model"""
        theta = {'I0':theta[0], 'x0':theta[1],'y0':theta[2],'r1':theta[3]}
        # Default params
        theta['k_exp'] = 0; theta['off'] = 0

        model = self.circle_model(theta)

        return model

    def pre_mcmc_fit(self, p0):
        """Do a simple curve_fit to estimate starting params for halo"""

        # Circle model case, boundary on parameters
        bounds = ([0.0,0.0,0.0,0.0]
             ,[np.inf
             , self.data_original.shape[0], self.data_original.shape[1]
             , self.data_original.shape[0]*3]
             )
        
        # Dont have to check finite because I already remove NaNs
        # Call pre_mcmc_func with f(self, *params) and make it match data_noNaN
        opt, pcov = curve_fit(self.pre_mcmc_func, self, self.data_mcmc
                            ,p0=p0, bounds=bounds, check_finite=True) 

        return opt # initial guess I0,x0,y0,r_e in Jy/beam and pixel coords

    def runMCMC(self, theta_guess, walkers=12, steps=400, burninfrac=0.25, save=True):
        """
        Start MCMC with initial guess theta_guess
        """

        ## TODO Maybe define this in the main class?
        self.walkers = walkers
        self.steps = steps
        self.theta_guess = theta_guess

        pos = [theta_guess*(1.+1.e-3*np.random.randn(self.dim)) for i in range(self.walkers)]

        num_CPU = cpu_count()
        with Pool(num_CPU) as pool:
            sampler = emcee.EnsembleSampler(self.walkers, self.dim, lnprob, pool=pool,
                            args=[self.data_mcmc, self.iminfo, self.circle_model])
            sampler.run_mcmc(pos, self.steps, progress=True)

        self.sampler_chain = sampler.chain
        self.burntime = int(burninfrac * steps)
        # nwalkers by nsteps-burnin by number of parameters: (12,300,4)
        self.samples = self.sampler_chain[:,self.burntime:,:]#.reshape((-1,self.dim))
        self.getpercentiles()

        if save:
            self.savechain()

        return sampler

    def calculate_chi2(self):
        """Calculate (reduced) chi2 value"""

        # Naive guess of number of DOF = len(data) - len(parameters)
        DOF = len(self.data_mcmc) - self.dim
        
        # Best fit model in 1D
        model = self.circle_model(self.bestfitp, returnfull=False)

        self.chi2 = np.sum( ((self.data_mcmc-model)/(self.rmsregrid))**2. )



    def getpercentiles(self):
        """From the burn-in version of the chain, get percentiles"""
        percentiles = np.ones((self.samples.shape[-1],3)) #(4,3) # 4 params, 3 percentiles
        for i in range(self.samples.shape[-1]):
            percentiles[i,:] = np.percentile(self.samples[:, :, i], [16, 50, 84])
        self.percentiles = percentiles
        
        self.bestfitp = {}
        self.bestfitp['I0'] = percentiles[0,1]
        self.bestfitp['x0'] = percentiles[1,1]
        self.bestfitp['y0'] = percentiles[2,1]
        self.bestfitp['r1'] = percentiles[3,1]
        # Default params
        self.bestfitp['k_exp'] = 0; self.bestfitp['off'] = 0

    def savechain(self):
        """Called by runMCMC if save=True"""
        chainpath = f"{self.output_dir}{self.imname.replace('.fits','')}_chain.fits"
        hdu      = fits.PrimaryHDU()
        # Save entire chain without burn in period removed
        hdu.data = self.sampler_chain
        hdu = self.set_sampler_header(hdu)
        print(f"Saving chain to {chainpath}")
        hdu.writeto(chainpath, overwrite=True)

        # Save the sampler header in the class
        self.sampler_header = hdu.header
        return

    def set_sampler_header(self, hdu):
        hdu.header['nwalkers'] = (self.walkers)
        hdu.header['steps']    = (self.steps)
        hdu.header['dim']      = (self.dim)
        hdu.header['burntime'] = (self.burntime)
        hdu.header['OBJECT']   = (self.imname,'Input fits file')
        hdu.header['IMAGE']    = (self.image)
        hdu.header['UNIT_0'] = ('JY/beam','unit of fit parameter 0')
        hdu.header['UNIT_1'] = ('PIX','unit of fit parameter 1')
        hdu.header['UNIT_2'] = ('PIX','unit of fit parameter 2')
        hdu.header['UNIT_3'] = ('PIX','unit of fit parameter 3')

        for i in range(len(self.theta_guess)):
            hdu.header['INIT_'+str(i)] = (self.theta_guess[i], 'MCMC initial guess')

        hdu.header['MASK'] = (self.mask,'Mask file used')

        return hdu

    def loadMCMC(self):
        """Load chain from a previous run"""
        chainpath = f"{self.output_dir}{self.imname.replace('.fits','')}_chain.fits"
        with fits.open(chainpath) as hdu:
            self.sampler_chain = hdu[0].data
            self.sampler_header = hdu[0].header
            # remove burnin
            self.samples = self.sampler_chain[:, self.sampler_header['burntime']:, :]
        self.getpercentiles()

    def print_bestfitparams(self, percentiles):
        """
        After runMCMC() call this to print best fit params
        """
        unitstr = ['Jy/beam', 'pixel', 'pixel','pixel'] # in case no units
        
        for i in range(self.dim):
            low, mid, up = percentiles[i]

            if type(mid) == u.Quantity:
                # Then we have units
                unitstr[i] = str(mid.unit)
                low, mid, up = low.value, mid.value, up.value

            if i == 0:
                print(f"{self.labels[i]} = {mid:.1E}^+{(up-mid):.1E}_-{(mid-low):.1E} {unitstr[i]}")
            else:
                print(f"{self.labels[i]} = {mid:.1F}^+{(up-mid):.1E}_-{(mid-low):.1E} {unitstr[i]}")

    def plot_data_model_residual(self, plotregrid=False, vmin=None, vmax=None, savefig=None):
        """Plot the data-model-residual plot"""
        if plotregrid:
            print("TODO")

        else:

            data = self.data_original
            model = self.circle_model(self.bestfitp, returnfull=True)
            residual = data-model

            if vmin is None:
                vmin = -2.*self.iminfo['imagerms']
            if vmax is None:
                vmax = 25.*self.iminfo['imagerms']
            NORMres = mplc.Normalize(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)
            fig.set_size_inches(3.2*5,5.1)

            im0 = axes[0].imshow(data,cmap='inferno', origin='lower', norm = NORMres)
            axes[0].set_title("Data")
            # Add and remove such that panels are same size
            cbar = fig.colorbar(im0,ax=axes[0],fraction=0.046, pad=0.04)
            # cbar.remove()
            plt.tight_layout()

            im1 = axes[1].imshow(model,cmap='inferno', origin='lower', norm = NORMres)
            axes[1].set_title("Model")
            # Add and remove such that panels are same size
            cbar = fig.colorbar(im1,ax=axes[1],fraction=0.046, pad=0.04)
            # cbar.remove()
            plt.tight_layout()

            im2 = axes[2].imshow(residual,cmap='inferno', origin='lower', norm = NORMres)
            axes[2].set_title("Residual = Data - Model")
            cbar = fig.colorbar(im2,ax=axes[2],fraction=0.046, pad=0.04)
            cbar.set_label("Intensity [Jy beam$^{-1}$]")
            plt.tight_layout()


            axes[0].set_ylabel('Pixels')
            for ax in axes.flat:
                ax.set_xlabel('Pixels')

            fig.subplots_adjust(hspace=0.01)
            if savefig is not None: plt.savefig(savefig.replace('.pdf','_residual.pdf'))
            # plt.show()
            plt.close()

    def convert_units(self):
        """Convert from pixel units to physical units"""

        # samples with removed burnin, with units.
        self.samples_I0 = self.samples[:, :, 0] * u.Jy # in Jy/beam
        # From Jy/beam to Jy/arcsec^2
        self.samples_I0 /= self.iminfo['beam_area']
        self.samples_I0 = self.samples_I0.to(u.Jy/u.arcsec**2)

        # From pixel to RA, DEC
        self.samples_x0, self.samples_y0 = self.wcs1.celestial.wcs_pix2world(self.samples[:, :, 1],self.samples[:, :, 2],1)
        # In units of degrees
        self.samples_x0 *= u.deg; self.samples_y0 *= u.deg

        # From pixel to arcsec
        self.samples_re = self.samples[:, :, 3] * self.iminfo['pix_size'].to(u.arcsec)
        # From arcsec to kpc
        self.samples_re *= cosmo.kpc_proper_per_arcmin(self.redshift).to(u.kpc/u.arcsec)

        # Calculate total flux density
        # From Jy/arcsec**2 to Jy/kpc**2
        self.samples_I0_kpc = self.samples_I0 / (cosmo.kpc_proper_per_arcmin(self.redshift).to(u.kpc/u.arcsec)**2)
        
        percentiles_units = np.ones((self.samples.shape[-1],3)) #(4,3) # 4 params, 3 percentiles
        percentiles_units = [] #List of quantities: (4,3) # 4 params, 3 percentiles
        for i, samples in enumerate([self.samples_I0,self.samples_x0,self.samples_y0, self.samples_re]):
            # percentiles_units[i,:] = np.percentile(samples[:, :], [16, 50, 84])
            percentiles_units.append(np.percentile(samples[:, :], [16, 50, 84]))
        self.percentiles_units = percentiles_units

        # Integrated up to infinity
        totalflux = 2*np.pi*self.samples_I0_kpc*self.samples_re**2
        # Integrating up to 2.6 r_e  *= 0.73
        print(f"Best-fit total flux density is {np.median(totalflux):.1f}")

        print ("TODO: check whether total flux calculation is correct")

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

def plotMCMC(samples, pinit, savefig=None, show=False):
    """
    Plot MCMC chain and initial guesses
    """
    labels = ['$I_0$', '$x_0$', '$y_0$', '$r_e$']
    dim = samples.shape[-1]

    ########## CORNER PLOT ###########
    fig = plt.figure(figsize=(6.64*2,6.64*2*0.74))
    fig = corner.corner(samples.reshape((-1,dim)),labels=labels, quantiles=[0.160, 0.5, 0.840],
                        truths=pinit, # plot blue line at inital value
                        show_titles=True, title_fmt='.5f', fig=fig)
    
    if savefig is not None: plt.savefig(savefig.replace('.pdf','_corner.pdf'))
    if show: plt.show()
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

    if savefig is not None: plt.savefig(savefig.replace('.pdf','_walkers.pdf'))
    if show: plt.show()
    plt.close()

def rotate_image(data, iminfo):
    """Rotate image
    data   -- 2d array   -- Should have no NaNs 
    iminfo -- dictionary -- Should have beam position angle 'bpa' 

    """
    img_rot = ndimage.rotate(data, -iminfo['bpa'].value, reshape=False)
    return img_rot

def regrid_to_beamsize(data, iminfo, accuracy=100., func=np.sum):
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

    if accuracy == 1:
        pseudo_array = np.copy(data)
    else:
        pseudo_array = np.zeros((pseudo_size))
        for j in range(data.shape[0]):
            for i in range(data.shape[1]):
                pseudo_array[orig_scale[1]*i:orig_scale[1]*(i+1),
                             orig_scale[0]*j:orig_scale[0]*(j+1)] = data[i,j]/elements

    # subsampled array where approx 1 pixel is 1 beam.
    f= block_reduce(pseudo_array, block_size=tuple(scale), func=func, cval=0)
    return f

def regridding(data, iminfo, func=np.sum):
    """Rotate image
    iminfo -- dictionary -- Should have beam position angle 'bpa' 
                            and 

    RETURNS
    IMAGE THAT IS ROTATED AND REGRIDDED TO 1 PIXEL IS 1 BEAM
    """    
    data_rot = rotate_image(data, iminfo)
    regrid   = regrid_to_beamsize(data_rot, iminfo, func=func)
    return regrid


def plottwofigures(data1, data2, savefig=None, titles=['',''],show=False):
    """
    For checking stuff
    """

    fig, axes = plt.subplots(1,2)

    for i, data in enumerate([data1,data2]):
        im = axes[i].imshow(data,origin='lower')
        cbar = fig.colorbar(im,ax=axes[i])
        axes[i].set_title(titles[i])

    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    else:
        plt.close()
    return

def check_createdir(directory):
    if not os.path.isdir(directory):
        print(f'Creating directory {directory}')
        os.makedirs(directory)
    return

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