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
from astropy.visualization import SqrtStretch, ImageNormalize, ManualInterval

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
import pyregion # for plotting regions

from . import utils
utils.paper_fig_params()


class MCMCfitter(object):
    """MCMC fitter class

    image      -- str   -- location of fits image
    rms        -- float -- rms in Jy/beam
    regrid     -- bool  -- whether to regrid to 1pix=1beam
    mask       -- str   -- mask file to remove sources
    rms_region -- str   -- where to calculate the (regridded) rms 
    """
    def __init__(self, image, rms, regrid=True, mask=None, output_dir='./output/', maskoutside=None, redshift=0.058, rms_region=None):
        self.image = image
        self.regrid = regrid # whether to regrid to 1 pixel approx 1 beam
        self.mask = mask
        self.maskoutside = maskoutside
        self.output_dir = output_dir
        self.redshift = redshift
        self.rms_region = rms_region # Only needed if we want chi2 value

        self.dim = 4 # Circle model
        self.labels = ['$I_0$', '$x_0$', '$y_0$', '$r_e$']

        # Load the image file, WCS information and other handy conversions
        # Note pickling WCS gives WARNING: https://github.com/astropy/astropy/issues/12834
        self.data_original, self.wcs1, self.iminfo  = self.loadfitsim(image, rms)

        # Bookkeeping of settings and directories
        self.check_settings()
    
        # Bookkeeping, store the data to use as self.data_mcmc 
        # and the mask as self.wherefinite
        self.set_data_to_use()

    def check_settings(self):
        """Check input settings"""
        check_createdir(self.output_dir)

        self.plotdir = self.output_dir + 'Plots/'
        check_createdir(self.plotdir)

        self.plotdir_png = self.output_dir + 'Plots/pngs/'
        check_createdir(self.plotdir_png)

        self.imname = self.image.split('/')[-1]
        print(f"Using image {self.imname}")

        if self.rms_region is not None:
            ### Calculate RMS inside the region

            w = self.wcs1
            while w.naxis > 2: # in case the wcs is 4 axes
                w = w.dropaxis(-1)
            self.rms_original = calculate_rms(self.data_original, w, self.rms_region)

            factor = self.rms_original / self.iminfo['imagerms']
            if 0.5 < factor < 2.0:
                pass
            else:
                print (f"WARNING: User defined rms {self.iminfo['imagerms']} is significantly ({factor}x) different from rms {self.rms_original} found inside the provided region") 
        
        ## TODO: log

    def loadfitsim(self, image, rms):
        """Load fits image and return useful info"""
        with fits.open(image) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs1 = wcs.WCS(header)
            while wcs1.naxis > 2: # in case the wcs is 4 axes
                wcs1 = wcs1.dropaxis(-1)

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
        w = self.wcs1 # with 2 axes only
        
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

    def zoom_data(self, radius=3.0):
        """
        Give back the xlim and ylim, so the plot_residual() function
        does not contain the full field of view.

        data     -- 2D array
        model    -- 2D array
        residual -- 2D array
        radius   -- float    -- radius in terms of r_e how big to make the image
        """
        # Central coords in pixels
        x0, y0 = self.bestfitp['x0'], self.bestfitp['y0']
        # zoom into 3R500
        radius = self.bestfitp['r1']*radius
        xlim = (x0-radius,x0+radius)
        ylim = (y0-radius,y0+radius)
        return xlim, ylim

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
        self.calculate_chi2()

        if save:
            self.savechain()

        return sampler

    def regridded_rms(self):
        """To calculate the chi2 value, we need to get the RMS in the regridded image"""

        # So we need to have a user defined region where we can calculate the RMS
        # regrid that region [T/F mask], and re-calculate the RMS in the re-gridded data in that region

        r = Regions.read(self.rms_region)
        rmask = r[0].to_pixel(self.wcs1).to_mask()
        # True where inside region, 0 where not, same shape as data
        whererms = np.array(rmask.to_image(self.data_original.shape),dtype='bool')

        # Regrid this region
        self.whererms = regridding(whererms, self.iminfo, func=np.max)
        # Regrid original data. 
        data_original_rebin = np.copy(self.data_original)
        # Make sure there's no nans
        if np.isnan(data_original_rebin).any():
            print("WARNING: NaNs found in the input image")
            data_original_rebin[np.isnan(data_original_rebin)] = 0

        data_original_rebin = regridding(data_original_rebin, self.iminfo)
        rmsregion = data_original_rebin[self.whererms]

        self.rms_regrid = np.sqrt(1./len(rmsregion)*np.sum(rmsregion**2))
        return self.rms_regrid

    def calculate_chi2(self):
        """Calculate (reduced) chi2 value"""

        # Naive guess of number of DOF = len(data) - len(parameters)
        self.DOF = len(self.data_mcmc) - self.dim
        
        # Best fit model in 1D
        model = self.circle_model(self.bestfitp, returnfull=False)

        # Caveat: to get a good value of chi2, we need to know the error per pixel
        # If we are regridding, the rms is hard to determine analytically
        # So we need a user provided region to calculate the rms
        if self.regrid:
            if self.rms_region is None:
                print("WARNING: Cannot calculate chi2 value. Need a region where the RMS is calculated to calculate the chi2 value.")
                return -1

            rms = self.regridded_rms()
        else:
            print("WARNING: If self.regrid=False, then chi2 value will be wrong. Can still use for model comparison. Perhaps better to look at the chi2 of the annulus plot.")
            # WARNING: If we are not regridding, the pixels are not independent
            # So DOF is (greatly) overestimated.
            rms = self.iminfo['imagerms'] # Just use image rms input by user

        self.chi2 = np.sum( ((self.data_mcmc-model)/(rms))**2. )
        self.chi2_red = self.chi2/self.DOF

        print(f"chi2 value: {self.chi2:.1F} | DOF = {self.DOF:.1F} | chi2/DOF = {self.chi2_red:.1F}")
        return 

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
        hdu.header['regrid']   = (self.regrid)
        hdu.header['OBJECT']   = (self.imname,'Input fits file')
        hdu.header['IMAGE']    = (self.image)
        hdu.header['UNIT_0'] = ('JY/beam','unit of fit parameter 0')
        hdu.header['UNIT_1'] = ('PIX','unit of fit parameter 1')
        hdu.header['UNIT_2'] = ('PIX','unit of fit parameter 2')
        hdu.header['UNIT_3'] = ('PIX','unit of fit parameter 3')
        hdu.header['chi2']   = (self.chi2)
        hdu.header['DOF']    = (self.DOF)
        hdu.header['chi2_red'] = (self.chi2_red) 

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
        self.calculate_chi2()

    def print_bestfitparams(self, percentiles):
        """
        After runMCMC() call this to print best fit params
        """
        unitstr = ['Jy/beam', 'pixel', 'pixel','pixel'] # in case no units
        
        print("============")
        for i in range(self.dim):
            low, mid, up = percentiles[i]

            if type(mid) == u.Quantity:
                # Then we have units
                unitstr[i] = str(mid.unit)
                low, mid, up = low.value, mid.value, up.value

            if i == 0:
                print(f"{self.labels[i]} = {mid:.3E}^+{(up-mid):.1E}_-{(mid-low):.1E} {unitstr[i]}")
            elif i == 1 or i == 2:
                print(f"{self.labels[i]} = {mid:.5F}^+{(up-mid):.1E}_-{(mid-low):.1E} {unitstr[i]}")
            else:
                print(f"{self.labels[i]} = {mid:.1F}^+{(up-mid):.1E}_-{(mid-low):.1E} {unitstr[i]}")
        print("============\n")

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

        # Also save best fit params with physical units in a dictionary
        self.bestfitp_units = {}
        self.bestfitp_units['I0'] = percentiles_units[0][1]
        self.bestfitp_units['x0'] = percentiles_units[1][1]
        self.bestfitp_units['y0'] = percentiles_units[2][1]
        self.bestfitp_units['r1'] = percentiles_units[3][1]
        # Default params
        self.bestfitp_units['k_exp'] = 0; self.bestfitp_units['off'] = 0

    def totalflux(self, d=np.inf, rkpc=None):
        """
        Can only be ran after convert_units() is called. 
        """

        if rkpc is not None:
            # Integrating up to a certain amount of kpc 
            d = (rkpc*u.kpc/self.percentiles_units[3][1]).to(1).value
            print (f"User defined: integrating up to {rkpc:.1f} kpc = {d:.1f}r_e")

        if d == np.inf:
            # Integrated up to infinity
            totalflux = 2*np.pi*self.samples_I0_kpc*self.samples_re**2
            print(f"Best-fit total flux density is {np.median(totalflux):.1f} integrated up to infinity")
        else:
            # Integrating up to a certain fraction of r_e
            totalflux = 2*np.pi*self.samples_I0_kpc*self.samples_re**2
            totalflux *= (1-np.exp(-d) *(d+1)) # e.g. d=2.6 gives fraction of 0.73
            print(f"Best-fit total flux density is {np.median(totalflux):.1f} integrated up to {d:.1f}r_e")

        return totalflux

    def plot_data_model_residual(self, plotregrid=False, vmin=None, vmax=None, savefig=None, presentation=False, sqrtstretch=True, add1D=True, zoomresidual=None):
        """Plot the data-model-residual plot"""
        if plotregrid:
            print("TODO: Plot regridded versions")
            return

        else:

            data = self.data_original
            model = self.circle_model(self.bestfitp, returnfull=True)
            residual = data-model

            if vmin is None:
                vmin = -2.*self.iminfo['imagerms']
            if vmax is None:
                if sqrtstretch:
                    vmax = 40.*self.iminfo['imagerms']
                else:
                    vmax = 25.*self.iminfo['imagerms']

            if sqrtstretch:
                NORMres = ImageNormalize(data, ManualInterval(vmin,vmax), stretch=SqrtStretch())
            else:
                NORMres = mplc.Normalize(vmin=vmin, vmax=vmax)

            # Load mask
            r = pyregion.open(self.mask).as_imagecoord(header=self.iminfo['header'])
            patch_list, _ = r.get_mpl_patches_texts()

            # Load mask_outside
            r_out = pyregion.open(self.maskoutside).as_imagecoord(header=self.iminfo['header'])
            patch_out, _ = r_out.get_mpl_patches_texts()

            # Plot
            if add1D:
                fig, axes = plt.subplots(ncols=4, nrows=1, sharey=False)
                fig.set_size_inches(21.3,5.1)
            else:
                fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)
                fig.set_size_inches(3.2*5,5.1)
        
            transparent = False
            titlecolor = 'k'
            if presentation:
                utils.white_axes(axes)
                transparent = True
                titlecolor='w'

            im0 = axes[0].imshow(data,cmap='inferno', origin='lower', norm = NORMres)
            axes[0].set_title("Data",color=titlecolor)
            # Add and remove such that panels are same size
            cbar = fig.colorbar(im0,ax=axes[0],fraction=0.046, pad=0.04)
            cbar.ax.tick_params(axis='both', colors=titlecolor,which='both')
            # cbar.remove()
            plt.tight_layout()
            # Plot the masked sources
            for i, p in enumerate(patch_list):
                p.set_edgecolor('g')
                axes[0].add_patch(p)
            # Plot the mask_outside region
            for i, p in enumerate(patch_out):
                p.set_edgecolor('gray')
                axes[0].add_patch(p)


            im1 = axes[1].imshow(model,cmap='inferno', origin='lower', norm = NORMres)
            axes[1].set_title("Model",color=titlecolor)
            # Add and remove such that panels are same size
            cbar = fig.colorbar(im1,ax=axes[1],fraction=0.046, pad=0.04)
            cbar.ax.tick_params(axis='both', colors=titlecolor,which='both')
            # cbar.remove()
            plt.tight_layout()

            im2 = axes[2].imshow(residual,cmap='inferno', origin='lower', norm = NORMres)
            axes[2].set_title("Residual = Data - Model",color=titlecolor)
            cbar = fig.colorbar(im2,ax=axes[2],fraction=0.046, pad=0.04)
            cbar.ax.tick_params(axis='both', colors=titlecolor,which='both')
            cbar.set_label("Intensity [Jy beam$^{-1}$]",color=titlecolor)
            plt.tight_layout()


            axes[0].set_ylabel('Pixels')
            if zoomresidual is not None: 
                xlim, ylim = self.zoom_data(zoomresidual)
            else:
                xlim, ylim = axes[0].get_xlim(), axes[0].get_ylim()
            for ax in axes.flat[:3]:
                ax.set_xlabel('Pixels')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            if add1D:
                self.plot_1D(d=3.0, d_int_kpc=None, savefig=None, saveradial=None,ax=axes[3],show=False,close=False)
                # fig.subplots_adjust(hspace=0.01)
                plt.tight_layout()
            else:
                fig.subplots_adjust(hspace=0.01)

            if savefig is not None: plt.savefig(savefig.replace('.pdf','_residual.pdf'),transparent=transparent)
            # plt.show()
            plt.close()
        return

    def plot_1D(self, d=3.0, d_int_kpc=None, savefig=None, plotconvolvedmodel=False, show=False, close=True, presentation=False, saveradial=None, ax=None):
        """Plot 1D annulus and model"""

        if d_int_kpc is not None:
            d = (d_int_kpc*u.kpc/self.percentiles_units[3][1]).to(1).value

        # width of annuli should be 1 beam
        width = (self.iminfo['bmaj'])/self.iminfo['pix_size']
        # in pixel coords
        width = width.to(1).value

        # Amount of pixels in 1 beam:
        NpixIn1Beam = (self.iminfo['beam_area']/self.iminfo['pix_size']**2).value

        ####### Calculate profile on the (masked) version of the original data
        radius, profile, uncertainty = utils.radialprofile(self.data_original, wcs=self.wcs1
            , x0_pix=self.bestfitp['x0'], y0_pix=self.bestfitp['y0']
            , maskoutside=self.maskoutside, maskinside=self.mask
            , pixradius=self.bestfitp['r1']*d, width=width, rms=self.iminfo['imagerms']
            , NpixIn1Beam=NpixIn1Beam)

        # Convert radius from pixel to kpc
        radius_kpc = radius * self.iminfo['pix_size'].to(u.arcsec) # From pixel to arcsec
        radius_kpc *= cosmo.kpc_proper_per_arcmin(self.redshift).to(u.kpc/u.arcsec) # From arcsec to kpc
        radius_kpc = radius_kpc.to(u.kpc).value

        ####### Calculate profile on the best-fit smoothed model
        model = self.circle_model(self.bestfitp, returnfull=True)

        _, modelprofile, _ = utils.radialprofile(model, wcs=self.wcs1
            , x0_pix=self.bestfitp['x0'], y0_pix=self.bestfitp['y0']
            , maskoutside=self.maskoutside, maskinside=self.mask
            , pixradius=self.bestfitp['r1']*d, width=width, rms=self.iminfo['imagerms']
            , NpixIn1Beam=NpixIn1Beam)

        ####### Calculate analytical profile
        I0 = self.bestfitp['I0'] # mind that this one is Jy/b
        r_e = self.bestfitp_units['r1'].to(u.kpc).value # and this one is kpc
        radius_kpc_analytical = np.linspace(np.min(radius_kpc),np.max(radius_kpc),100)
        analytical = I0 * np.exp(-radius_kpc_analytical/r_e)

        ####### Calculate chi2 between average in annulus and analytical model
        analytical_at_data = I0 * np.exp(-radius_kpc/r_e)
        residuals = analytical_at_data-profile
        self.chi2_annulus = np.nansum( (residuals/uncertainty)**2 )
        self.DOF_annulus = np.sum(np.isfinite(residuals))-4 # Number of (valid) datapoints - params
        self.chi2_red_annulus = self.chi2_annulus/self.DOF_annulus

        print(f"Radial profile chi2 value: {self.chi2_annulus:.1F} | DOF = {self.DOF_annulus:.1F} | chi2/DOF = {self.chi2_red_annulus:.1F}")

        ####### Plot
        if ax is None:  
            fig, ax = plt.subplots()
        else: # Its a subplot
            # Plot with y label and ticks on the right side.
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_aspect('equal')
            ax.set_title('Circular annuli')

        transparent = False
        if presentation:
            utils.white_axes()
            transparent = True

        msize = 4 #marker size
        mew = 1 # marker edge width
        elw = 1 # error line width
        ax.errorbar(radius_kpc, profile, yerr=uncertainty
            ,marker='s', markeredgecolor='k', color='C0', markersize=msize
            ,elinewidth=elw,alpha=1.0,capsize=3.0, label='Data',zorder=0)
        ax.plot(radius_kpc_analytical, analytical,color='C1',label='Best-fit model',zorder=1)
        if plotconvolvedmodel:
            ax.plot(radius_kpc, modelprofile
                ,color='C2',label='Best-fit convolved model',zorder=2)

        
        ax.axvline(r_e,ls='dashed',color='k',alpha=0.5,label=f'$r_e$={r_e:.0f} kpc',zorder=3)

        ax.set_xlabel('Annulus central radius [kpc]')
        ax.set_ylabel('Average Intensity [Jy/beam]')
        plt.legend(ncol=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        xlim = np.array(ax.get_xlim())
        # Start plot xlim at least at 10 kpc and end at least at 1000 kpc
        if xlim[0] > 1e1:
            xlim[0] = 0.9e1
        if xlim[1] < 1e3:
            xlim[1] = 1.1e3
        ax.set_xlim(xlim)

        if savefig is not None: 
            plt.savefig(savefig.replace('.pdf','_annulus.pdf'),transparent=transparent)
            # also save as png
            savefig = savefig.replace('/Plots/','/Plots/pngs/')
            plt.savefig(savefig.replace('.pdf','_annulus.png'),transparent=transparent)
        if show: plt.show()
        if close: plt.close()

        # Store values in class
        self.radius_annuli = radius_kpc
        self.data_annuli = profile
        self.uncertainty_annuli = uncertainty
        self.radius_annuli_model = radius_kpc_analytical
        self.model_annuli = analytical
        self.r_e = r_e

        if saveradial is not None:
            np.save(saveradial, np.array([radius_kpc, profile, uncertainty]))

def lnprob(theta, data, info, modelf):
    """Log posterior, inputting a vector of parameters and data"""    

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
    """Log likelihood, inputting a vector of parameters and data"""   
    model = modelf(theta)

    ## NOTE THAT IF WE REGRID, imagerms IS WRONG. 
    ## BUT BECAUSE IT'S A CONSTANT, AND WE MAXIMIZE THE LIKELIHOOD, IT DOESNT MATTER
    return -np.sum( ((data-model)**2.)/(2*info['imagerms']**2.)\
                        + np.log(np.sqrt(2*np.pi)*info['imagerms']) )

def lnprior(theta, info):
    """Log prior, checks the boundaries of the params"""
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
    """Returns a dictionary with all image information in one place"""
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
        "header":            header, # the rest of the header
        }
    return image_info

def calculate_rms(data, wcs, regionfile):
    """Calculate RMS in the region provided by the user"""
    r = Regions.read(regionfile)
    if len(r)>1:
        raise ValueError(f"{len(r)} regions found in RMS region file. Expected 1.")
    
    rmask = r[0].to_pixel(wcs).to_mask()
    # True where inside region, 0 where not, same shape as data
    rmask = np.array(rmask.to_image(data.shape),dtype='bool')
    # calculate rms inside the region
    data_region = data[rmask]
    rms = np.sqrt (1./len(data_region) * np.sum(data_region**2))
    return rms

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