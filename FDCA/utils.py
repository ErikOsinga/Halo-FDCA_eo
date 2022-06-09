import os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import coordinates
from astropy.table import Table, Column
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
from regions import Regions
from scipy.optimize import leastsq

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def argsfromfile(bashfile):
    """Read args from bashfile for interactive python usage"""
    with open(bashfile) as f:
        foundcall = False
        while not foundcall:
            line = f.readline()
            if 'python' in line:
                foundcall = True

        line = line.strip('\n')
        # Assumes first two words are 'python HaloFitting_eo.py'
        arguments = ' '.join(line.split(' ')[2:])

    return arguments

def paper_fig_params(TW=6.64, AR=0.74, FF=1., fontsize=16.0, fontst=["Times New Roman"
                    ,"Computer Modern Roman", "STIXGeneral"]):
    """ 
    Set figure parameters. TW is figure width. TW*AR is Figure height in inches.
    FF is a multiplier. 

    Modified from Wendy
    """
    mpl.rc('figure', figsize=(FF*TW, FF*TW*AR), dpi=100)
    mpl.rc('figure.subplot', left=0.15, right=0.95, bottom=0.15, top=0.92)
    mpl.rc('lines', linewidth=1.75, markersize=8.0, markeredgewidth=0.75)
    mpl.rc('font', size=fontsize, family="serif", serif=fontst)
    mpl.rc('xtick', labelsize='small')
    mpl.rc('ytick', labelsize='small')
    mpl.rc('xtick.major', width=1.0, size=8)
    mpl.rc('ytick.major', width=1.0, size=8)
    mpl.rc('xtick.minor', width=1.0, size=4)
    mpl.rc('ytick.minor', width=1.0, size=4)
    mpl.rc('axes', linewidth=1.5)
    mpl.rc('legend', fontsize='small', numpoints=1, labelspacing=0.4, frameon=False) 
    mpl.rc('text', usetex=True) 
    mpl.rc('savefig', dpi=300)

def standard_error(data, NpixIn1Beam=1):
    """Standard error on the mean of Gaussian distributed datapoints"""
    return np.nanstd(data)/np.sqrt(np.sum(np.isfinite(data))/NpixIn1Beam)

def radialprofile(data, header=None, wcs=None, RA=None, DEC=None, x0_pix=None, y0_pix=None
    ,maskoutside=None,pixradius=100, width=1, maskinside=None, rms=None, NpixIn1Beam=None):
    """
    Calculate radial profile starting from central point RA,DEC in deg or x0 y0 in pixel coords

    maskinside -- mask all data inside these regions
    maskoutside -- mask all data outside these regions
    width       -- width in pixel of the annuli
    """
    if header is not None:
        w = WCS(header)
    else:
        w = wcs

    if RA is None:
        # Use input pixel coords
        x0 = x0_pix
        y0 = y0_pix
    else: # Convert RA DEC to pixel coords
        sc = np.array([[RA,DEC]])
        x0, y0 = w.celestial.wcs_world2pix(sc,1)[0]
        x0 = int(x0)
        y0 = int(y0)

    data_masked = np.copy(data)
    if maskoutside is not None:
        # Mask all data outside the region called maskoutside
        # Load the mask
        r = Regions.read(maskoutside)
        for i in range(len(r)):
            rmask = r[i].to_pixel(w).to_mask()
            # True where inside region, 0 where not, same shape as data
            rmask = np.array(rmask.to_image(data.shape),dtype='bool')
            # Mask outside the region
            data_masked[~rmask] = np.nan        

    if maskinside is not None:
        # Load the mask
        r = Regions.read(maskinside)
        for i in range(len(r)):
            rmask = r[i].to_pixel(w).to_mask()
            # True where masked, 0 where not, same shape as data
            rmask = np.array(rmask.to_image(data.shape),dtype='bool')
            data_masked[rmask] = np.nan

    x,y = np.meshgrid(np.arange(data_masked.shape[1]),np.arange(data_masked.shape[0]))
    R = np.sqrt( (x-x0_pix)**2 + (y-y0_pix)**2)
    
    # Define inner radii at the annuli
    r = np.arange(1,pixradius,step=width)
    # take as the radius of the annulus the half-way point between inner and outer
    r_annulus = r+width/2

    # function to calculate the mean at inner radius r until r+width
    # i.e. the average intensity
    f = lambda r : np.nanmean(data_masked[(R >= r) & (R < r+(width))])
    mean = np.vectorize(f)(r)

    # for the error, if we know the RMS (i.e. uncertainty per pixel/beam) we can calculate it
    if rms is not None and NpixIn1Beam is not None:
        # Standard error of the mean of N datapoints with Gaussian errors is simply the std/sqrt(N)
        # However, we must take into account that pixels inside one beam are not independent
        # so N is not the number of pixels but the number of beams.
        # This is an approximation works best when there is no masking. 
        f_unc = lambda r: standard_error(data_masked[(R >= r) & (R < r+(width))],NpixIn1Beam)
        uncertainty = np.vectorize(f_unc)(r)
    else:
        print("WARNING: RMS or amount of pixels in 1 beam not given. Cannot calculate uncertainty on radial profile.") 
        uncertainty = -1



    return r_annulus, mean, uncertainty # In pixels, Jy/beam, Jy/beam

def stokesImodel(p, nu):
    """
    Simple power-law model 
        I = I0*nu**alpha
    """
    I0 = p[0]
    alpha = p[1]
    return I0 * nu**(alpha)

def stokesImodelcurved(p, nu):
    """
    Curved power law model (log-parabolic) See Eq. 8 gabri paper
    """
    nu_ref = 1. # 1 GHz for simplicity.

    I0 = p[0]
    alpha = p[1]
    b = p[2]
    return I0*nu**(alpha+b*np.log10(nu/nu_ref))

def fit_spix(stokesI, noiseI, nu,  curvature=False, uncertainty=False, model=None, guess=None):
    """
    Do a least square fit of Stokes I data with a simple power law model
    stokesI   -- array -- array with fluxes
    noiseI    -- array -- array with uncertainty on fluxes
    nu        -- array -- array with Freq where fluxes are measured
    curvature -- bool  -- whether to use curved model

    Returns
    pbest -- best fit parameters for model. [I0, alpha, (b)]
    chi2  -- best fit chi2 value
    """
    Iguess = np.mean(stokesI)

    if model is None:
        # Use implemented models
        if curvature:
            guess = [Iguess, -1.0, 0]
            model = stokesImodelcurved
        else:
            guess= [Iguess, -1.0] #I0, alpha
            model = stokesImodel

    else:
        if guess is None:
            raise ValueError("When giving model manually, also give guess for bestfit params")
            
    # Chi2 error, but not squared
    errfunc = lambda p, nu, fluxI, errI: (fluxI - model(p, nu))/errI

    out = leastsq(errfunc, guess
                        , args=(nu, stokesI, noiseI), full_output=True)
    pbest = out[0]
    pbest_invH = out[1]

    dof = len(nu)-len(pbest) # dof = number of obs - number of fitted params
    
    chi2red = np.sum(errfunc(pbest, nu, stokesI, noiseI)**2)/dof

    if uncertainty:
        # also return the inverse of the Hessian
        # To obtain the covariance matrix of the parameters
        # multiply this with the variance of the residuals. 
        # See scipy.optimize.leastsq documentation
        return pbest, chi2red, pbest_invH

    return pbest, chi2red