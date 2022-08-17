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

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

from . import utils
utils.paper_fig_params()

msize = 4 #marker size
mew = 1 # marker edge width
elw = 1 # error line width

def compare_annuliplots(fitters, colors=['C0','C1','C2']
    , labels=['23 MHz', '46 MHz', '144 MHz'], savefig=None,show=False,close=True):


    for i, fitter in enumerate(fitters):

        plt.errorbar(fitter.radius_annuli, fitter.data_annuli, yerr=fitter.uncertainty_annuli
            ,marker='s', markeredgecolor='k', color=colors[i], markersize=msize
            ,elinewidth=elw,alpha=1.0,capsize=3.0, label=labels[i],zorder=0
            ,ls='none')
        plt.plot(fitter.radius_annuli_model, fitter.model_annuli
            ,color=colors[i],zorder=1)

        # plt.axvline(r_e,ls='dashed',color='k',alpha=0.5,label=f'$r_e$={r_e:.0f} kpc',zorder=3)

    plt.xlabel('Annulus central radius [kpc]')
    plt.ylabel('Average Intensity [Jy/beam]')
    plt.legend(ncol=1)
    plt.xscale('log')
    plt.yscale('log')

    if savefig is not None: plt.savefig(savefig.replace('.pdf','_annulus_comparison.pdf'))
    if show: plt.show()
    if close: plt.close()

    ## NORMALISED
    for i, fitter in enumerate(fitters):

        plt.errorbar(fitter.radius_annuli, fitter.data_annuli/fitter.data_annuli[0], yerr=fitter.uncertainty_annuli
            ,marker='s', markeredgecolor='k', color=colors[i], markersize=msize
            ,elinewidth=elw,alpha=1.0,capsize=3.0, label=labels[i],zorder=0
            ,ls='none')

    plt.xlabel('Annulus central radius [kpc]')
    plt.ylabel('Average Intensity [normalised]')
    plt.legend(ncol=1)
    plt.xscale('log')
    plt.yscale('log')

def compare_fluxmodels(fitters, d_int=2.6, d_int_kpc=1300, frac_unc=0.1):
    """Calculate total flux within region and calc spectral index"""

    fluxes = []
    unces = []
    freqs = np.array([23e6,46e6,144e6])
    for i, fitter in enumerate(fitters):
        totalflux = np.median(fitter.totalflux(d_int, d_int_kpc)).value
        unc = frac_unc*totalflux

        fluxes.append(totalflux)
        unces.append(unc)

    fluxes = np.array(fluxes)
    unces = np.array(unces)

    # pbest, chi2red = utils.fit_spix(fluxes, unces, freqs)
    pbest, chi2red = fit_spix(fluxes, unces, freqs)
    I0, alpha = pbest
    print ("Best alpha for model %i is: %.2f with chi2red: %.1f"%(i,pbest[1],chi2red))

    # plot data
    fig, ax = plt.subplots()
    ax.errorbar(freqs/1e6,fluxes,yerr=unces,color='k'
        ,capsize=3.0,elinewidth=elw,zorder=2,markersize=msize, alpha=1.0, linestyle='none'
        ,marker='o',label=labels[i])
    # plot model
    freqs_model = np.logspace(np.log10(np.min(freqs-1e6)),np.log10(np.max(freqs+1e6)))
    bestmodel = stokesImodel(pbest, freqs_model)
    label = None
    ax.plot(freqs_model/1e6,bestmodel,ls='dashed',label=label,color='C0',zorder=1,alpha=1.0)    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Total flux density [Jy]')
    plt.legend()