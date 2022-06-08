import os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import coordinates
from astropy.table import Table, Column
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

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