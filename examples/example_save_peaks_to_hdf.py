# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:37:44 2020

@author: mauricio
Example: save peak info to hdf
"""
from nasagamma import spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
import lmfit

# dataset 1
file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key="data")
df = df.iloc[1:, :]


cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)

# Required input parameters (always in channels)
fwhm_at_0 = 1.0
ref_fwhm = 31
ref_x = 1220
min_snr = 1

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np, energies=erg, e_units="MeV")

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
# search.plot_peaks()

bkgs = [
    "poly2",
    "poly2",
    "poly1",
    "poly1",
    "poly1",
    "poly2",
    "poly1",
    "poly1",
    "poly1",
    "poly1",
    "poly2",
]

ranges_m = np.array(
    [
        [71, 103],
        [125, 161],
        [171, 193],
        [274, 356],
        [411, 445],
        [503, 691],
        [702, 789],
        [850, 906],
        [986, 1038],
        [1087, 1450],
    ]
)

ranges_e = [
    [0.45, 0.61],
    [0.72, 0.9],
    [0.95, 1.1],
    [1.46, 1.87],
    [2.13, 2.3],
    [2.58, 3.52],
    [3.57, 4.0],
    [4.3, 4.6],
    [4.98, 5.23],
    [5.44, 5.78],
    [6.0, 7.4],
]


n = 10
fit0 = pf.PeakFit(search, ranges_e[n], bkg=bkgs[n])
fit0.plot(plot_type="full")

# save peaks
fitted_peaks = pf.AddPeaks("fitted_peaks_delete2")
for ran, bk in zip(ranges_m, bkgs):
    fit = pf.PeakFit(search, ran, bkg=bk)
    # fit.plot(plot_type="full")
    fitted_peaks.add_peak(fit)
