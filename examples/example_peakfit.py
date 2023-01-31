# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:49:45 2020

@author: mauricio
Example using the peakfit class
"""
from nasagamma import spectrum as sp
import numpy as np
import pandas as pd
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf


# dataset 1
file = "data/gui_test_data_cebr.csv"
df = pd.read_csv(file)

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 31
ref_x = 1220
min_snr = 5

# instantiate a Spectrum object
spect = sp.Spectrum(counts=df["counts"])

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
search.plot_peaks()

# peakfit class
bkg0 = "poly1"
xrange = [1250, 1375]
fit = pf.PeakFit(search, xrange, bkg=bkg0)

res = fit.fit_result

# print(res.fit_report())

# search.plot_peaks()

fit.plot(plot_type="full", legend="on", table_scale=[2, 2.3])


## check area under the curve
def gaussian(x, A, mu, sig):
    return A / np.sqrt(2 * np.pi * sig**2) * np.exp(-((x - mu) ** 2) / (2 * sig**2))


xg = fit.x_data
Ag = res.best_values["g1_amplitude"]  # should also be the area under the curve!
mug = res.best_values["g1_center"]
sigg = res.best_values["g1_sigma"]
gauss = gaussian(xg, Ag, mug, sigg)
