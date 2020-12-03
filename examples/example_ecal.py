# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:04:27 2020

@author: mauricio
energy calibration example
"""

import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
import matplotlib.pyplot as plt
import numpy as np

file = "data/10-23-2020-800-3.csv"
# the columns are not in a 'nice' format (blank spaces)
# so, rename them
dict0 = {" Energy (keV)": "Energy", " Counts": "Counts"}

dfi = pd.read_csv(file, nrows=5)
df = pd.read_csv(file, header=6)
df = df.rename(columns=dict0)

spect = sp.Spectrum(counts=df.Counts)
fwhm_at_0 = 1.0
ref_x = 1315
ref_fwhm = 42
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=4)

xrange = [1250, 1600]
bkg = "poly2"
fit = pf.PeakFit(search, xrange=xrange, bkg=bkg)
fit.plot(plot_type="full")

# energy calibration
ch = df.Channel.to_numpy(dtype=int)
peak_info = fit.peak_info
mean_values = [peak_info[0]["mean1"], peak_info[1]["mean2"]]
mean_values.insert(0, 0)  # add the origin
erg = [0, 1173.2, 1332.5]  # in keV

pred_erg, efit = pf.ecalibration(
    mean_vals=mean_values, erg=erg, channels=ch, n=1, plot=True, residual=True
)

pred_erg[0] = 0  # because negative entry

spect2 = sp.Spectrum(counts=df.Counts, energies=pred_erg, e_units="keV")
spect2.plot()
