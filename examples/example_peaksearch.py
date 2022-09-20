# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:24:22 2020

@author: mauricio
Example of peaksearch functionality
"""

from nasagamma import spectrum as sp
import pandas as pd
from nasagamma import peaksearch as ps

# dataset 1
file = "data/gui_test_data_cebr.csv"
df = pd.read_csv(file)

# instantiate a Spectrum object
spect = sp.Spectrum(counts=df["counts"])

# Required input parameters (in channels)
fwhm_at_0 = 1
ref_fwhm = 20
ref_x = 420

# instantiate a peaksearch object
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=5)
# plot kernel, peaks, and components
search.plot_kernel()
search.plot_peaks()
search.plot_components()
