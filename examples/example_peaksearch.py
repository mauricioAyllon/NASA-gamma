# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:24:22 2020

@author: mauricio
Example of peaksearch functionality
"""

from nasagamma import spectrum as sp
import numpy as np
import pandas as pd
from nasagamma import peaksearch as ps

# dataset 1
file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key="data")

# delete first (large) bin
df = df.iloc[1:, :]

cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np, energies=erg)

# Required input parameters (in channels)
fwhm_at_0 = 1
ref_fwhm = 35
ref_x = 1220

# instantiate a peaksearch object
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=0.1)

search.plot_kernel()

search.plot_peaks()

search.plot_components()
