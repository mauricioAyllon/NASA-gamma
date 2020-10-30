# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:14:38 2020

@author: mauricio
Example using auto_scan
"""
from nasagamma import spectrum as sp
import numpy as np
import pandas as pd
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf

# dataset 1
file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key='data')
df = df.iloc[1:,:]

cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)
chan = np.arange(0,len(cts_np),1)

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 31
ref_x = 1220
min_snr = 1

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np)

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)


## plot peak positions (channels)
search.plot_peaks()

## auto_scan
ranges_m = [[8,18],[32,44],[72,88],[93,101],[127,157],[174,193],[277,315],
            [320,353],[414,445],[500,698],[700,787],[852,902],[986,1035],
            [1086, 1450]]

peak_lst = pf.auto_scan(search, lst=ranges_m, plot=False, save_to_hdf=False)