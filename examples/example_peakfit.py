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
file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key='data')
df = df.iloc[1:,:]


cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 31
ref_x = 1220
min_snr = 1

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np, energies=erg)

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)

# peakfit class
bkg0 = 'poly1'
xrange = [1.46, 1.86]
#xrange = [2.1, 2.3]
xrange = [2.6, 4]
#xrange = [5, 6.3]
#xrange = [5, 7.5]
#xrange = [1.65, 4]
xrange = [6.75, 7.25]
fit = pf.PeakFit(search, xrange, bkg=bkg0)

res = fit.fit_result

#print(res.fit_report())

# search.plot_peaks()

fit.plot(plot_type="full", legend='on')

