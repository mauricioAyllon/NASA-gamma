"""
Add peaks to a peaksearch object with energies and not channels
"""
from nasagamma import spectrum as sp
import pandas as pd
import numpy as np
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf

# dataset 1
file = "data/gui_test_data_labr.csv"
df = pd.read_csv(file)

# instantiate a Spectrum object
spect = sp.Spectrum(counts=df["counts"], energies=df["energy_MeV"])

# Required input parameters (in channels)
fwhm_at_0 = 1
ref_fwhm = 20
ref_x = 420
min_snr = 5

# instantiate a peaksearch object
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
search.plot_peaks()

# fit
fit = pf.PeakFit(search=search, xrange=[3.5, 4])
fit.plot(plot_type="full", legend="on", table_scale=[2, 2.3])


# # assume we want to add a peak at x=3.86 MeV
xnew_e = 3.86  # MeV
xnew = np.where(spect.energies >= xnew_e)[0][0]
x_idx = np.searchsorted(search.peaks_idx, xnew)
search.peaks_idx = np.insert(search.peaks_idx, x_idx, xnew)
fwhm_guess_new = search.fwhm(xnew)
search.fwhm_guess = np.insert(search.fwhm_guess, x_idx, fwhm_guess_new)

search.plot_peaks()

# fit
fit = pf.PeakFit(search=search, xrange=[3.5, 4])
fit.plot(plot_type="full", legend="on", table_scale=[2, 2.3])
