"""
Example usage of the chi-saquared optimizer
"""
from nasagamma import spectrum as sp
import numpy as np
import pandas as pd
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader


# dataset 1
file = "../data/gui_test_data_lab_sources.cnf"

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 20
ref_x = 420
min_snr = 5

# instantiate a Spectrum object
spect = file_reader.read_cnf(file)

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
search.plot_peaks()

# peakfit class
bkg0 = "poly1"
xrange = [573, 666]
fit = pf.PeakFit(search, xrange, bkg=bkg0)
res = fit.fit_result
print(f"Reduced chi-squared = {res.redchi}")
fit.plot()

# optimizer
best_range, best_redchi = fit.optimize_xrange(max_extend=5.0, n_steps=50, verbose=True)
fit.plot()
print(f"Reduced chi-squared after optimization= {best_redchi}")


