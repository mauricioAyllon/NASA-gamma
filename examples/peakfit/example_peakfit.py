"""
Example usage of the peakfit class
"""
from nasagamma import spectrum as sp
import numpy as np
import pandas as pd
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader


# dataset 1
file = "../data/gui_test_data_cebr.csv"

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 31
ref_x = 1220
min_snr = 5

# instantiate a Spectrum object
spect = file_reader.read_csv(file)

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
search.plot_peaks()

# peakfit class
bkg0 = "poly1"
xrange = [1250, 1600]
fit = pf.PeakFit(search, xrange, bkg=bkg0)
print(fit.peak_info)
res = fit.fit_result

# print(res.fit_report())

# search.plot_peaks()

fit.plot()


## check area under the curve
def gaussian(x, A, mu, sig):
    return A / np.sqrt(2 * np.pi * sig**2) * np.exp(-((x - mu) ** 2) / (2 * sig**2))


xg = fit.x_data
Ag = res.best_values["g1_amplitude"]  # should also be the area under the curve!
mug = res.best_values["g1_center"]
sigg = res.best_values["g1_sigma"]

gauss = gaussian(xg, Ag, mug, sigg)

print(f"Area using peakfit = {Ag}")
print(f"Area using independent Gaussian = {gauss.sum()}")

