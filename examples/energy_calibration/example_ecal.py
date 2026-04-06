"""
energy calibration example
"""

import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import energy_calibration as ecal
from nasagamma import file_reader
import matplotlib.pyplot as plt
import numpy as np

file = "../data/gui_test_data_cebr.csv"
spect = file_reader.read_csv(file)
fwhm_at_0 = 1.0
ref_x = 1315
ref_fwhm = 42
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=4)

xrange = [1250, 1600]
bkg = "poly2"
fit = pf.PeakFit(search, xrange=xrange, bkg=bkg)
fit.plot()

# energy calibration
ch = spect.channels
peak_info = fit.peak_info
mean_values = [peak_info[0]["mean"], peak_info[1]["mean"]]
mean_values.insert(0, 0)  # add the origin
erg = [0, 1173.2, 1332.5]  # in keV

cal = ecal.EnergyCalibration(mean_vals=mean_values, erg=erg, channels=ch,
                             n=1, e_units="keV")

cal.plot()
predicted_energies = cal.predicted

spect_cal = sp.Spectrum(counts=spect.counts, energies=predicted_energies, e_units="keV")
spect_cal.plot()
plt.title("Calibrated spectrum")

