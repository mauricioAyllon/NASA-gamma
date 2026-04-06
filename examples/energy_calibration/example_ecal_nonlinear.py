"""
energy calibration example with non-linear spectrum
"""

import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import energy_calibration as ecal
from nasagamma import file_reader
import matplotlib.pyplot as plt
import numpy as np

file = "../data/gui_test_data_nonlinear.csv"
df = pd.read_csv(file)
spect = file_reader.read_csv(file)
fwhm_at_0 = 1.0
ref_x = 420
ref_fwhm = 20
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=10)

mean_vals = [347, 506, 727, 1684, 1883, 2013, 2136]
erg = [846.78, 1238.33, 1779.03, 4439, 5107.89, 5618.89, 6129.89]
pcal = ecal.PiecewiseLinearCalibration(
    mean_vals=mean_vals,
    erg=erg,
    channels=spect.channels,
    e_break=4000,
    e_units="keV",
)

meta = pcal.metadata()
print("\n--- Calibration metadata ---")
for k, v in meta.items():
    if k not in ("mean_vals", "erg"):
        print(f"  {k}: {v}")

energies = pcal.predicted    

plt.figure()
plt.plot(energies, spect.counts)

## Smart calibration
energies_orig = df.channels
spect.plot()
plt.title("Spectrum before calibration")
search2 = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=10,
                        xrange=[250,4000])

peak_chan = search2.peaks_idx
pos_ergs = [846.78, 1238.33, 1779.03, 5107.89, 5618.89, 6129.89]

result = ecal.smart_calibration(channels=peak_chan, energies=pos_ergs, n=2,
                                max_combinations=300000) 
print(result)

ch = spect.channels
energies_calc = result["c2"]*ch**2 + result["c1"]*ch + result["c0"]

vals = result["channels"]
ergs = result["energies"]

cal = ecal.EnergyCalibration(mean_vals=vals, erg=ergs,
                             channels=spect.channels, n=2, e_units="keV")
cal.plot()


spect2 = sp.Spectrum(counts=spect.counts, energies=energies_calc, e_units="keV")
spect2.plot()
plt.title("Spectrum after calibration")



# xrange = [1250, 1600]
# bkg = "poly2"
# fit = pf.PeakFit(search, xrange=xrange, bkg=bkg)
# fit.plot()

# # energy calibration
# ch = spect.channels
# peak_info = fit.peak_info
# mean_values = [peak_info[0]["mean"], peak_info[1]["mean"]]
# mean_values.insert(0, 0)  # add the origin
# erg = [0, 1173.2, 1332.5]  # in keV

# cal = ecal.EnergyCalibration(mean_vals=mean_values, erg=erg, channels=ch,
#                              n=1, e_units="keV")

# cal.plot()
# predicted_energies = cal.predicted

# spect_cal = sp.Spectrum(counts=spect.counts, energies=predicted_energies, e_units="keV")
# spect_cal.plot()
# plt.title("Calibrated spectrum")

