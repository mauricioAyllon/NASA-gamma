"""
Example usage of smart calibration
"""
import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader
from nasagamma import energy_calibration as ecal
import matplotlib.pyplot as plt
import numpy as np

file_hpge = "data/gui_test_data_hpge_NH3.txt"

## HPGe
spe_hpge = file_reader.read_txt(file_hpge)
energies_orig = spe_hpge.energies
spe_hpge.remove_calibration()
spe_hpge.plot()
plt.title("Spectrum before calibration")
fwhm_at_0 = 1.0
ref_x = 420
ref_fwhm = 3
search_hpge = ps.PeakSearch(spe_hpge, ref_x, ref_fwhm, fwhm_at_0, min_snr=100,
                            xrange=[2500,9000])

pos_ergs = [2223.248, 3853.51, 6129.89-2*511, 6129.89-511, 6129.89]
peak_chan = search_hpge.peaks_idx

result = ecal.smart_calibration(channels=peak_chan, energies=pos_ergs) 
print(result)

ch = spe_hpge.channels
energies_calc = result["c1"]*ch + result["c0"]

vals = result["channels"]
ergs = result["energies"]

cal = ecal.EnergyCalibration(mean_vals=vals, erg=ergs,
                             channels=spe_hpge.channels, n=1, e_units="keV")
cal.plot()

plt.figure()
plt.plot(energies_orig, energies_calc, "--")
plt.xlabel("Original energy (keV)")
plt.ylabel("Calculated energy (keV)")

spe_hpge2 = sp.Spectrum(counts=spe_hpge.counts, energies=energies_calc, e_units="keV")
spe_hpge2.plot()
plt.title("Spectrum after calibration")




