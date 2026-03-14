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

file_cebr = "data/gui_test_data_cebr.csv"
file_hpge = "data/gui_test_data_hpge_NH3.txt"

spe_cebr = file_reader.read_csv_file(file_cebr)
fwhm_at_0 = 1.0
ref_x = 420
ref_fwhm = 20
search_cebr = ps.PeakSearch(spe_cebr, ref_x, ref_fwhm, fwhm_at_0, min_snr=4)

# list of possible energies
pos_ergs = [661.7, 1173.2, 1332.5] # common lab sources
peak_chan = search_cebr.peaks_idx

efit = ecal.smart_calibration(channels=peak_chan, set_ergs=pos_ergs)

c0 = efit.best_values["c0"]
c1 = efit.best_values["c1"]

energy = c0 + spe_cebr.channels*c1

plt.figure()
plt.plot(energy, spe_cebr.counts)