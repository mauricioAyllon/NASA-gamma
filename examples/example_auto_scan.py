"""
Example using auto_scan
"""
from nasagamma import spectrum as sp
import numpy as np
import pandas as pd
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader

# Required input parameters (in channels)
ref_fwhm = 3
ref_x = 420
min_snr = 15

# instantiate a Spectrum object
file = "Data/gui_test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, min_snr=min_snr, method="fast")

## plot peak positions
search.plot_peaks()

## auto_scan
ranges_m = [
    [8, 18],
    [32, 44],
    [72, 88],
    [93, 101],
    [127, 157],
    [174, 193],
    [277, 315],
    [320, 353],
    [414, 445],
    [500, 698],
    [700, 787],
    [852, 902],
    [986, 1035],
    [1086, 1450],
]

ranges_m = [0,16000]
peak_lst = pf.auto_scan(search, plot=False)
