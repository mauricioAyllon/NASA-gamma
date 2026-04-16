"""
Example of how Poisson errors are treated
"""
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset 1
file1 = "../data/test_data_Fe_API.txt"
file2 = "../data/test_data_bkg_API.txt"
spect1 = file_reader.read_txt(file1)
spect2 = file_reader.read_txt(file2)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
spect1.plot(ax=ax)
spect2.plot(ax=ax)

search1 = ps.PeakSearch(spect1, ref_x=420, ref_fwhm=20, min_snr=50)
fit1 = pf.PeakFit(search1, xrange=[800,900])
err1 = fit1.peak_err[0]["area_err"]
print(f"Error in spectrum 1 = {err1}")

search2 = ps.PeakSearch(spect2, ref_x=420, ref_fwhm=20, min_snr=15)
fit2 = pf.PeakFit(search2, xrange=[800,900])
err2 = fit2.peak_err[0]["area_err"]
print(f"Error in spectrum 2 = {err2}")

spect3 = spect1 - spect2
spect3.replace_neg_vals()
search3 = ps.PeakSearch(spect3, ref_x=420, ref_fwhm=20, min_snr=20)
fit3 = pf.PeakFit(search3, xrange=[800,900])
err3 = fit3.peak_err[0]["area_err"]
print(f"Error in spectrum 3 = {err3}")


# check
err3_v2 = np.sqrt(err1**2 + err2**2)
print(f"Check error in spectrum 3 = {err3_v2}")


