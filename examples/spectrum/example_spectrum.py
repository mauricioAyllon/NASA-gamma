"""
Minimum working example of the Spectrum class
Plus rebinning and smoothing features
"""
from nasagamma import spectrum as sp
from nasagamma import file_reader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset 1
file = "../data/gui_test_data_cebr_cal.csv"
df = pd.read_csv(file)

# Instantiate a Spectrum object
spect = sp.Spectrum(counts=df["counts"], energies=df["Energy [keV]"], e_units="keV")

# Print some useful info
print(spect)

# Print more details (spectrum metadata)
print(spect.metadata())

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

spect.label = "Original spectrum"
spect.plot(ax=ax)

# Rebin spectrum by 2
spect.label = "Rebinned by 2"
spect.rebin(by=2)
spect.plot(ax=ax)

# Smooth spectrum by taking every 6 adjacent points
spect.label = "Smoothed"
spect.smooth(num=6)
spect.plot(ax=ax)
