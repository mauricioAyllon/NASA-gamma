# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:16:49 2021

@author: mauricio
Example Spectrum smoothing/rebinning
"""

from nasagamma import spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset 1
file = "data/gui_test_data_labr_uncalibrated.csv"
df = pd.read_csv(file)

cts_np = df.counts.to_numpy()

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
spect.plot(fig=fig, ax=ax)

# test smoothing by rebinning
for i in range(2):
    spect.rebin()
spect.plot(fig=fig, ax=ax)

# # test smoothing by moving average
# spect.smooth(num=12)
# spect.plot(fig=fig, ax=ax)
