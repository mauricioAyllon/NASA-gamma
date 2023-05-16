# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:11:59 2020

@author: mauricio
Example Spectrum class
"""
from nasagamma import spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset 1
file = "data/gui_test_data_cebr_calibrated.csv"
df = pd.read_csv(file)


# instantiate a Spectrum object
spect = sp.Spectrum(counts=df["counts"], energies=df["Energy [keV]"])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
spect.plot(fig=fig, ax=ax)

spect.rebin(by=2)
spect.plot(fig=fig, ax=ax)

spect.smooth(num=6)
spect.plot(fig=fig, ax=ax)
