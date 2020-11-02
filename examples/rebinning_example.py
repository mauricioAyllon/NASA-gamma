# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:16:46 2020

@author: mauricio
Rebinning and smoothing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
import natsort
import glob
import mcnptools.mcnpio as io


file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key='data')

# delete first (large) bin
df = df.iloc[1:,:]

cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)
chan = np.arange(0,len(cts_np),1)

# instantiate a Spectrum object and rebin by 2
spect = sp.Spectrum(counts=cts_np, energies=erg)
ener2, cts2 = spect.rebin()

spect4 = sp.Spectrum(counts=cts2, energies=ener2)
ener4, cts4 = spect4.rebin()


plt.figure()
plt.plot(erg, cts_np, label="Original")
plt.plot(ener2, cts2, label="Rebinned by 2")
plt.plot(ener4, cts4, label="Rebinned by 4")
plt.yscale("log")
plt.legend()
plt.ylabel("cts/s/MeV")
plt.xlabel("Energy [MeV]")


## smoothing

spect_mv = sp.Spectrum(counts=cts_np, energies=erg)
cts_mv = spect_mv.smooth(num=8)

plt.figure()
plt.plot(erg, cts_np, label="Original")
plt.plot(erg, cts_mv, label="Smoothing by 5")
plt.yscale("log")
plt.legend()
plt.ylabel("cts/s/MeV")
plt.xlabel("Energy [MeV]")
