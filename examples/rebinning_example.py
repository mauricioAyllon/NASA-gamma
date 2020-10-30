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

file1V1 = glob.glob('../../BECA-simulations/MCNP-BECA/detector-response/SSW-SSR/many/out/02-*.o')
file1V2 = glob.glob('../../BECA-simulations/MCNP-BECA/detector-response/SSW-SSR/many/out2/02-*.o')
file1 = file1V1 + file1V2
files1 = natsort.natsorted(file1)

## read file 1
## read file 3
cts_sum = 0
for f in files1:
    df1 = io.read_output(f, tally=8, n=1, tally_type='e', particle='p')
    cts_sum = cts_sum + df1['cts']   
df1['cts'] = cts_sum

df1.set_index("energy", inplace=True)

cts_np= df1.to_numpy()[:,0]


# instantiate a Spectrum object and rebin by 2
spect = sp.Spectrum(counts=cts_np, energies=df1.index)
ener2, cts2 = spect.rebin()

spect4 = sp.Spectrum(counts=cts2, energies=ener2)
ener4, cts4 = spect4.rebin()


plt.figure()
plt.plot(df1.index, df1.cts, label="Original")
plt.plot(ener2, cts2, label="Rebinned by 2")
plt.plot(ener4, cts4, label="Rebinned by 4")
plt.yscale("log")
plt.legend()
plt.ylabel("cts/s/MeV")
plt.xlabel("Energy [MeV]")


## smoothing

spect_mv = sp.Spectrum(counts=cts_np, energies=df1.index)
cts_mv = spect_mv.smooth(num=5)

plt.figure()
plt.plot(df1.index, df1.cts, label="Original")
plt.plot(df1.index, cts_mv, label="Smoothing by 5")
plt.yscale("log")
plt.legend()
plt.ylabel("cts/s/MeV")
plt.xlabel("Energy [MeV]")
