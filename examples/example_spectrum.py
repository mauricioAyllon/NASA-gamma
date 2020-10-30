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
file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key='data')

# delete first (large) bin
df = df.iloc[1:,:]

cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)
chan = np.arange(0,len(cts_np),1)

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np, energies=erg)

spect.plot()
