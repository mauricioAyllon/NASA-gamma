# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:35:09 2020

@author: mauricio
First example using the Spectrum class and the PeakSearch class
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spectrum import Spectrum
import peaksearch as ps


filePath = "../BECA-simulations/MCNP-BECA/pulsing-scheme/F1/hdf-files/"
file_prompt = "spect_prompt_dry.hdf"
df_prompt = pd.read_hdf(filePath + file_prompt)

# take any spectrum
df0 = df_prompt.iloc[:,0]
cts_np = df0.to_numpy()
erg = df0.index.to_numpy()

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 35
ref_x = 1220

# instantiate a Spectrum object
spect = Spectrum(counts=cts_np)

# spectrum decomposition
ps_obj = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=2)


## plot 
# plot spectrum decomposition
plt.figure()
plt.plot(spect.counts, label='Raw spectrum')
plt.plot(ps_obj.peak_plus_bkg.clip(1e-1), label='Peaks+Continuum')
plt.plot(ps_obj.bkg.clip(1e-1), label='Continuum')
plt.plot(ps_obj.signal.clip(1e-1), label='Peaks')
plt.yscale('log')
#plt.xlim(0, len(spec))
plt.ylim(3e-1)
plt.xlabel('Channels')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()

## plot peak positions
plt.figure()
plt.plot(erg, ps_obj.snr, label="SNR all")
plt.plot(erg, spect.counts, label="Raw spectrum")
plt.yscale("log")
for xc in ps_obj.peaks_idx:
    x0 = erg[xc]
    plt.axvline(x=x0, color='red', linestyle='-', alpha=0.5)
    i = 1
plt.legend()
plt.title("SNR > 2")
plt.ylim(1e-1)
plt.ylabel("Cts/MeV/s")
plt.xlabel("Energy [MeV]")




