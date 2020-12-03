# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:33:41 2020

@author: mauricio
Read dataframe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nasagamma import peakfit as pf

file = "data/SSR-mcnp.hdf"
df = pd.read_hdf(file, key="data")
df = df.iloc[1:, :]

cts_np = df.cts.to_numpy() * 1e8
erg = np.array(df.index)
chan = np.arange(0, len(cts_np), 1)

dfp = pd.read_hdf("fitted_peaks_delete2.hdf", key="data")

# plot gaussian components
gc = pf.GaussianComponents(df_peak=dfp)
gc.plot_gauss(plot_type="full", table_scale=[1, 1.5])

# plt.figure()
# for i in dfp.index:
#     #plt.plot(dfp.loc[i,"x_data"], dfp.loc[i, "gauss"])
#     x = dfp.loc[i,"x_data"]
#     y = dfp.loc[i, "gauss"]
#     plt.fill_between(x, 0, y, alpha=0.5)

#     x0 = round(dfp.loc[i,"mean"], 2)
#     y0 = y.max()
#     a = round(dfp.loc[i,"area"], 2)
#     str0 = f"{x0} \n{a}"
#     plt.text(x0,y0, str0)

# plt.xlabel("Channels")

# # creating the bar plot
# plt.figure()
# for i in dfp.index:
#     x = dfp.loc[i,"mean"]
#     ch = np.where(chan>=x)[0][0]
#     erg0 = erg[ch]
#     h = dfp.loc[i,"area"]
#     plt.bar(erg0, h, color ='maroon',
#             width = 0.05)

#     str0 = f"{erg0}"
#     plt.text(erg0,h, str0)
# plt.xlabel("Channels")
# plt.ylabel("Area under curve")
