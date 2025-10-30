"""
Data retrieval tools for Lunar Prospector.
Lunar Prospector Gamma Ray Spectrometer and Neutron Spectrometer Reduced Data:
https://pds-geosciences.wustl.edu/missions/lunarp/reduced_grsns.html
PDS Viewer:
https://pdssbn.astro.umd.edu/tools/pds4_tools_info/pds4_viewer.html
"""

import pds4_tools as pds
import numpy as np
import matplotlib.pyplot as plt
import glob


def get_data(file):
    struct = pds.read(file, lazy_load=False)
    struct.info(abbreviated=False)
    iden = struct[0].id
    data = struct[iden].data
    names = data.dtype.names
    print("NAMES: ", names)
    data_dict = {}
    for n in names:
        data0 = data[n]
        data_dict[n] = data0
    return data_dict
    

file = "LP_data/1998_016_grs.xml"
all_data = get_data(file)
keys = list(all_data.keys())

acc_spe = all_data[keys[0]].sum(axis=0)
plt.figure()
plt.plot(acc_spe, drawstyle="steps", label=keys[0])
plt.legend()
plt.yscale("log")