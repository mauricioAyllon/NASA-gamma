"""
Data retrieval tools for DAWN.
Dawn Data Archive:
https://arcnav.psi.edu/urn:nasa:pds:context:instrument:dawn.grand
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
    

file = "DAWN_data/GRD-L1B-150316-150423_160121-CSA-BGOC.xml"
all_data = get_data(file)
keys = list(all_data.keys())

bgo_spe = all_data[keys[3]].sum(axis=0)
plt.figure()
plt.plot(bgo_spe, drawstyle="steps", label=keys[3])
plt.legend()
plt.yscale("log")