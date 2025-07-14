"""
Gaussian energy broadening example 
"""
from nasagamma import file_reader
import matplotlib.pyplot as plt
import numpy as np

data = "data/gui_test_data_hpge_NH3.txt"
spe = file_reader.read_txt(data)
fig = plt.figure()
ax = fig.add_subplot()
spe.plot(ax=ax)
spe.gaussian_energy_broadening(spe.fwhm_LaBr_example)
spe.label = "NH3 - LaBr"
spe.plot(ax=ax)

# Or define your own FWHM function
def fwhm_hpge(E):
    return 0.1*np.sqrt(E) + 0.001*E

data = "data/gui_test_data_hpge_NH3.txt"
spe = file_reader.read_txt(data)
fig = plt.figure()
ax = fig.add_subplot()
spe.plot(ax=ax)
spe.gaussian_energy_broadening(fwhm_hpge)
spe.label = "NH3 - HPGe with slightly worse resolution"
spe.plot(ax=ax)