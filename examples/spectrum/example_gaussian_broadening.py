"""
Gaussian energy broadening example 
"""
from nasagamma import file_reader
import matplotlib.pyplot as plt
import numpy as np

data = "../data/gui_test_data_hpge_NH3.txt"

spe1 = file_reader.read_txt(data)
fig1 = plt.figure()
ax1 = fig1.add_subplot()
spe1.plot(ax=ax1)
spe1.gaussian_energy_broadening(spe1.fwhm_LaBr_example)
spe1.label = "Energy broadened to look like LaBr"
spe1.plot(ax=ax1)

# Or define your own FWHM function
def fwhm_hpge(E):
    return 0.1*np.sqrt(E) + 0.001*E

spe2 = file_reader.read_txt(data)
fig2 = plt.figure()
ax2 = fig2.add_subplot()
spe2.plot(ax=ax2)
spe2.gaussian_energy_broadening(fwhm_hpge)
spe2.label = "HPGe with slightly worse resolution"
spe2.plot(ax=ax2)