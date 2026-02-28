"""
Example showing a simple test case to overlay several spectra
"""

import numpy as np
from nasagamma import spectrum as sp


# Create two synthetic spectra with a few Gaussian peaks
channels = np.arange(0, 1024)

counts1 = (np.exp(-0.5 * ((channels - 300) / 10) ** 2) * 1000 +
           np.exp(-0.5 * ((channels - 600) / 15) ** 2) * 500 +
           np.random.poisson(10, size=len(channels)))

counts2 = (np.exp(-0.5 * ((channels - 310) / 12) ** 2) * 800 +
           np.exp(-0.5 * ((channels - 600) / 15) ** 2) * 600 +
           np.random.poisson(10, size=len(channels)))

spec1 = sp.Spectrum(counts=counts1, label="Detector A")
spec2 = sp.Spectrum(counts=counts2, label="Detector B")

ax = sp.plot_overlay([spec1, spec2], scale="log", colors=["steelblue", "tomato"])

# Further customize using the returned ax
ax.set_title("Detector Comparison")