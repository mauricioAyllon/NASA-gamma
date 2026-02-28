"""
Minimal working examples for new Spectrum methods:
    - copy()
    - __add__, __sub__, __mul__, __rmul__, __truediv__, __radd__
"""
import numpy as np
from nasagamma import spectrum as sp
import matplotlib.pyplot as plt

# --- Synthetic data ---
channels = np.arange(0, 512)

# Two spectra with Gaussian peaks and Poisson noise
counts1 = (np.exp(-0.5 * ((channels - 200) / 10) ** 2) * 1000 +
           np.exp(-0.5 * ((channels - 350) / 15) ** 2) * 500 +
           np.random.poisson(10, size=len(channels))).astype(float)

counts2 = (np.exp(-0.5 * ((channels - 200) / 10) ** 2) * 800 +
           np.exp(-0.5 * ((channels - 400) / 15) ** 2) * 400 +
           np.random.poisson(10, size=len(channels))).astype(float)

spec1 = sp.Spectrum(counts=counts1, realtime=100.0, livetime=95.0, label="Spectrum 1")
spec2 = sp.Spectrum(counts=counts2, realtime=100.0, livetime=93.0, label="Spectrum 2")
fig0, ax0 = plt.subplots()
spec1.plot(ax=ax0)
spec2.plot(ax=ax0)

# --- copy() ---
spec1_copy = spec1.copy()
spec1_copy.smooth()  # modifies copy, spec1 is unchanged
spec1_copy.label = "Spectrum 1 smoothed"
print("copy():")
print(f"  spec1 max counts:      {spec1.counts.max():.2f}")
print(f"  spec1_copy max counts: {spec1_copy.counts.max():.2f}  (smoothed)")
fig1, ax1 = plt.subplots()
spec1.plot(ax=ax1)
spec1_copy.plot(ax=ax1)


# --- __add__ ---
spec_sum = spec1 + spec2
spec_sum.label = "Spectrum 1 + Spectrum 2"
print("\n__add__:")
print(f"  spec1 livetime:    {spec1.livetime}")
print(f"  spec2 livetime:    {spec2.livetime}")
print(f"  spec_sum livetime: {spec_sum.livetime}  (summed)")
print(f"  spec_sum counts[200]: {spec_sum.counts[200]:.2f}")
fig2, ax2 = plt.subplots()
spec1.plot(ax=ax2)
spec2.plot(ax=ax2)
spec_sum.plot(ax=ax2)

# --- __radd__ (enables sum() on a list of spectra) ---
spec_list = [spec1, spec2, spec1.copy()]
spec_total = sum(spec_list)
print("\n__radd__ via sum():")
print(f"  sum of 3 spectra counts[200]: {spec_total.counts[200]:.2f}")

# --- __sub__ ---
spec_diff = spec1 - spec2
spec_diff.label = "Spectrum 1 - Spectrum 2"
print("\n__sub__:")
print(f"  spec_diff counts[200]: {spec_diff.counts[200]:.2f}  (spec1 - spec2 at peak)")
fig3, ax3 = plt.subplots()
spec1.plot(ax=ax3)
spec2.plot(ax=ax3)
spec_diff.plot(ax=ax3, scale="linear")

# --- __mul__ (scalar) ---
spec_scaled = spec1 * 2.0
print("\n__mul__ (scalar):")
print(f"  spec1 counts[200]:        {spec1.counts[200]:.2f}")
print(f"  spec_scaled counts[200]:  {spec_scaled.counts[200]:.2f}  (x2)")

# --- __rmul__ (scalar on left) ---
spec_rscaled = 0.5 * spec1
print("\n__rmul__ (scalar on left):")
print(f"  spec_rscaled counts[200]: {spec_rscaled.counts[200]:.2f}  (x0.5)")

# --- __mul__ (numpy array — e.g. efficiency correction) ---
efficiency = np.linspace(0.8, 1.2, len(channels))  # mock efficiency curve
spec_corrected = spec1 / efficiency
spec_corrected.label = "Spectrum 1 * eff"
print("\n__truediv__ (efficiency correction array):")
print(f"  spec1 counts[200]:        {spec1.counts[200]:.2f}")
print(f"  efficiency at 200:        {efficiency[200]:.3f}")
print(f"  spec_corrected counts[200]: {spec_corrected.counts[200]:.2f}")
fig4, ax4 = plt.subplots()
spec1.plot(ax=ax4)
spec_corrected.plot(ax=ax4, scale="linear")

