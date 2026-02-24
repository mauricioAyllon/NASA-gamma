"""
Example Spectrum gain shift
"""

from nasagamma import spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset 1
file = "../examples/data/gui_test_data_cebr.csv"
df = pd.read_csv(file)

cts_np = df.counts.to_numpy()

# instantiate a Spectrum object
spect = sp.Spectrum(counts=cts_np)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
spect.plot(ax=ax)

# test gain shifting
spect.gain_shift(by=30)
spect.plot(ax=ax)
