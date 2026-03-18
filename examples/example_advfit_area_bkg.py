"""
Example of advanced fitting: peak area minus linear background
"""
from nasagamma import spectrum as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nasagamma import file_reader
from nasagamma import advanced_fit as af

# dataset 1
file = "data/gui_test_data_cebr_cal.csv"
spect = file_reader.read_csv(file)

# Print spectrum metadata
print(spect.metadata())

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

spect.label = "Original spectrum"
spect.plot(ax=ax)

# Fit the background only
fit = af.PeakAreaLinearBkg(spect, x1=1060, x2=1260)
fit.plot(areas=False)

e1 = [1060, 1105]
e2 = [1214, 1260]
fit.calculate_peak_area(x1=e1, x2=e2)
fit.plot(areas=True)


