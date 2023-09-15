"""
Example file for 'Diagnostics' tab
"""
from nasagamma import diagnostics
from nasagamma import file_reader
import matplotlib.pyplot as plt

folder = "data/test_folder_diag"

spe = file_reader.ReadSPE(folder + "/RUN000.Spe")
diag = diagnostics.Diagnostics(folder)
diag.calculate_integral(xmid=2478, width=70)

plt.figure()
plt.plot(diag.absolute_time, diag.areas, "o-", label="Peak integrals")

diag.fit_peaks(xmid=2478, width=70)
plt.plot(diag.absolute_time, diag.areas, "o-", label="Peak fits")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Counts")
