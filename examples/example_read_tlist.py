"""
Example tlist
"""

from nasagamma import tlist
from nasagamma import decay_exponential as decay
import numpy as np
import matplotlib.pyplot as plt
import lmfit


file = "data/gui_test_data_tlist.txt"
png = tlist.Tlist(fname=file, period=1000)
# plt.rcParams.update({'font.size': 22})
png.plot_time_hist()
png.hist_erg()
png.plot_spect_erg_all(x=png.x, y=png.spect)


# Die-away
png.filter_tdata(trange=[210, 1000])
png.hist_time()
png.plot_die_away()

# Exponential fitting
x_data = png.xd
y_data = png.yd
y_err = png.yerrd

exp = decay.Decay_exp(x=x_data, y=y_data, yerr=y_err)
exp.fit_single_decay()
exp.plot(show_components=True)
exp.fit_double_decay()
exp.plot(show_components=True)
print(exp.fit_result.fit_report())
