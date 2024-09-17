"""
Example use of tlist module.
The following example comes from an experiment with a pulsed DT neutron generator
and a CeBr detector. The pulse period is 1000 us
"""

from nasagamma import tlist

file = "Data/gui_test_data_tlist.txt"

png = tlist.Tlist(fname=file, period=1000)

png.plot_time_hist()
