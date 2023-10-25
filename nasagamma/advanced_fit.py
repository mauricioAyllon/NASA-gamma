"""
Advanced fitting functions
"""
import matplotlib.pyplot as plt
import numpy as np
from nasagamma import file_reader
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf


spe1 = file_reader.read_csv_file("../examples/data/gui_test_data_cebr.csv")
xrange = [1200, 1400]
x = spe1.channels[xrange[0] : xrange[1]]
y = spe1.counts[xrange[0] : xrange[1]]
