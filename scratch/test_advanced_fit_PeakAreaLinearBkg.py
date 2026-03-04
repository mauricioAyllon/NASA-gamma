"""
Test advanced fitting PeakAreaLinearBkg class
"""
import matplotlib.pyplot as plt
from nasagamma import spectrum as sp
from nasagamma import advanced_fit as adv
from nasagamma import file_reader

f1 = "../examples/data/gui_test_data_cebr.csv"
f2 = "../examples/data/gui_test_data_cebr_cal.csv"

spe1 = file_reader.read_csv_file(f1)
spe2 = file_reader.read_csv_file(f2)

x1 = 1267 # channel no.
x2 = 1368 # channel no.

alb = adv.PeakAreaLinearBkg(spe1, x1, x2)