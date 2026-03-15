"""
Example to read a .pha.txt file, and then create and read .txt files with metadata
"""
import numpy as np
import matplotlib.pyplot as plt
from nasagamma import spectrum as sp
from nasagamma import file_reader

file = "data/gui_test_data_hpge_NH3.pha.txt"

# Read multiscan file
spect = file_reader.read_multiscan(file)
spect.plot()



# # Change label and save as a .txt file with metadata
# spe_ms.label = "This is a better label"
# spe_ms.to_txt("test-txt-file.txt")

# # Read the newly created test file
# spe_txt = file_reader.read_txt("test-txt-file.txt")
# spe_txt.plot()
