"""
Initial parameter settings for GUI. It is used only when parameters are 
passed directly in the command line.
"""

import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import read_cnf
import re
from nasagamma import file_reader


def get_spect_search(commands):
    if commands["-o"]:
        return None
    file_name = commands["<file_name>"]
    # The detector types below are accurate only for the example files.
    # Add a similar command for your own detector or modify the values below.
    if commands["--cebr"]:
        fwhm_at_0 = 1.0
        ref_x = 1317
        ref_fwhm = 41  # 41
    elif commands["--labr"]:
        fwhm_at_0 = 1.0
        ref_x = 427
        ref_fwhm = 10
    elif commands["--hpge"]:
        fwhm_at_0 = 0.1
        ref_x = 948
        ref_fwhm = 4.4
    else:
        fwhm_at_0 = float(commands["--fwhm_at_0"])
        ref_x = float(commands["--ref_x"])
        ref_fwhm = float(commands["--ref_fwhm"])

    if commands["--min_snr"] is None:
        min_snr = 1.0
    else:
        min_snr = float(commands["--min_snr"])

    file_name = file_name.lower()
    if file_name[-4:] == ".csv":
        e_units, spect = file_reader.read_csv_file(file_name)
    elif file_name[-4:] == ".cnf":
        e_units, spect = file_reader.read_cnf.read_cnf_to_spect(file_name)
    # peaksearch class
    search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
    return spect, search, e_units, ref_x, fwhm_at_0, ref_fwhm
