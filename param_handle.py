# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:24:32 2021

@author: mauricio
parameter handle for GUI
"""

import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
import re


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

    e_units, spect, x = read_file(file_name)
    # peaksearch class
    search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
    return spect, search, e_units, x, ref_x, fwhm_at_0, ref_fwhm


def read_file(file_name):
    df = pd.read_csv(file_name)
    ###
    name_lst = ["count", "counts", "cts", "data"]
    e_lst = ["energy", "energies", "erg"]
    u_lst = ["eV", "keV", "MeV", "GeV"]
    col_lst = list(df.columns)
    # cts_col = [s for s in col_lst if "counts" in s.lower()][0]
    cts_col = 0
    erg = 0
    for s in col_lst:
        s2 = re.split("[^a-zA-Z]", s)  # split by non alphabetic character
        if s.lower() in name_lst:
            cts_col = s
            next
        for st in s2:
            if st.lower() in e_lst:
                erg = s
            if st in u_lst:
                unit = st
    if cts_col == 0:
        print("ERROR: no column named with counts keyword e.g counts, data, cts")
    elif erg == 0:
        print("working with channel numbers")
        e_units = "channels"
        spect = sp.Spectrum(counts=df[cts_col], e_units=e_units)
        x = spect.channels
    elif erg != 0:
        print("working with energy values")
        e_units = unit
        spect = sp.Spectrum(counts=df[cts_col], energies=df[erg], e_units=e_units)
        x = spect.energies

    return e_units, spect, x
