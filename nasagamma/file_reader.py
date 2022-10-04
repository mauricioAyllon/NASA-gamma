"""
Classes and functions to read different file types
"""
import numpy as np
import pandas as pd
import re
from nasagamma import spectrum as sp
from nasagamma import read_cnf
import datetime


def read_csv_file(file_name):
    """
    Read .csv file.
    Must have at least one header with one of the key words listed
    in name_lst.

    Parameters
    ----------
    file_name : string.
        file path.

    Returns
    -------
    e_units : string
        X-axis units e.g. channels, keV, MeV.
    spect : Spectrum instance.
        Spectrum object from nasagamma.

    """
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.replace(" ", "")  # remove white spaces
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
        # print("working with channel numbers")
        e_units = "channels"
        spect = sp.Spectrum(counts=df[cts_col], e_units=e_units)
        spect.x = spect.channels
    elif erg != 0:
        # print("working with energy values")
        e_units = unit
        spect = sp.Spectrum(counts=df[cts_col], energies=df[erg], e_units=e_units)
        spect.x = spect.energies

    return e_units, spect


def read_cnf_to_spect(filename):
    """
    Read CNF file.

    Parameters
    ----------
    filename : string.
        file path.

    Returns
    -------
    e_units : string
        X-axis units e.g. channels, keV, MeV.
    spect : Spectrum instance.
        Spectrum object from nasagamma.

    """
    dict_cnf = read_cnf.read_cnf_file(filename, False)

    counts = dict_cnf["Channels data"]
    energy = dict_cnf["Energy"]
    livetime = dict_cnf["Live time"]

    if energy is None:
        e_units = "channels"
        spect = sp.Spectrum(counts=counts, e_units=e_units, livetime=livetime)
        spect.x = spect.channels
    else:
        e_units = dict_cnf["Energy unit"]
        spect = sp.Spectrum(
            counts=counts, energies=energy, e_units=e_units, livetime=livetime
        )
        spect.x = spect.energies

    return e_units, spect


class ReadMCA:
    def __init__(self, file):
        """
        Read .mca file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.tag = None
        self.description = None
        self.gain = None
        self.threshold = None
        self.live_mode = None
        self.preset_time = None
        self.live_time = None
        self.real_time = None
        self.start_time = None
        self.serial_no = None
        self.counts = None
        if file[-3:].lower() != "mca":
            print("ERROR: Must be an mca file")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()

        for i, line in enumerate(filelst):
            l = line.lower().split()
            if l[0] == "tag":
                self.tag = l[-1]
            elif l[0] == "description":
                self.description = l[-1]
            elif l[0] == "gain":
                try:
                    self.gain = float(l[-1])
                except:
                    pass
            elif l[0] == "threshold":
                try:
                    self.threshold = float(l[-1])
                except:
                    pass
            elif l[0] == "live_mode":
                try:
                    self.live_mode = float(l[-1])
                except:
                    pass
            elif l[0] == "preset_time":
                try:
                    self.preset_time = float(l[-1])
                except:
                    pass
            elif l[0] == "live_time":
                try:
                    self.live_time = float(l[-1])
                except:
                    pass
            elif l[0] == "real_time":
                try:
                    self.real_time = float(l[-1])
                except:
                    pass
            elif l[0] == "start_time":
                self.start_time = l[-2] + " " + l[-1]
            elif l[0] == "serial_number":
                self.serial_no = l[-1]
            elif l[0] == "<<data>>":
                start_idx = i
                break
        self.counts = np.array(filelst[start_idx + 1 : -1], dtype=int)


def read_lynx_csv(file_name):
    with open(file_name, "r") as myfile:
        filelst = myfile.readlines()

    for i, line in enumerate(filelst):
        l = line.lower().split()
        if "channel," in l and "counts" in l:
            istart = i
            break
    df = pd.read_csv(file_name, skiprows=istart, dtype=float)
    df.columns = df.columns.str.replace(" ", "")  # remove white spaces
    df.columns = df.columns.str.lower()
    ###
    cols = ["channel", "energy(kev)", "counts"]  # as listed on lynx
    # print("working with energy values")
    e_units = "keV"
    spect = sp.Spectrum(counts=df[cols[2]], energies=df[cols[1]], e_units=e_units)
    spect.x = spect.energies

    return e_units, spect


class ReadLynxCsv:
    def __init__(self, file):
        """
        Read -lynx.csv file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.start_time = None
        self.energy_calibration = None
        self.live_time = None
        self.real_time = None
        self.elapsed_computational = None
        self.eunits = None
        self.counts = None
        self.count_rate = None
        self.energy = None
        self.spect = None
        self.nch = None
        if file[-9:].lower() != "-lynx.csv":
            print("ERROR: Must be a -lynx.csv file")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            l = line.lower().split()
            if "start" in l and "time," in l:
                self.start_time = " ".join(l[2:])
            if "energy" in l and "calibration," in l:
                self.energy_calibration = " ".join(l[2:])
            if "live" in l and "time" in l:
                self.live_time = l[-1] + l[-2]
            if "real" in l and "time" in l:
                self.real_time = l[-1] + l[-2]
            if "elapsed" in l and "computational," in l:
                self.elapsed_computational = l[-1]
            if "channel," in l and "counts" in l:
                istart = i
                break
        df = pd.read_csv(self.file, skiprows=istart, dtype=float)
        df.columns = df.columns.str.replace(" ", "")  # remove white spaces
        df.columns = df.columns.str.lower()
        cols = ["channel", "energy(kev)", "counts"]  # as listed on lynx
        e_units = "keV"
        self.spect = sp.Spectrum(
            counts=df[cols[2]],
            energies=df[cols[1]],
            e_units=e_units,
            livetime=float(self.live_time[:-4]),
        )
        self.spect.x = self.spect.energies
        self.counts = self.spect.counts.sum()
        self.count_rate = self.counts / float(self.live_time[0:-4])
        self.nch = self.spect.counts.shape[0]


class ReadMultiscanPHA:
    def __init__(self, file):
        self.file = file
        self.start_time = None
        self.energy_calibration = None
        self.live_time = None
        self.real_time = None
        self.eunits = None
        self.counts = None
        self.count_rate = None
        self.spect = None
        self.nch = None
        if file[-8:].lower() != ".pha.txt":
            print("ERROR: Must be a .pha.txt file")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            l = line.lower().strip().split(",")
            if "time started" in l:
                self.start_time = ",".join(l[1:]).strip('"')
            if "live time when finished" in l:
                tme = datetime.datetime.strptime(l[1], "%H:%M:%S.%f")
                self.live_time = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
            if "real time when finished" in l:
                tme = datetime.datetime.strptime(l[1], "%H:%M:%S.%f")
                self.real_time = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
            if "energy equation" in l:
                self.energy_calibration = l[1]
                self.eunits = l[1].split("+")[0][-3:]
            if "total counts" in l:
                self.counts = float(l[1])
            if ["channel", "energy", "counts"] == l:
                istart = i
                cols = l
                break
        df = pd.read_csv(self.file, skiprows=istart, dtype=float)
        df.columns = cols
        self.spect = sp.Spectrum(
            counts=df["counts"],
            energies=df["energy"],
            e_units=self.eunits,
            livetime=self.live_time,
        )
        self.spect.x = self.spect.energies
        self.counts = self.spect.counts.sum()
        self.count_rate = self.counts / self.live_time
        self.nch = self.spect.counts.shape[0]
