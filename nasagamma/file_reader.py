"""
Classes and functions to read different file types
"""

import numpy as np
import pandas as pd
import re
from nasagamma import spectrum as sp
from nasagamma import read_cnf
import datetime


def process_df(df):
    """
    Process dataframe.
    Must have at least one header with one of the key words listed
    in name_lst.

    Parameters
    ----------
    df : pandas dataframe.
        dataframe containing counts or counts and energy.

    Returns
    -------
    unit : string.
        X-axis units e.g. channels, keV, MeV.
    cts_col : string.
        name of column header for counts.
    erg : string.
        name of column header for energies.
    """
    # remove white spaces and convert to lower case
    df.columns = df.columns.str.replace(" ", "").str.lower()
    ###
    name_lst = ["count", "counts", "cts", "data", "countrate(cps)"]
    e_lst = ["energy", "energies", "erg"]
    unit_dict = {"ev": "eV", "kev": "keV", "mev": "MeV", "gev": "GeV"}
    col_lst = list(df.columns)
    # cts_col = [s for s in col_lst if "counts" in s.lower()][0]
    cts_col = 0
    erg = 0
    unit = "keV_default"  # if no units given, default to keV
    for s in col_lst:
        s2 = re.split("[^a-zA-Z]", s)  # split by non alphabetic character
        s2 = [x for x in s2 if x]  # remove empty string
        if s in name_lst:
            cts_col = s
            next
        for st in s2:
            if st in e_lst:
                erg = s
            if st in list(unit_dict.keys()):
                unit = unit_dict[st]
    return unit, cts_col, erg


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

    unit, cts_col, erg = process_df(df)

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

    return spect


def read_txt(filename):
    description = None
    plot_label = None
    date_created = None
    realtime = None
    livetime = None
    erg_cal = None
    with open(filename, "r") as myfile:
        filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            l = line.split()
            if l[0].lower() == "description:" and len(l) > 1:
                description = " ".join(l[1:])
            if l[0].lower() == "label:" and len(l) > 1:
                plot_label = " ".join(l[1:])
            if l[0].lower() == "date" and l[1].lower() == "created:" and len(l) > 2:
                date_created = " ".join(l[2:])
            if l[0].lower() == "real" and l[1].lower() == "time" and len(l) > 2:
                realtime = l[3]
            if l[0].lower() == "live" and l[1].lower() == "time" and len(l) > 2:
                livetime = l[3]
            if l[0].lower() == "energy" and l[1].lower() == "calibration:":
                if len(l) > 2:
                    erg_cal = " ".join(l[2:])
                start_idx = i + 1
                break
    df = pd.read_csv(filename, skiprows=start_idx)
    unit, cts_col, erg = process_df(df)

    if realtime == "None":
        realtime = None
    else:
        realtime = float(realtime)
    if livetime == "None":
        livetime = None
    else:
        livetime = float(livetime)
    if description == "None":
        description = None
    if plot_label == "None":
        plot_label = None
    if date_created == "None":
        date_created = None
    if erg_cal == "None":
        erg_cal = None

    if cts_col == 0:
        print("ERROR: no column named with counts keyword e.g counts, data, cts")
    elif erg == 0:
        e_units = "channels"
        spect = sp.Spectrum(
            counts=df[cts_col],
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            description=description,
            acq_date=date_created,
            energy_cal=erg_cal,
            label=plot_label,
        )
        spect.x = spect.channels
    elif erg != 0:
        # print("working with energy values")
        e_units = unit
        spect = sp.Spectrum(
            counts=df[cts_col],
            energies=df[erg],
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            description=description,
            acq_date=date_created,
            energy_cal=erg_cal,
            label=plot_label,
        )
    return spect


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
    dict_cnf = read_cnf.read_cnf_file(filename, write_output=False)

    counts = dict_cnf["Channels data"]
    energy = dict_cnf["Energy"]
    livetime = dict_cnf["Live time"]
    realtime = dict_cnf["Real time"]
    start_date_time = dict_cnf["Start time"]

    if energy is None:
        e_units = "channels"
        spect = sp.Spectrum(
            counts=counts,
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            acq_date=start_date_time,
        )
    else:
        erg_coeff = dict_cnf["Energy coefficients"]
        erg_eqn = f"{erg_coeff[0]} + {erg_coeff[1]}*ch + {erg_coeff[2]}*ch^2 + {erg_coeff[3]}*ch^3"
        e_units = dict_cnf["Energy unit"]
        spect = sp.Spectrum(
            counts=counts,
            energies=energy,
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            acq_date=start_date_time,
            energy_cal=erg_eqn,
        )

    return spect


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
        counts = np.array(filelst[start_idx + 1 : -1], dtype=int)
        self.spect = sp.Spectrum(
            counts=counts,
            livetime=self.live_time,
            realtime=self.real_time,
            acq_date=self.start_time,
            description=self.description,
            label=self.tag,
        )


class ReadSPE:
    def __init__(self, file):
        """
        Read .Spe file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.description = None
        self.detector = None
        self.detector_description = None
        self.version = None
        self.time_str = None
        self.time_s = None
        self.date = None
        self.start_time = None
        self.live_time = None
        self.real_time = None
        self.channels = None
        self.ROI = None
        self.counts = None
        self.erg_cal = None

        if file[-3:].lower() != "spe":
            print("ERROR: Must be a Spe file")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            l = line.lower().split()
            if l[0] == "$spec_id:":
                self.description = filelst[i + 1]
            if "det#" in l:
                self.detector = l[1]
            if "detdesc#" in l:
                self.detector_description = l[1:]
            if "ap#" in l:
                self.version = l[1:]
            if "$date_mea:" in l:
                self.date = filelst[i + 1].split()[0]
                self.time_str = filelst[i + 1].split()[1]
                tme = datetime.datetime.strptime(self.time_str, "%H:%M:%S")
                self.time_s = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
            if "$meas_tim:" in l:
                self.real_time = float(filelst[i + 1].split()[1])
                self.live_time = float(filelst[i + 1].split()[0])
            if "$data:" in l:
                self.channels = int(filelst[i + 1].split()[1])
                start_idx = i
            if "$roi:" in l:
                self.ROI = filelst[i + 1].split()[0]
                end_idx = i
            if "$ener_fit:" in l:
                self.erg_cal = filelst[i + 1].split()
        self.counts = np.array(filelst[start_idx + 2 : end_idx - 1], dtype=int)
        self.spect = sp.Spectrum(
            counts=self.counts,
            livetime=self.live_time,
            realtime=self.real_time,
            acq_date=self.date + " " + self.time_str,
            description=self.description,
            energy_cal=self.erg_cal,
        )


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


def read_multiscan(file):
    description = None
    start_time = None
    energy_calibration = None
    live_time = None
    real_time = None
    eunits = None
    tot_counts = None
    count_rate = None
    if file[-8:].lower() != ".pha.txt":
        print("ERROR: Must be a .pha.txt file")
    else:
        with open(file, "r") as myfile:
            filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            l = line.lower().strip().split(",")
            if "name" in l and len(l) > 1:
                description = l[1]
            if "time started" in l:
                start_time = ",".join(l[1:]).strip('"')
            if "live time when finished" in l:
                tme = datetime.datetime.strptime(l[1], "%H:%M:%S.%f")
                live_time = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
            if "real time when finished" in l:
                tme = datetime.datetime.strptime(l[1], "%H:%M:%S.%f")
                real_time = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
            if "energy equation" in l:
                energy_calibration = l[1]
                eunits = l[1].split("+")[0][-3:]
            if "total counts" in l:
                tot_counts = float(l[1])
            if ["channel", "energy", "counts"] == l:
                istart = i
                cols = l
                break
        df = pd.read_csv(file, skiprows=istart, dtype=float)
        df.columns = cols
        spect = sp.Spectrum(
            counts=df["counts"],
            energies=df["energy"],
            description=description,
            e_units=eunits,
            livetime=live_time,
            realtime=real_time,
            energy_cal=energy_calibration,
            acq_date=start_time,
        )
        return spect

class ReadMultiScanTlist:
    def __init__(self, file):
        """
        Read MultiScan .txt Tlist file.
        
        Parameters
        ----------
        file : sting
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.energy_flag = False # default
        self.df = None
        
    def read_file(self):
        if self.file[-3:] == "txt":
            with open(self.file, mode="r") as f:
                file_lst = f.readlines()
            split_data = []
            for line in file_lst:
                l = line.strip().split()
                split_data.append(l)
        try:
            cols = ["channel", "ts"]
            df = pd.DataFrame(columns=cols, data=split_data, dtype=np.float64)
            self.df = df
        except Exception as e:
            print("ERROR: Could not open file")
            print("An unknown error occurred:", str(e))
   

class ReadCaenListMode:
    def __init__(self, file):
        """
        Read CAEN .txt list mode data file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.header0 = None
        self.header1 = None
        self.header2 = None
        self.header3 = None
        self.header4 = None
        self.idx_start = None  # start of data after header
        self.df = None
        self.read_file()
        self.parse_header()

    def read_file(self):
        with open(self.file, "r") as myfile:
            self.filelst = [line.rstrip() for line in myfile]

    def parse_header(self):
        data = []
        for i, line in enumerate(self.filelst):
            l = line.lower().split(":")
            if l[0] == "header0":
                self.header0 = int(l[1])
            if l[0] == "header1":
                self.header1 = int(l[1])
            if l[0] == "header2":
                self.header2 = int(l[1])
            if l[0] == "header3":
                self.header3 = int(l[1])
            if l[0] == "header4":
                self.header4 = int(l[1])
                self.idx_start = i
                break

    def parse_data(self):
        data = []
        for i, line in enumerate(self.filelst[self.idx_start :]):
            l = line.split()
            data.append(l)
        data = np.array(data[1:], dtype=int)
        cols = ["ts (ns)", "channel", "flag"]
        self.df = pd.DataFrame(columns=cols, data=data)
