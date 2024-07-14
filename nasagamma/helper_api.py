"""
Helper functions
"""
import numpy as np
import dateparser
from pathlib import Path
import pkg_resources
from nasagamma.read_parquet_api import get_data_path
from nasagamma import read_parquet_api
from nasagamma import apipandas as api
import os


def find_data_path(date, runnr):
    RUNNR = runnr
    DATE = dateparser.parse(date)

    path_data = get_data_path()
    DATA_DIR = path_data / f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{RUNNR:05d}"
    file_path = DATA_DIR / fname
    return file_path


# Read .npy MCA data, join if more than 1 file
def read_mca(date, runnr):
    # only channels 4 (LaBr==True) and 5 (LaBr==False)
    file_path = find_data_path(date, runnr)
    path_data = get_data_path()
    # load data
    files = list(file_path.glob("MCA-data/*.npy"))
    if len(files) > 1:
        data = 0
        for f in files:
            data0 = np.load(f)
            data = data + data0
    else:
        data = np.load(files[0])

    return data


def read_time_from_settings(settings_file, ch):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    for i, l in enumerate(lines):
        tmp = l.replace('"', "").split()
        if "live_time:" in tmp:
            idx = i
    time = float(lines[idx + ch + 1].split(",")[0])
    return time


def read_input_CR_from_settings(settings_file, ch=9):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    for i, l in enumerate(lines):
        tmp = l.replace('"', "").split()
        if "input_count_rate:" in tmp:
            idx = i
    CR = float(lines[idx + ch + 1].split(",")[0])
    return CR


def read_input_counts_from_settings(settings_file, ch=9):
    with open(settings_file, mode="r") as myfile:
        lines = myfile.readlines()

    for i, l in enumerate(lines):
        tmp = l.replace('"', "").split()
        if "input_counts:" in tmp:
            idx = i
    CR = float(lines[idx + ch + 1].split(",")[0])
    return CR


def get_total_time(date, runnr, ch, mca=False):
    file_path = find_data_path(date, runnr)
    # load data
    if mca:
        files = sorted(list(file_path.glob("MCA-data/*-stats-*")))[1:]
    else:
        files = sorted(list(file_path.glob("settings/*-stats-*")))[1:]
    t_tot = 0
    for f in files:
        t0 = read_time_from_settings(f, ch=ch)
        t_tot += t0
    return t_tot


def get_total_counts(date, runnr, ch):
    file_path = find_data_path(date, runnr)
    # load data
    files = list(file_path.glob("settings/*-stats-*"))[1:]
    cts_tot = 0
    for f in files:
        cts0 = read_input_counts_from_settings(f, ch=ch)
        cts_tot += cts0
    return cts_tot


def calculate_neutron_flux(date, runnr, ch, L=30):
    # L = neutron source to sample distance in cm
    alpha_counts = get_total_counts(date, runnr, ch)
    time_total = get_total_time(date, runnr, ch)
    alpha_cr = alpha_counts / time_total
    d = 6.7  # cm alpha detector-neutron source distance
    alpha_area = 4.8 * 4.8  # cm2
    phi_a = alpha_cr / alpha_area  # flux at alpha detector
    Y0 = 4 * np.pi * d**2 * phi_a  # neutron yield (n/s)
    phi_s = Y0 / (4 * np.pi * L**2)  # neutron flux on sample
    alpha_frac = 0.91 # correction factor for true alphas
    return Y0*alpha_frac, phi_s*alpha_frac

def approximate_fa(L=30, S=10):
    """
    Approximate fraction of alpha particles that intersect a square sample 
    of length S located at a distance L from the neutron source.

    Parameters
    ----------
    L : float, optional
        Distance in cm from the neutron source to the sample. The default is 30.
    S : float, optional
        Side lenght of square sample in cm. The default is 10.

    Returns
    -------
    fa : float
        fraction of counts in the alpha detector covered by the sample.

    """
    d = 6.7  # cm alpha detector-neutron source distance
    xa = 4.8 # cm alpha detector active area
    alpha_area = xa * xa  # cm2
    L_alpha = (d/L)*(S/2)*2
    sample_area = L_alpha * L_alpha
    fa = sample_area / alpha_area
    return fa

def create_directory(directory):
    """
    Ensure that the directory exists; if it does not, create it.
    
    Parameters:
    directory (str): Directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
        
def data_reduction(dates, run_numbers, new_date, new_run_number, ch):
    pass


def data_cleanup(runs_dict):
    # Initial filters
    xrange_labr = [-0.5, 0.5]
    yrange_labr = [-0.62, 0.655]
    trange_labr = [-20, 60]
    
    xrange_cebr = [-0.518, 0.526]
    yrange_cebr = [-0.685, 0.655]
    trange_cebr = [-20,60]
    
    date = runs_dict["date"]
    runnr = runs_dict["run"]
    ch = runs_dict["channel"]
    dfs = read_parquet_api.read_parquet_file(date=date, runnr=runnr, ch=ch)
    
    dfs["dt"] = dfs["dt"] + runs_dict["dt"]
    if ch == 5:
        dfs["energy_orig"] = dfs["energy_orig"] + 5435.24 - runs_dict["erg846"]
        dfxy = api.dftxy(df=dfs, xrange=xrange_labr, yrange=yrange_labr,
                         trange=trange_labr, xkey="X2", ykey="Y2", tkey="dt")
    elif ch == 4:
        dfs["energy_orig"] = dfs["energy_orig"] + 6565.2 - runs_dict["erg846"]
        dfxy = api.dftxy(df=dfs, xrange=xrange_cebr, yrange=yrange_cebr,
                         trange=trange_cebr, xkey="X2", ykey="Y2", tkey="dt")
        
    df_final = dfxy[["dt", "energy", "energy_orig", "LaBr[y/n]", "X2", "Y2"]]
    df_final["dt"] *= 1e-9
    return df_final





