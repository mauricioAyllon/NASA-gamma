"""
Functions used to analyze API data

"""

import numpy as np
import matplotlib.pyplot as plt

# from mayavi import mlab
import pandas as pd
import os

# import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import dateparser
from nasagamma.read_parquet_api import get_data_path
from nasagamma import read_parquet_api


def test_limits(df, elog=True, Vmax=None, ekey="energy", tkey="dt", xkey="X", ykey="Y"):
    """Plots all data to define energy, time, and xy limits. df is a pandas dataframe"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    plt.figure()
    plt.hist(df[ekey], 256)
    plt.yscale("log" if elog else "linear")
    plt.title("Energy")

    res, xed, yed = np.histogram2d(df[ykey], df[xkey], bins=100)
    plt.figure()
    plt.imshow(
        res,
        extent=[df[xkey].min(), df[xkey].max(), df[ykey].min(), df[ykey].max()],
        cmap="inferno",
        interpolation="bilinear",
        vmax=Vmax,
        origin="lower",
    )
    plt.title("All events")

    plt.figure()
    plt.hist(df[tkey], 256)
    plt.xlabel("ns")
    plt.title("dt - all events")


def calc_own_pos(df, xkey_new="X2", ykey_new="Y2"):
    """Return a pandas data frame with two extra columns X2, Y2
    with calculated positions based on the equations below"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if xkey_new in df.columns and ykey_new in df.columns:
        return df
    else:
        tot = df["A"] + df["B"] + df["C"] + df["D"]
        X2 = (df["B"] + df["C"] - df["D"] - df["A"]) / tot
        Y2 = (df["B"] + df["A"] - df["D"] - df["C"]) / tot
        df2 = df.copy(deep=True)
        df2[xkey_new] = X2
        df2[ykey_new] = Y2
        df2.fillna(0, inplace=True)
        return df2


def dfe(df, erange, ekey="energy"):
    """Return a pandas data frame filtered in energy"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfenergy = df[(df[ekey] > erange[0]) & (df[ekey] < erange[1])]
    return dfenergy


def dfxy(df, xrange, yrange, xkey="X", ykey="X"):
    """Return a pandas data frame filtered in x-y"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    df_xy = df[
        (df[xkey] > xrange[0])
        & (df[xkey] < xrange[1])
        & (df[ykey] > yrange[0])
        & (df[ykey] < yrange[1])
    ]
    return df_xy


def dfxye(df, xrange, yrange, erange, xkey="X", ykey="Y", ekey="energy"):
    """Return a pandas data frame filtered in x-y and energy"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfen = dfe(df, erange, ekey)
    df_xy_e = dfen[
        (dfen[xkey] > xrange[0])
        & (dfen[xkey] < xrange[1])
        & (dfen[ykey] > yrange[0])
        & (dfen[ykey] < yrange[1])
    ]
    return df_xy_e


def dftxye(
    df, xrange, yrange, erange, trange, xkey="X", ykey="Y", ekey="energy", tkey="dt"
):
    """Return a pandas data frame filtered in x-y, energy, and time"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    dfxyen = dfxye(df, xrange, yrange, erange, xkey, ykey, ekey)
    dftxy_e = dfxyen[(dfxyen[tkey] > trange[0]) & (dfxyen[tkey] < trange[1])]

    return dftxy_e


def dftxy(df, xrange, yrange, trange, xkey="X", ykey="Y", tkey="dt"):
    """Return a pamdas data frame filtered in x-y, and time"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    df_xy = dfxy(df, xrange, yrange, xkey, ykey)
    dft_xy = df_xy[(df_xy[tkey] > trange[0]) & (df_xy[tkey] < trange[1])]

    return dft_xy


def dft(df, trange, tkey="dt"):
    """Return a pandas data frame filtered in time"""
    df_t = df[(df[tkey] > trange[0]) & (df[tkey] < trange[1])]
    return df_t


def dfet(df, erange, trange, ekey="energy", tkey="dt"):
    """Return a pandas data frame filtered in energy and time"""
    dfen = dfe(df, erange, ekey)
    df_e_t = dfen[(dfen[tkey] > trange[0]) & (dfen[tkey] < trange[1])]
    return df_e_t


def time_cut(df, t_start, t_stop, step, xkey="X", ykey="Y", tkey="dt"):
    """2D plots of time cuts in ns. Use a dataframe filtered in x-y and energy"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    tdiff = df[tkey]
    X = df[xkey]
    Y = df[ykey]
    for k in np.arange(t_start, t_stop, step):
        mask = (tdiff > k) * (tdiff < k + step)
        tdiff2 = tdiff[mask]
        X2 = X[mask]
        Y2 = Y[mask]
        # Z2 = [x for i,x in enumerate(Z) if i not in idx2]

        result, xedges, yedges = np.histogram2d(X2, Y2, 80)
        plt.figure()
        plt.imshow(result, interpolation="bilinear", cmap="inferno")
        plt.title(f"time cut {k,k+step} ns")
        plt.show()

def compute_fft(signal, sampling_rate):
    """
    Compute the FFT of a signal.

    Parameters:
        signal (numpy.ndarray): The input signal.
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        freqs (numpy.ndarray): Frequencies corresponding to FFT components.
        fft_magnitude (numpy.ndarray): Magnitude of the FFT.
    """
    n = len(signal)
    fft_values = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_values) / n  # Normalize
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)

    # Only keep the positive frequencies
    pos_mask = freqs >= 0
    return freqs[pos_mask], fft_magnitude[pos_mask]

## helper functions
def find_data_path(date, runnr, data_path=None):
    RUNNR = runnr
    DATE = dateparser.parse(date)

    path_data = get_data_path(data_path)
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


def get_total_time(date, runnr, ch, data_path=None, mca=False):
    file_path = find_data_path(date, runnr, data_path)
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


def get_settings_files(date, runnr, data_path=None):
    file_path = find_data_path(date, runnr, data_path)
    files = list(file_path.glob("settings/*-stats-*"))
    return files


def get_total_counts(date, runnr, ch, data_path=None):
    files = get_settings_files(date, runnr, data_path)
    cts_tot = 0
    for f in files:
        cts0 = read_input_counts_from_settings(f, ch=ch)
        cts_tot += cts0
    return cts_tot


def combine_settings_files(dates, runnrs, data_path=None):
    all_files = []
    for d, r in zip(dates, runnrs):
        files = get_settings_files(date=d, runnr=r, data_path=data_path)
        all_files.append(files)
    all_files_flatten = [x for xs in all_files for x in xs]
    return all_files_flatten


def calculate_neutron_yield(date, runnr, ch=9, data_path=None):
    alpha_counts = get_total_counts(date, runnr, ch, data_path)
    time_total = get_total_time(date, runnr, ch, data_path)
    alpha_cr = alpha_counts / time_total
    d = 6.7  # cm alpha detector-neutron source distance
    alpha_area = 4.8 * 4.8  # cm2
    phi_a = alpha_cr / alpha_area  # flux at alpha detector
    Y0 = 4 * np.pi * d**2 * phi_a  # neutron yield (n/s)
    # alpha_frac = 0.91  # correction factor for true alphas
    alpha_frac = 1
    return alpha_frac * Y0


def calculate_neutron_flux(date, runnr, ch, L=30):
    Y0 = calculate_neutron_yield(date=date, runnr=runnr, ch=ch)
    phi_s = Y0 / (4 * np.pi * L**2)  # neutron flux on sample
    return phi_s


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
    xa = 4.8  # cm alpha detector active area
    alpha_area = xa * xa  # cm2
    L_alpha = (d / L) * (S / 2) * 2
    sample_area = L_alpha * L_alpha
    fa = sample_area / alpha_area
    return fa


def create_directory(directory):
    """
    Ensure that the directory exists; if it does not, create it.

    Parameters
    ----------
    directory : str
        Directory path to check/create.

    Returns
    -------
    None.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")


def data_cleanup(runs_dict):
    # Initial filters
    xrange_labr = [-0.5, 0.5]
    yrange_labr = [-0.62, 0.655]
    trange_labr = [-20, 60]

    xrange_cebr = [-0.518, 0.526]
    yrange_cebr = [-0.685, 0.655]
    trange_cebr = [-20, 60]

    date = runs_dict["date"]
    runnr = runs_dict["run"]
    ch = runs_dict["channel"]
    dfs = read_parquet_api.read_parquet_file(date=date, runnr=runnr, ch=ch)

    dfs["dt"] = dfs["dt"] + runs_dict["dt"]
    if ch == 5:
        dfs["energy_orig"] = dfs["energy_orig"] + 5435.24 - runs_dict["erg846"]
        dfxy = dftxy(
            df=dfs,
            xrange=xrange_labr,
            yrange=yrange_labr,
            trange=trange_labr,
            xkey="X2",
            ykey="Y2",
            tkey="dt",
        )
    elif ch == 4:
        dfs["energy_orig"] = dfs["energy_orig"] + 6565.2 - runs_dict["erg846"]
        dfxy = dftxy(
            df=dfs,
            xrange=xrange_cebr,
            yrange=yrange_cebr,
            trange=trange_cebr,
            xkey="X2",
            ykey="Y2",
            tkey="dt",
        )

    df_final = dfxy[["dt", "energy", "energy_orig", "LaBr[y/n]", "X2", "Y2"]]
    df_final["dt"] *= 1e-9
    return df_final


## plotting functions
def plot_dt(dt, tbins, trange, ax=None, **kwargs):
    """plot histogram of time differences"""
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    ax.hist(
        dt,
        bins=tbins,
        range=trange,
        edgecolor="black",
        **kwargs,
    )
    ax.set_xlabel("Time [ns]")


def plot_dz(z, zbins, zrange):
    """plot histogram of Z"""
    plt.figure()
    plt.hist(
        z, bins=zbins, range=zrange, alpha=0.7, edgecolor="black", label="Z histogram"
    )
    plt.xlabel("Z [cm]")
    plt.grid()
    plt.legend()


def plot_2D_alphas(
    df,
    xkey="X",
    ykey="Y",
    colormap="plasma",
    hexbins=80,
    xyplane=(-0.9, 0.9, -0.9, 0.9),
    ax=None,
    **kwargs,
):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    df.plot.hexbin(
        x=xkey,
        y=ykey,
        gridsize=hexbins,
        cmap=colormap,
        ax=ax,
        colorbar=True,
        extent=xyplane,
        **kwargs,
    )  # , bins="log")


def plot_2Dposition(X, Y, pbins, Vmax=None):
    """Plot 2D intensity map"""
    result, xedges, yedges = np.histogram2d(X, Y, pbins)
    plt.figure()
    if Vmax is not None:
        plt.imshow(
            result,
            extent=[-1, 1, -1, 1],
            interpolation="bilinear",
            cmap="inferno",
            vmax=Vmax,
            origin="lower",
        )
    else:
        plt.imshow(result, interpolation="bilinear", cmap="inferno", origin="lower")
    plt.title("Intensity Map")
    plt.show()


def plot_2Dposition_hexbins(df, xkey, ykey, ax, colormap="Grays"):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    df.plot.hexbin(x=xkey, y=ykey, gridsize=80, cmap=colormap, colorbar=None, ax=ax)


def plot_energy(en, ebins, erange=[0, 10], ax=None, log=True, **kwargs):
    """Plot energy histogram"""
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
    ax.hist(en, bins=ebins, range=erange, **kwargs)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_title("Energy")


def plot_3D(X, Y, Z, Vmax=None):
    """Plot 3D cloud using Mayavi"""
    from mayavi import mlab

    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    r, ed = np.histogramdd((X, Y, Z), bins=15)
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    if Vmax is not None:
        mlab.pipeline.volume(mlab.pipeline.scalar_field(r), vmin=0, vmax=Vmax)
    else:
        mlab.pipeline.volume(mlab.pipeline.scalar_field(r))
    # mlab.axes()
    mlab.outline()
    # mlab.orientation_axes()


def plot_scatter_density(df):
    """Viridis-like" colormap with white background"""
    white_viridis = LinearSegmentedColormap.from_list(
        "white_viridis",
        [
            (0, "#ffffff"),
            (1e-20, "#440053"),
            (0.2, "#404388"),
            (0.4, "#2a788e"),
            (0.6, "#21a784"),
            (0.8, "#78d151"),
            (1, "#fde624"),
        ],
        N=256,
    )

    fig = plt.figure()
    ax = ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(df.X2, df.Y2, cmap=white_viridis)
    fig.colorbar(density, label="Number of points per pixel")
    plt.show()


def api_xyz(df, det_pos=[0, 22.2, -25.515], toffset=None, use_det=True):
    """Returns X, Y, Z reconstructed positions. Use the raw dataframe.
    if use_det=False, the location of the gamma detector is ignored"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    if toffset is not None:
        df["dt"] = df["dt"] - toffset
    # convert alpha detector hits to cm
    res, xed, yed = np.histogram2d(df.Y2, df.X2, bins=1000)
    resx = res.sum(axis=0)
    maskx = resx > resx.max() / 3
    minx = xed[0:-1][maskx][0]
    maxx = xed[0:-1][maskx][-1]

    resy = res.sum(axis=1)
    masky = resy > resy.max() / 3
    miny = yed[0:-1][masky][0]
    maxy = yed[0:-1][masky][-1]

    mx = (2.4 + 2.4) / (maxx - minx)  # assume 4.8 cm active detector area
    xa = mx * (df["X2"] - maxx) + 2.4
    my = (2.4 + 2.4) / (maxy - miny)  # assume 4.8 cm active detector area
    ya = my * (df["Y2"] - maxy) + 2.4
    za = 6.7 * np.ones(len(xa))  # distance YAP-target (cm)

    # gamma detector location [cm]
    dx = det_pos[0]
    dy = det_pos[1]
    dz = det_pos[2]
    dg = np.sqrt(dx**2 + dy**2 + dz**2)  # cm
    c = 3e10  # speed of light [cm/s]
    # normal directional vectors
    alength = np.sqrt(xa**2 + ya**2 + za**2)  # a
    ux = -xa / alength
    uy = -ya / alength
    uz = za / alength
    # nTOF
    Ea = 3.5  # MeV
    ma = 3727.379  # MeV/c2
    va = np.sqrt(2 * Ea / ma) * 3e10  # cm/s
    ta = alength / va  # alpha travel time [s]
    En = 14.1  # MeV
    mn = 939.56563  # MeV/c2
    vn = np.sqrt(2 * En / mn) * 3e10  # cm/s
    ntof = df.dt * 1e-9 + ta  # neutron time of flight [s], dta

    if use_det:
        theta = np.arccos((ux * dx + uy * dy + uz * dz) / dg)  # radians
        aa = 1 - c**2 / vn**2
        bb = 2 * ntof * c**2 / vn - 2 * dg * np.cos(theta)
        cc = dg**2 - c**2 * ntof**2

        Ln = (-bb - np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)  # cm

        X2 = -xa / alength * Ln
        Y2 = -ya / alength * Ln
        Z2 = za / alength * Ln + za
    else:
        # don't take into account the position of the gamma detector
        Ln = vn * ntof  # neutron travel path (cm)
        # coordinates of interaction
        X2 = Ln * ux  # x-coordinate of interaction (cm)
        Y2 = Ln * uy  # y-coordinate of interaction (cm)
        Z2 = Ln * uz  # z-coordinate of interaction (cm)

    return X2, Y2, Z2
