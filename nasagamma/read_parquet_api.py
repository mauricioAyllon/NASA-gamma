"""
Read ROOTS parquet files
"""

import dateparser
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
import matplotlib
from nasagamma import apipandas as api
import pkg_resources
from pathlib import Path

# matplotlib.use('qtagg')
def get_data_path(data_path=None):
    # directory to monitor
    if data_path is None:
        data_path_file = pkg_resources.resource_filename("nasagamma", "data-path.txt")
        path_file = Path(data_path_file)
        with path_file.open() as f:
            data_path_txt = f.readline().strip()
    else:
        data_path_txt = data_path
    return Path(data_path_txt)


def read_parquet_file(date, runnr, ch, flat_field=False, data_path_txt=None):
    # only channels 4 (LaBr==True) and 5 (LaBr==False)
    RUNNR = runnr
    DATE = dateparser.parse(date)
    if DATE is None:
        print("ERROR: cannot parse date")

    DATA_PATH = get_data_path(data_path_txt)
    DATA_DIR = DATA_PATH / f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"

    if not DATA_DIR.is_dir():
        print(
            (
                f"ERROR: cannot find directory {DATA_DIR}."
                "ERROR: Make sure you create a text file with your data path"
                " named 'data-path.txt' in the directory NASA-gamma/nasagamma"
                " For example, my data path is: "
                "C:/Users/mayllonu/Documents/NASA-GSFC/Technical/Data-LBL"
            )
        )

    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{RUNNR:05d}"
    FILE = DATA_DIR / fname
    if not FILE.is_dir():
        print(f"ERROR: cannot find file {FILE}")

    # load data
    files = list(FILE.glob(f"parquet-data/{fname}-*-pandas.parquet"))
    df = pd.read_parquet(files[0])

    if len(files) > 1:
        for f in files[1:]:
            df0 = pd.read_parquet(f)
            df = pd.concat([df, df0])
    df = api.calc_own_pos(df)
    if flat_field:
        return df
    else:
        df["dt"] *= 1e9  # to ns
        if ch == 4 or ch == 3:
            df = df[df["LaBr[y/n]"] == True]
        elif ch == 5:
            df = df[df["LaBr[y/n]"] == False]
        df.reset_index(drop=True, inplace=True)
        return df


def read_parquet_file_from_path(filepath, ch):
    # load data
    path = Path(filepath)
    files = list(path.glob("parquet-data/*-pandas.parquet"))
    df = pd.read_parquet(files[0])

    if len(files) > 1:
        for f in files[1:]:
            df0 = pd.read_parquet(f)
            df = pd.concat([df, df0])

    df["dt"] *= 1e9  # to ns
    if ch == 4:
        df = df[df["LaBr[y/n]"] == True]
    elif ch == 5:
        df = df[df["LaBr[y/n]"] == False]
    df = api.calc_own_pos(df)
    df.reset_index(drop=True, inplace=True)
    return df


def initialize_plots(df, ax_g, ax_t, ax_xy, ax_xz):
    tr = [-40, 80]  # dt range
    ebins = 2048  # energy bins
    hexbins = 80  # x-y bins
    tbins = 512  # time bins
    xyplane = (-0.9, 0.9, -0.9, 0.9)  # x and y limits
    # xyplane = (-0.1,0.1,-0.1,0.1) # for X_alpha, y_alpha
    # magma, plt.cm.BuGn_r, plt.cm.Greens, plasma, jet, viridis, cvidis
    colormap = "plasma"
    gam, e = plot_energy_hist(df=df, ebins=ebins, ax=ax_g)
    plot_time_hist(df=df, tbins=tbins, trange=tr, ax=ax_t)
    plot_xy(df, hexbins, colormap, xyplane, ax=ax_xy, cbar=True)


def plot_energy_hist(df, ebins, ax):
    gam, e = np.histogram(df["energy"], bins=ebins)
    ch = np.arange(0, ebins, 1)
    ax.plot(ch, gam, color="green")
    return gam, e


def plot_time_hist(df, tbins, trange, ax):
    df["dt"].plot.hist(bins=tbins, ax=ax, range=trange, alpha=0.7, edgecolor="black")


def plot_xy(df, hexbins, colormap, xyplane, ax, cbar=True):
    df.plot.hexbin(
        x="X2",
        y="Y2",
        gridsize=hexbins,
        cmap=colormap,
        ax=ax,
        colorbar=cbar,
        extent=xyplane,
    )  # , bins="log")
