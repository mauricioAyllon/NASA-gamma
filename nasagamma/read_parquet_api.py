"""
Read ROOTS binary file 
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

# matplotlib.use('qtagg')


def read_parquet_file(date, runnr, ch, flat_field=False):
    import ROOTS

    # only channels 4 (LaBr==True) and 5 (LaBr==False)
    RUNNR = runnr
    DATE = dateparser.parse(date)
    if DATE is None:
        print("ERROR: cannot parse date")
        sys.exit(1)

    # directory to monitor
    DATA_DIR = (
        ROOTS.helper.get_save_data_dir()
        / f"{DATE.year}-{DATE.month:02d}-{DATE.day:02d}"
    )
    if not DATA_DIR.is_dir():
        print(f"ERROR: cannot find directory {DATA_DIR}.")
        sys.exit(2)

    fname = f"RUN-{DATE.year}-{DATE.month:02d}-{DATE.day:02d}-{RUNNR:05d}"
    FILE = DATA_DIR / fname
    if not FILE.is_dir():
        print(f"ERROR: cannot find file {FILE}")
        sys.exit(3)

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
    from pathlib import Path

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


# def xySelect(eclick, erelease):
#     'eclick and erelease are the press and release events'
#     ax_g.clear()
#     ax_t.clear()
#     x1, y1 = eclick.xdata, eclick.ydata
#     x2, y2 = erelease.xdata, erelease.ydata
#     print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
#     #print(" The button you used were: %s %s" % (eclick.button, erelease.button))
#     xlim = (dfnew['X'] > x1) & (dfnew['X'] < x2)
#     ylim = (dfnew['Y'] > y1) & (dfnew['Y'] < y2)
#     dfxy = dfnew[xlim & ylim]
#     plot_energy_hist(df=dfxy, ebins=ebins)
#     plot_time_hist(df=dfxy, tbins=tbins, trange=tr)
#     fig.canvas.draw_idle()
#     toggle_selector.update()


# def enSelect(emin, emax):
#     idxmin = int(round(emin))
#     idxmax = int(round(emax))
#     print('emin: ', idxmin)
#     print('emax: ', idxmax)
#     ax_t.clear()
#     ax_xy.clear()
#     elim = (dfnew['energy'] > e[idxmin]) & (dfnew['energy'] < e[idxmax])
#     dfe = dfnew[elim]
#     plot_time_hist(df=dfe, tbins=tbins, trange=tr)
#     plot_xy(dfe, hexbins, colormap, xyplane, cbar=False)
#     fig.canvas.draw_idle()
#     espan.update()
#     tspan.update()
#     #espan.interactive


# def timeSelect(tmin, tmax):
#     print('tmin: ', round(tmin,3))
#     print('tmax: ', round(tmax,3))
#     ax_g.clear()
#     ax_xy.clear()
#     tlim = (dfnew['dt'] > tmin) & (dfnew['dt'] < tmax)
#     dft = dfnew[tlim]
#     plot_energy_hist(df=dft, ebins=ebins)
#     plot_xy(dft, hexbins, colormap, xyplane, cbar=False)
#     fig.canvas.draw_idle()
#     espan.update()
#     tspan.update()
#     #tspan.interactive
#     tspan.active = True
#     #save_csv(gam)

# def save_csv(arr):
#     df = pd.DataFrame(data=arr, columns=["counts"])
#     df.to_csv("third_time_peak.csv", index=False)
#     pass


# # drawtype is 'box' or 'line' or 'none'
# toggle_selector = RectangleSelector(ax_xy, xySelect,
#                                         useblit=True,
#                                         button=[1, 3],  # don't use middle button
#                                         minspanx=1, minspany=1,
#                                         spancoords='pixels',
#                                         interactive=True,
#                                         props = dict(facecolor='white', edgecolor='black',
#                                                     alpha=0.5, fill=True),
#                                         )

# # set useblit True on gtkagg for enhanced performance
# espan = SpanSelector(ax_g, enSelect, 'horizontal', useblit=True, interactive=True,
#                     props=dict(alpha=0.4, facecolor='red'))

# tspan = SpanSelector(ax_t, timeSelect, 'horizontal', useblit=True, interactive=True,
#                     props=dict(alpha=0.4, facecolor='yellow'))

# plt.show()
