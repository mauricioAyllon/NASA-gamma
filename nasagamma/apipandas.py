"""
Functions used to analyze API data

"""
import numpy as np
import matplotlib.pyplot as plt

# from mayavi import mlab
import pandas as pd

# import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap


def test_limits(df, elog=True, Vmax=None, ekey="energy", tkey="dt", xkey="X", ykey="Y"):
    """Plots all data to define energy, time, and xy limits. df is a pandas dataframe"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    plt.figure()
    plt.hist(df[ekey], 256)
    if elog:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.title("Energy")

    res, xed, yed = np.histogram2d(df[ykey], df[xkey], bins=100)
    plt.figure()
    if Vmax != None:
        plt.imshow(
            res,
            extent=[df[xkey].min(), df[xkey].max(), df[ykey].min(), df[ykey].max()],
            cmap="inferno",
            interpolation="bilinear",
            vmax=Vmax,
            origin="lower",
        )
    else:
        plt.imshow(
            res,
            extent=[df[xkey].min(), df[xkey].max(), df[ykey].min(), df[ykey].max()],
            cmap="inferno",
            interpolation="bilinear",
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


def plot_dt(dt, tbins, trange):
    """plot histogram of time differences"""
    plt.figure()
    plt.hist(
        dt,
        bins=tbins,
        range=trange,
        alpha=0.7,
        edgecolor="black",
        label="Time histogram",
    )
    plt.xlabel("Time [ns]")
    plt.grid()
    plt.legend()


def plot_dz(z, zbins, zrange):
    """plot histogram of Z"""
    plt.figure()
    plt.hist(
        z, bins=zbins, range=zrange, alpha=0.7, edgecolor="black", label="Z histogram"
    )
    plt.xlabel("Z [cm]")
    plt.grid()
    plt.legend()


def plot_2D_alphas(df, xkey="X", ykey="Y"):
    hexbins = 80  # x-y bins
    xyplane = (-0.9, 0.9, -0.9, 0.9)  # x and y limits
    colormap = "plasma"
    df.plot.hexbin(
        x=xkey,
        y=ykey,
        gridsize=hexbins,
        cmap=colormap,
        # ax=ax_api_xy,
        colorbar=True,
        extent=xyplane,
    )  # , bins="log")


def plot_2Dposition(X, Y, pbins, Vmax=None):
    """Plot 2D intensity map"""
    result, xedges, yedges = np.histogram2d(X, Y, pbins)
    plt.figure()
    if Vmax != None:
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


def plot_energy(en, ebins, erange, log=True):
    """Plot energy histogram"""
    en, ed = np.histogram(en, bins=800, range=[0, 40000])
    plt.figure()
    plt.plot(en)
    if log:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.title("Energy")


def plot_3D(X, Y, Z, Vmax=None):
    """Plot 3D cloud using Mayavi"""
    from mayavi import mlab

    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    r, ed = np.histogramdd((X, Y, Z), bins=15)
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    if Vmax != None:
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


def api(df, xrange, yrange, erange, trange, det_pos, toffset=None, use_det=True):
    """Returns X, Y, Z reconstructed positions. Use the raw dataframe.
    if use_det=False, the location of the gamma detector is ignored"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    df_txye = dftxye(df, xrange, yrange, erange, trange)
    if toffset != None:
        df_txye["dt"] = df_txye["dt"] - toffset
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
    xa = mx * (df_txye["X2"] - maxx) + 2.4
    my = (2.4 + 2.4) / (maxy - miny)  # assume 4.8 cm active detector area
    ya = my * (df_txye["Y2"] - maxy) + 2.4
    za = 6 * np.ones(len(xa))  # distance YAP-target (cm)

    # gamma detector location [cm]
    dx = det_pos[0]
    dy = det_pos[1]
    dz = det_pos[2]
    dg = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # cm
    c = 3e10  # speed of light [cm/s]
    # normal directional vectors
    alength = np.sqrt(xa ** 2 + ya ** 2 + za ** 2)  # a
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
    ntof = df_txye.dt * 1e-9 + ta  # neutron time of flight [s], dta

    if use_det:
        theta = np.arccos((ux * dx + uy * dy + uz * dz) / dg)  # radians
        aa = 1 - c ** 2 / vn ** 2
        bb = 2 * ntof * c ** 2 / vn - 2 * dg * np.cos(theta)
        cc = dg ** 2 - c ** 2 * ntof ** 2

        Ln = (-bb - np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)  # cm

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
