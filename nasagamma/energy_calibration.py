"""
Energy calibration functions
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations


def ecalibration(
    mean_vals,
    erg,
    channels,
    n=1,
    e_units="keV",
    plot=False,
    residual=True,
    fig=None,
    ax_fit=None,
    ax_res=None,
):
    """
    Perform energy calibration with LMfit. Polynomial degree = n.

    Parameters
    ----------
    mean_vals : list or numpy array
        mean values of fitted peaks (in channel numbers).
    erg : list or numpy array
        energy values corresponding to mean_vals.
    channels : list or numpy array
        channel values.
    n : integer, optional
        polynomial degree. The default is 1.
    e_units : string, optional
        energy units. The default is "keV".
    plot : bool, optional
        plot calibration curve and calibration points. The default is False.

    Returns
    -------
    predicted : numpy array
        predicted energy values spanning all channel values.
    fit : LMfit object.
        curve fit.

    """
    y0 = np.array(erg, dtype=float)
    x0 = np.array(mean_vals, dtype=float)
    poly_mod = lmfit.models.PolynomialModel(degree=n)
    pars = poly_mod.guess(y0, x=x0)
    model = poly_mod
    fit = model.fit(y0, params=pars, x=x0)
    predicted = fit.eval(x=channels)
    ye = fit.eval_uncertainty()
    coeffs = list(fit.best_values.values())
    terms = [f"${coeffs[0]:.3E}$", f"${coeffs[1]:.3E}x$"]
    for i, c in enumerate(coeffs):
        if i >= 2:
            terms.append(rf"${c:.3E}x^{i}$")
    equation = " + ".join(terms)
    if plot:
        plt.rc("font", size=14)
        plt.style.use("seaborn-v0_8-darkgrid")
        x_offset = 100  # max(channels)*0.01
        if fig is None:
            fig = plt.figure(constrained_layout=False, figsize=(12, 8))
        if residual:
            if ax_res is None:
                gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
                ax_res = fig.add_subplot(gs[0, 0])
            ax_res.plot(mean_vals, fit.residual, ".", ms=15, alpha=0.5, color="red")
            ax_res.hlines(y=0, xmin=min(channels) - x_offset, xmax=max(channels), lw=3)
            ax_res.set_ylabel("Residual")
            ax_res.set_xlim([min(channels) - x_offset, max(channels)])
            ax_res.set_xticks([])
        # else:
        #     gs = fig.add_gridspec(1, 1)

        if ax_fit is None:
            if residual:
                ax_fit = fig.add_subplot(gs[1, 0])
            else:
                gs = fig.add_gridspec(1, 1)
                ax_fit = fig.add_subplot(gs[0, 0])

        ax_fit.set_title(f"Reduced $\chi^2$ = {round(fit.redchi,4)}")
        ax_fit.errorbar(
            mean_vals,
            erg,
            yerr=ye,
            ecolor="red",
            elinewidth=5,
            capsize=12,
            capthick=3,
            marker="s",
            mfc="black",
            mec="black",
            markersize=7,
            ls=" ",
            lw=3,
            label="Data",
        )
        ax_fit.plot(
            channels,
            predicted,
            ls="--",
            lw=3,
            color="green",
            label=f"Predicted: {equation}",
        )
        ax_fit.set_xlim([min(channels) - x_offset, max(channels)])
        ax_fit.set_xlabel("Channels")
        ax_fit.set_ylabel(f"Energy [{e_units}]")
        ax_fit.legend()
        # ax_fit.set_xlim([0,10])
        # plt.style.use("default")
    return predicted, fit


def cal_table(
    ch_lst,
    e_lst,
    sig_lst,
    t_scale=[1, 1.8],
    decimals=3,
    e_units=None,
    ax=None,
    fig=None,
):
    if fig is None:
        fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    if ax is None:
        ax = fig.add_subplot()

    chs = np.round(np.array(ch_lst), decimals=decimals)
    ergs = np.round(np.array(e_lst), decimals=decimals)
    sigs = np.round(np.array(sig_lst), decimals=decimals)
    cols = ["N", "centroid", f"energy [{e_units}]", f"sigma [{e_units}]"]
    N = np.arange(1, len(chs) + 1, 1)
    rs = np.array([N, chs, ergs, sigs]).T
    colors = [["lightblue"] * len(cols)] * len(rs)
    df = pd.DataFrame(rs, columns=cols)
    df = df.astype({"N": "int32"})
    df = df.astype({"N": "str"})

    t = ax.table(
        cellText=df.values,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colWidths=[1 / 8, 1 / 3, 1 / 3, 1 / 3],
        colColours=["palegreen"] * len(cols),
        cellColours=colors,
    )
    t.scale(t_scale[0], t_scale[1])
    t.auto_set_font_size(False)
    t.set_fontsize(14)
    ax.axis("off")

def is_within_epsilon(a, b, epsilon=0.5):
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    return abs(a - b) <= epsilon

def smart_calibration(channels, set_ergs=[197.32, 511, 661.657, 846.78]):
    red_chi = []
    epsilon = []
    res = []
    fit = []
    # Generate all combinations of channels out of all energies
    # You may need to have more selected peaks (channels) than set energies
    channel_groups = list(combinations(channels, len(set_ergs)))
    for group in channel_groups:
        mean_vals = group
        ch = channels
        pred_erg, efit = ecalibration(
            mean_vals=mean_vals, erg=set_ergs, channels=ch, n=1,
            plot=False, residual=True
        )
        epsilon.append(is_within_epsilon(efit.best_fit, set_ergs, epsilon=2))
        red_chi.append(efit.redchi)
        res.append(efit.residual)
        fit.append(efit)
    
    epsilon = np.array(epsilon)
    red_chi = np.array(red_chi)
    res = np.array(res)
     
    min1 = red_chi.argmin() # minimum reduced chi-squared
    min2 = np.abs(res).sum(axis=1).argmin() # smallest residual
    min3 = epsilon.sum(axis=1).argmax() # smallest energy difference
    
    print(min1, min2, min3)
    
    efit = fit[min1]
    return efit
    
    
    
    
    
    
    
    
    
    