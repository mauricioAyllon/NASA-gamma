"""
FWHM functions
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fwhm1(E, a, b):
    return a + b * np.sqrt(E)


def fwhm2(E, a, b, c):
    return a + b * np.sqrt(E + c * E ** 2)


def fwhm_vs_erg(energies, fwhms, x_units, e_units, order=2, fig=None, ax=None):
    plt.rc("font", size=14)
    plt.style.use("seaborn-v0_8-darkgrid")
    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    if ax is None:
        ax = fig.add_subplot()

    energies = np.array(energies)
    fwhms = np.array(fwhms)

    if order == 1:
        gmodel = lmfit.Model(fwhm1)
        fit = gmodel.fit(fwhms, E=energies, a=0, b=0)
        best_vals = fit.best_values
        ye = fit.eval_uncertainty()
        a = round(best_vals["a"], 5)
        b = round(best_vals["b"], 5)
        y = fwhm1(E=energies, a=a, b=b)
        erg_continuous = np.linspace(energies[0], energies[-1], num=100)
        y = fwhm1(E=erg_continuous, a=a, b=b)
        # predicted = result.eval(x=channels)

        ax.errorbar(
            energies,
            fwhms,
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
        ax.plot(
            erg_continuous,
            y,
            ls="-",
            lw=3,
            color="green",
            label=f"a={a}\nb={b}",
        )

        ax.legend(loc="best")
        ax.set_xlabel(f"{x_units}")
        ax.set_ylabel(f"FWHM [{e_units}]")
        ax.set_title("$a+b\sqrt{E}$")
    elif order == 2:
        gmodel = lmfit.Model(fwhm2)
        fit = gmodel.fit(fwhms, E=energies, a=0, b=0, c=0)
        best_vals = fit.best_values
        ye = fit.eval_uncertainty()
        a = round(best_vals["a"], 5)
        b = round(best_vals["b"], 5)
        c = round(best_vals["c"], 5)
        # erg_continuous = np.arange(energies[0], energies[-1], 0.1)
        erg_continuous = np.linspace(energies[0], energies[-1], num=100)
        y = fwhm2(E=erg_continuous, a=a, b=b, c=c)
        # predicted = result.eval(x=channels)

        ax.errorbar(
            energies,
            fwhms,
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
        ax.plot(
            erg_continuous,
            y,
            ls="-",
            lw=3,
            color="green",
            label=f"a={a}\nb={b}\nc={c}",
        )

        ax.legend(loc="best")
        ax.set_xlabel(f"{x_units}")
        ax.set_ylabel(f"FWHM [{e_units}]")
        ax.set_title("$a+b\sqrt{E+cE^2}$")
    return fit


def fwhm_extrapolate(energies, fit, order=1, ax=None, fig=None):
    # plot extrapolated best fit line
    if fig is None:
        fig = plt.figure(constrained_layout=False, figsize=(12, 8))
    if ax is None:
        ax = fig.add_subplot()
    dic_vals = fit.best_values
    if order == 1:
        a = dic_vals["a"]
        b = dic_vals["b"]
        y = fwhm1(E=energies, a=a, b=b)
        ax.plot(energies, y, ls="--", lw=3, color="k")
    elif order == 2:
        a = dic_vals["a"]
        b = dic_vals["b"]
        c = dic_vals["c"]
        y = fwhm2(E=energies, a=a, b=b, c=c)
        ax.plot(energies, y, ls="--", lw=3, color="k")
    else:
        print("Invalid polynomial order")


def fwhm_table(
    x_lst,
    fwhm_lst,
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

    flag = 1
    if 0 in x_lst:
        x_lst.pop(0)
        fwhm_lst.pop(0)
        flag = 0

    percent = np.array(fwhm_lst) / np.array(x_lst) * 100
    xs = np.round(np.array(x_lst), decimals=decimals)
    fwhms = np.round(np.array(fwhm_lst), decimals=decimals)
    fwhm_perc = np.round(percent, decimals=2)
    cols = ["N", f"energy [{e_units}]", f"FWHM [{e_units}]", "FWHM [%]"]
    N = np.arange(1, len(xs) + 1, 1)
    rs = np.array([N, xs, fwhms, fwhm_perc]).T
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

    if flag == 0:
        x_lst.insert(0, 0)
        fwhm_lst.insert(0, 0)
