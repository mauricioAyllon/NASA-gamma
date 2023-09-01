"""
Efficiency calibration class
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Efficiency:
    def __init__(self, t_half, A0, Br, livetime, t_elapsed, which_peak=0):
        self.t_half = float(t_half)  # s
        self.A0 = float(A0)  # Bq
        self.Br = float(Br)  # < 1
        self.livetime = float(livetime)  # s
        self.t_elapsed = float(t_elapsed)  # s
        self.which_peak = which_peak
        self.error = 0

    def calculate_N_emitted(self):
        A0_bq = self.A0
        lmda = np.log(2) / (self.t_half)  # s^-1
        A_now = A0_bq * np.exp(-lmda * self.t_elapsed) * self.Br  # [g/s]
        N_emitted = np.array(A_now) * self.livetime
        return N_emitted

    def calculate_N_detected(self, fit_obj):
        mean_val = fit_obj.peak_info[self.which_peak][f"mean{self.which_peak+1}"]
        area_val = fit_obj.peak_info[self.which_peak][f"area{self.which_peak+1}"]
        N_detected = float(area_val)
        self.mean_val = float(mean_val)
        return N_detected

    def calculate_efficiency(self, fit_obj):
        N_emitted = self.calculate_N_emitted()
        N_detected = self.calculate_N_detected(fit_obj)
        self.eff = N_detected / N_emitted

    def calculate_error(
        self, fit_obj, t_half_sig, A0_sig, Br_sig, livetime_sig, t_elapsed_sig
    ):
        N_detected = fit_obj.peak_info[self.which_peak][f"area{self.which_peak+1}"]
        N_detected_sig = fit_obj.peak_err[self.which_peak][
            f"area_err{self.which_peak+1}"
        ]
        # denominator function denoted as f
        # sig_f = sqrt(df_dA0**2 * sig_A0**2 + ...)
        lmd = np.log(2) / self.t_half
        lmd_sig = lmd * t_half_sig / self.t_half
        f = self.A0 * np.exp(-lmd * self.t_elapsed) * self.Br * self.livetime
        dfdA0 = np.exp(-lmd * self.t_elapsed) * self.Br * self.livetime
        dfdlmd = (
            self.A0
            * self.Br
            * self.livetime
            * (-self.t_elapsed)
            * np.exp(-lmd * self.t_elapsed)
        )
        dfdte = (
            self.A0 * self.Br * self.livetime * (-lmd) * np.exp(-lmd * self.t_elapsed)
        )
        dfdb = self.A0 * np.exp(-lmd * self.t_elapsed) * self.livetime
        dfdtc = self.A0 * np.exp(-lmd * self.t_elapsed) * self.Br
        sig_f = np.sqrt(
            dfdA0**2 * A0_sig
            + dfdlmd**2 * lmd_sig**2
            + dfdte**2 * t_elapsed_sig**2
            + dfdb**2 * Br_sig**2
            + dfdtc**2 * livetime_sig**2
        )
        self.error = f * np.sqrt((N_detected_sig / N_detected) ** 2 + (sig_f / f) ** 2)


def eff_fit(en, eff, order=1, fig=None, ax=None):
    plt.rc("font", size=14)
    plt.style.use("seaborn-darkgrid")

    def eff_func(x, a0, a1, a2, a3):
        return (a0 + a1 * np.log(x) + a2 * (np.log(x)) ** 2 + a3 * (np.log(x)) ** 3) / x

    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    if ax is None:
        ax = fig.add_subplot()

    energies = np.array(en)
    effs = np.array(eff)
    if order == 1:
        emodel = lmfit.Model(eff_func)
        fit = emodel.fit(effs, x=energies, a0=0, a1=0, a2=0, a3=0)
        best_vals = fit.best_values
        ye = fit.eval_uncertainty()
        a0 = best_vals["a0"]
        a1 = best_vals["a1"]
        a2 = best_vals["a2"]
        a3 = best_vals["a3"]
        erg_continuous = np.linspace(energies[0], energies[-1], num=100)
        y = eff_func(x=erg_continuous, a0=a0, a1=a1, a2=a2, a3=a3)

        ax.errorbar(
            energies,
            effs,
            yerr=ye,
            ecolor="red",
            elinewidth=5,
            capsize=12,
            capthick=3,
            marker="s",
            # mfc="black",
            # mec="black",
            markersize=7,
            ls=" ",
            lw=3,
            # label="Distance = 2.5 cm",
        )
    ax.plot(
        erg_continuous,
        y,
        ls="-",
        lw=3,
        # color="green",
    )

    ax.set_title(r"$eff = \frac{a0 + a1ln(E) + a2ln(E)^2 + a3ln(E)^3}{E}$")
    return ye


def eff_table(
    x_lst,
    eff_lst,
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

    xs = np.round(np.array(x_lst), decimals=decimals)
    effs = np.round(np.array(eff_lst), decimals=decimals)
    cols = ["N", f"energy [{e_units}]", "efficiency [%]"]
    N = np.arange(1, len(xs) + 1, 1)
    rs = np.array([N, xs, effs]).T
    colors = [["lightblue"] * len(cols)] * len(rs)
    df = pd.DataFrame(rs, columns=cols)
    df = df.astype({"N": "int32"})
    df = df.astype({"N": "str"})

    t = ax.table(
        cellText=df.values,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colWidths=[1 / 8, 1 / 2, 1 / 2],
        colColours=["palegreen"] * len(cols),
        cellColours=colors,
    )
    t.scale(t_scale[0], t_scale[1])
    t.auto_set_font_size(False)
    t.set_fontsize(14)
    ax.axis("off")
