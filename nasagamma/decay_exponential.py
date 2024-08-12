"""
Exponential decay functions for fitting
"""

import lmfit
import numpy as np
import matplotlib.pyplot as plt


def single_decay_plus_constant(t, A, k, C):
    return A * np.exp(-k * t) + C


def double_decay_plus_constant(t, A1, A2, k1, k2, C):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C


def guess_halflife(x, y):
    ymid = (y.min() + y.max()) / 2
    idx_m = y > ymid
    thalf = x[idx_m][-1]
    return thalf


class Decay_exp:
    def __init__(self, x, y, yerr):
        self.x_data = x
        self.y_data = y
        self.yerr = yerr
        self.degree = None

    def fit_single_decay(self):
        self.degree = 1
        exp_model = lmfit.Model(single_decay_plus_constant, nan_policy="omit")
        A_guess = self.y_data.max()
        k_guess = np.log(2) / guess_halflife(x=self.x_data, y=self.y_data)
        C_guess = self.y_data.min()
        params = exp_model.make_params(A=A_guess, k=k_guess, C=C_guess)
        # Perform the fit
        self.fit_result = exp_model.fit(
            self.y_data, params, t=self.x_data, weights=1.0 / self.yerr
        )

    def fit_double_decay(self):
        self.degree = 2
        exp_model = lmfit.Model(double_decay_plus_constant, nan_policy="omit")
        A_guess1 = self.y_data.max()
        k_guess1 = np.log(2) / guess_halflife(x=self.x_data, y=self.y_data)
        A_guess2 = self.y_data.max()
        k_guess2 = np.log(2) / guess_halflife(x=self.x_data, y=self.y_data)
        C_guess = self.y_data.min()
        params = exp_model.make_params(
            A1=A_guess1, A2=A_guess2, k1=k_guess1, k2=k_guess2, C=C_guess
        )
        # Perform the fit
        self.fit_result = exp_model.fit(
            self.y_data, params, t=self.x_data, weights=1.0 / self.yerr
        )

    def plot(self, ax_fit=None, ax_res=None, show_components=False):
        only_fit = False
        if ax_fit is None and ax_res is None:
            # plt.rc("font", size=12)
            plt.style.use("seaborn-v0_8-darkgrid")
            fig = plt.figure(constrained_layout=True, figsize=(8, 6))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
            ax_res = fig.add_subplot(gs[0, 0])
            ax_fit = fig.add_subplot(gs[1, 0])
        elif ax_fit is None and ax_res is not None:
            # plt.rc("font", size=12)
            plt.style.use("seaborn-v0_8-darkgrid")
            fig = plt.figure(constrained_layout=True, figsize=(8, 6))
            ax_fit = fig.add_subplot()
        elif ax_fit is not None and ax_res is None:
            # plt.rc("font", size=12)
            plt.style.use("seaborn-v0_8-darkgrid")
            only_fit = True

        if only_fit is False:
            ax_res.plot(self.x_data, self.fit_result.residual, ".", ms=10, alpha=0.5)
            ax_res.hlines(y=0, xmin=self.x_data.min(), xmax=self.x_data.max(), lw=3)
            ax_res.set_ylabel("Residual")
            ax_res.set_xlim([self.x_data.min(), self.x_data.max()])
            ax_res.set_xticks([])
        # fig.patch.set_alpha(0.3)

        best_fit = self.fit_result.best_fit
        best_values = self.fit_result.best_values
        ax_fit.set_title(f"Reduced $\chi^2$ = {round(self.fit_result.redchi,4)}")
        ax_fit.errorbar(
            x=self.x_data,
            y=self.y_data,
            yerr=self.yerr,
            fmt="bo",
            alpha=0.5,
            capsize=6,
            markersize=5,
            label="data",
        )
        if self.degree == 1:
            ax_fit.plot(
                self.x_data,
                best_fit,
                "r",
                lw=3,
                alpha=0.7,
                label=f"Decay constant = {best_values['k']:.3E}",
            )
            if show_components:
                A = best_values["A"]
                k = best_values["k"]
                C = best_values["C"]
                ax_fit.plot(
                    [self.x_data.min(), self.x_data.max()], [C, C], lw=3, label="C"
                )
                ax_fit.plot(
                    self.x_data,
                    single_decay_plus_constant(t=self.x_data, A=A, k=k, C=0),
                    lw=3,
                    label="Exponential",
                )
        elif self.degree == 2:
            lab1 = f"Decay constant 1 = {best_values['k1']:.3E}"
            lab2 = f"Decay constant 2 = {best_values['k2']:.3E}"
            ax_fit.plot(
                self.x_data, best_fit, "r", lw=3, alpha=0.7, label=f"{lab1}\n{lab2}"
            )
            if show_components:
                A1 = best_values["A1"]
                k1 = best_values["k1"]
                A2 = best_values["A2"]
                k2 = best_values["k2"]
                C = best_values["C"]
                ax_fit.plot(
                    [self.x_data.min(), self.x_data.max()], [C, C], lw=3, label="C"
                )
                ax_fit.plot(
                    self.x_data,
                    single_decay_plus_constant(t=self.x_data, A=A1, k=k1, C=0),
                    lw=3,
                    label="Exponential 1",
                )
                ax_fit.plot(
                    self.x_data,
                    single_decay_plus_constant(t=self.x_data, A=A2, k=k2, C=0),
                    lw=3,
                    label="Exponential 2",
                )
        ax_fit.legend()
        ax_fit.set_xlabel("Time (us)")
        ax_fit.set_ylabel("Counts")
