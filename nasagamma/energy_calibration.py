"""
Energy calibration functions
"""
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb
from scipy.stats import linregress


class EnergyCalibration:
    def __init__(self, mean_vals, erg, channels, n=1, e_units="keV"):
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

        Returns
        -------
        None.
        """
        if len(mean_vals) != len(erg):
            raise ValueError(
                f"mean_vals and erg must have the same length, "
                f"got {len(mean_vals)} and {len(erg)}"
            )
        if len(mean_vals) < n + 1:
            raise ValueError(
                f"Need at least {n + 1} points to fit a degree-{n} polynomial, "
                f"got {len(mean_vals)}"
            )
        self.mean_vals = np.array(mean_vals, dtype=float)
        self.erg = np.array(erg, dtype=float)
        self.channels = np.array(channels, dtype=float)
        self.n = n
        self.e_units = e_units
        self.predicted = None
        self.fit = None
        self._calibrate()

    def _calibrate(self):
        """
        Run the polynomial calibration fit.

        Returns
        -------
        None.
        """
        poly_mod = lmfit.models.PolynomialModel(degree=self.n)
        pars = poly_mod.guess(self.erg, x=self.mean_vals)
        self.fit = poly_mod.fit(self.erg, params=pars, x=self.mean_vals)
        self.predicted = self.fit.eval(x=self.channels)

    def _build_equation(self):
        """
        Build a string representation of the calibration equation.

        Returns
        -------
        string
            equation string for plot labels.
        """
        coeffs = list(self.fit.best_values.values())
        terms = [f"${coeffs[0]:.3E}$", f"${coeffs[1]:.3E}x$"]
        for i, c in enumerate(coeffs):
            if i >= 2:
                terms.append(rf"${c:.3E}x^{i}$")
        return " + ".join(terms)

    def metadata(self):
        """
        Return calibration metadata as a dictionary.

        Returns
        -------
        dict
            dictionary containing calibration parameters and results.
        """
        return {
            "n": self.n,
            "e_units": self.e_units,
            "mean_vals": self.mean_vals,
            "erg": self.erg,
            "redchi": self.fit.redchi if self.fit is not None else None,
            "coefficients": list(self.fit.best_values.values()) if self.fit is not None else None,
        }

    def plot(
        self,
        residual=True,
        ax_fit=None,
        ax_res=None,
    ):
        ye = self.fit.eval_uncertainty(x=self.mean_vals)
        equation = self._build_equation()
    
        plt.rc("font", size=14)
        plt.style.use("seaborn-v0_8-darkgrid")
        x_offset = 100
    
        if residual:
            if ax_res is None or ax_fit is None:
                fig = plt.figure(constrained_layout=False, figsize=(12, 8))
                gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
                ax_res = fig.add_subplot(gs[0, 0])
                ax_fit = fig.add_subplot(gs[1, 0])
            ax_res.plot(
                self.mean_vals, self.fit.residual, ".", ms=15, alpha=0.5, color="red"
            )
            ax_res.hlines(
                y=0,
                xmin=min(self.channels) - x_offset,
                xmax=max(self.channels),
                lw=3,
            )
            ax_res.set_ylabel("Residual")
            ax_res.set_xlim([min(self.channels) - x_offset, max(self.channels)])
            ax_res.set_xticks([])
        else:
            if ax_fit is None:
                fig = plt.figure(constrained_layout=False, figsize=(12, 8))
                ax_fit = fig.add_subplot()
    
        ax_fit.set_title(rf"Reduced $\chi^2$ = {self.fit.redchi:.4f}")
        ax_fit.errorbar(
            self.mean_vals,
            self.erg,
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
            self.channels,
            self.predicted,
            ls="--",
            lw=3,
            color="green",
            label=f"Predicted: {equation}",
        )
        ax_fit.set_xlim([min(self.channels) - x_offset, max(self.channels)])
        ax_fit.set_xlabel("Channels")
        ax_fit.set_ylabel(f"Energy [{self.e_units}]")
        ax_fit.legend()
    
        return ax_fit
    
def smart_calibration(
    channels: list,
    energies: list,
    min_points: int = 3,
    require_positive_slope: bool = True,
    max_combinations: int = 100_000):
    """
    Automatically find the best linear energy calibration E = a·ch + b by
    exhaustively searching over all monotone-preserving subsets of the larger
    input list paired against the full smaller list.

    The two input lists can have different lengths. The algorithm fixes the
    shorter list and iterates over all C(N, k) size-k subsets of the longer
    list (where k = min(len(channels), len(energies))), scoring each candidate
    pairing by its coefficient of determination R².

    Parameters
    ----------
    channels : list
        Detected peak positions in channel number. Need not be sorted.
    energies : list
        Known reference energies in keV. Need not be sorted.
    min_points : int, optional
        Minimum number of channel–energy pairs required (default 3).
    require_positive_slope : bool, optional
        Reject fits with a non-positive slope, which are physically
        meaningless for standard detector geometries (default True).
    max_combinations : int, optional
        Maximum number of subset combinations to search before raising
        an error. The default is 100_000.

    Returns
    -------
    best
        dictionary of best fitting values including slope (c1), intercept (c0),
        r2, channels and energies.

    Raises
    ------
    ValueError
        If fewer than `min_points` channels or energies are supplied, or
        if no valid fit (positive slope) is found.

    Examples
    --------
    >>> channels = [237, 1764, 351, 609, 1120, 1460]
    >>> energies = [121.8, 344.3, 778.9, 964.1, 1112.1, 1408.0]
    >>> result = smart_calibration(channels, energies)
    >>> print(result)
    """
    channels = np.sort(np.asarray(channels, dtype=float))
    energies = np.sort(np.asarray(energies, dtype=float))

    n_ch = len(channels)
    n_en = len(energies)
    k = min(n_ch, n_en)

    if n_ch < min_points:
        raise ValueError(f"Need at least {min_points} channels; got {n_ch}.")
    if n_en < min_points:
        raise ValueError(f"Need at least {min_points} energies; got {n_en}.")
    n_combos = comb(max(n_ch, n_en), k)
    if n_combos > max_combinations:
        raise ValueError(
            f"Too many combinations to search ({n_combos:,}). "
            f"Reduce the number of input points or increase max_combinations."
        )

    best_r2 = -np.inf
    best: dict = {}

    def _evaluate(x_arr, y_arr):
        """Fit and score a single channel–energy pairing."""
        nonlocal best_r2, best
        slope, intercept, r, *_ = linregress(x_arr, y_arr)
        if require_positive_slope and slope <= 0:
            return
        r2 = r ** 2
        if r2 > best_r2:
            best_r2 = r2
            best = {
                "c1": slope,
                "c0": intercept,
                "r2": r2,
                "channels": x_arr.copy(),
                "energies": y_arr.copy(),
            }

    if n_ch <= n_en:
        # Channels are the fixed axis; choose k energies from the pool.
        for idx in combinations(range(n_en), k):
            _evaluate(channels, energies[list(idx)])
    else:
        # Energies are the fixed axis; choose k channels from the pool.
        for idx in combinations(range(n_ch), k):
            _evaluate(channels[list(idx)], energies)

    if not best:
        raise ValueError(
            "No valid calibration found. All candidate fits had a "
            "non-positive slope. Check that channels and energies are "
            "in ascending order or set require_positive_slope=False."
        )
    return best
    
    
    
    
    
    
    
    