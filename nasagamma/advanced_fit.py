"""
Advanced fitting classes and functions
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplcursors
from nasagamma import file_reader
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf


class PeakAreaLinearBkg:
    def __init__(self, spectrum):
        if not isinstance(spectrum, sp.Spectrum):
            raise Exception("spectrum must be a Spectrum object")
        self.spect = spectrum
        # results — populated by calculate_peak_area
        self.A = 0
        self.B = 0
        self.sigA = 0
        self.sigB = 0
        self.y_eqn = 0
        self.prange = None
        self.pchrange = None
        self.xr = None
        self.yr = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _x_to_ch(self, x):
        """Convert a single x value (channel or energy) to a channel index."""
        if self.spect.energies is None:
            return int(x)
        else:
            return int(self.spect.channels[self.spect.energies >= x][0])

    def _collect_bkg_points(self, x_input):
        """
        Given a scalar or 2-element list, return all x/y points for background fitting.

        Parameters
        ----------
        x_input : scalar or 2-element list
            Single point or range [a, b].

        Returns
        -------
        x_bkg : numpy array
            x-axis values of background points (channels or energies).
        y_bkg : numpy array
            Corresponding counts.
        """
        if np.isscalar(x_input):
            ch = self._x_to_ch(x_input)
            x_val = x_input if self.spect.energies is not None else float(ch)
            x_bkg = np.array([x_val])
            y_bkg = np.array([self.spect.counts[ch]])
        else:
            ch_a = self._x_to_ch(x_input[0])
            ch_b = self._x_to_ch(x_input[1])
            if self.spect.energies is None:
                x_bkg = self.spect.channels[ch_a : ch_b + 1].astype(float)
            else:
                x_bkg = self.spect.energies[ch_a : ch_b + 1]
            y_bkg = self.spect.counts[ch_a : ch_b + 1]
        return x_bkg, y_bkg

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate_peak_area(self, x1, x2):
        """
        Calculate the net peak area above a linear background.

        Parameters
        ----------
        x1 : scalar or 2-element list
            Left background region. Scalar = single point; list = range [a, b].
        x2 : scalar or 2-element list
            Right background region. Scalar = single point; list = range [a, b].
        """
        # --- collect background points from both sides ---
        x_bkg_l, y_bkg_l = self._collect_bkg_points(x1)
        x_bkg_r, y_bkg_r = self._collect_bkg_points(x2)

        # derive the four edges explicitly:
        #   outer_l  = leftmost of x1  (full plot left edge)
        #   inner_l  = rightmost of x1 (left boundary of peak region)
        #   inner_r  = leftmost of x2  (right boundary of peak region)
        #   outer_r  = rightmost of x2 (full plot right edge)
        ch_outer_l = self._x_to_ch(x1[0] if not np.isscalar(x1) else x1)
        ch_inner_l = self._x_to_ch(x1[1] if not np.isscalar(x1) else x1)
        ch_inner_r = self._x_to_ch(x2[0] if not np.isscalar(x2) else x2)
        ch_outer_r = self._x_to_ch(x2[1] if not np.isscalar(x2) else x2)

        # combine all background points and fit a line
        x_bkg_all = np.concatenate([x_bkg_l, x_bkg_r])
        y_bkg_all = np.concatenate([y_bkg_l, y_bkg_r])
        slope, intercept = np.polyfit(x_bkg_all, y_bkg_all, 1)
        self._slope = slope
        self._intercept = intercept

        # --- peak region: between the two inner edges ---
        self.pchrange = [ch_inner_l, ch_inner_r]
        if self.spect.energies is None:
            self.prange = [float(ch_inner_l), float(ch_inner_r)]
            self.xr = self.spect.channels[ch_inner_l : ch_inner_r + 1].astype(float)
            x_full_range = self.spect.channels[ch_outer_l : ch_outer_r + 1].astype(float)
        else:
            self.prange = [
                self.spect.energies[ch_inner_l],
                self.spect.energies[ch_inner_r],
            ]
            self.xr = self.spect.energies[ch_inner_l : ch_inner_r + 1]
            x_full_range = self.spect.energies[ch_outer_l : ch_outer_r + 1]
        self.yr = self.spect.counts[ch_inner_l : ch_inner_r + 1]

        # evaluate the fitted line over the full outer range (for plotting the line)
        self.y_eqn = slope * x_full_range + intercept

        # evaluate over the peak region (for area calculation and fill_between)
        self.y_eqn_peak = slope * self.xr + intercept

        # --- areas ---
        self.A = self.yr.sum() - self.y_eqn_peak.sum()
        self.B = self.y_eqn_peak.sum()

        # --- errors (Poisson) ---
        sigAB = np.sqrt(self.yr.sum())
        self.sigB = np.sqrt(np.abs(self.y_eqn_peak.sum()))
        self.sigA = np.sqrt(sigAB**2 + self.sigB**2)

        # store outer edges, full x range, and x-axis edge values for plotting
        self._ch_outer_l = ch_outer_l
        self._ch_outer_r = ch_outer_r
        self._x_full_range = x_full_range
        if self.spect.energies is None:
            self._x_outer_l = float(ch_outer_l)
            self._x_inner_l = float(ch_inner_l)
            self._x_inner_r = float(ch_inner_r)
            self._x_outer_r = float(ch_outer_r)
        else:
            self._x_outer_l = self.spect.energies[ch_outer_l]
            self._x_inner_l = self.spect.energies[ch_inner_l]
            self._x_inner_r = self.spect.energies[ch_inner_r]
            self._x_outer_r = self.spect.energies[ch_outer_r]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, ax=None, areas=False):
        plt.rc("font", size=14)
        plt.style.use("seaborn-v0_8-darkgrid")
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            fig.patch.set_alpha(0.3)
            ax = fig.add_subplot()

        # full x/y range for display
        if self.prange is not None:
            ch_l = self._ch_outer_l
            ch_r = self._ch_outer_r
        else:
            raise RuntimeError("Call calculate_peak_area() before plot().")

        if self.spect.energies is None:
            x_full = self.spect.channels[ch_l : ch_r + 1].astype(float)
        else:
            x_full = self.spect.energies[ch_l : ch_r + 1]
        y_full = self.spect.counts[ch_l : ch_r + 1]

        line = ax.plot(x_full, y_full, drawstyle="steps")

        if areas:
            ax.plot(self._x_full_range, self.y_eqn, color="C1", label="Linear background fit")
            ax.fill_between(
                x=self._x_full_range, y1=0, y2=self.y_eqn,
                step="pre", alpha=0.2, color="r",
                label=f"B = {round(self.B, 3)}",
            )
            ax.fill_between(
                x=self.xr, y1=self.y_eqn_peak, y2=self.yr,
                step="pre", alpha=0.2, color="g",
                label=f"A = {round(self.A, 3)}",
            )
            # draw boundary lines — stop at the background line height at each x
            y_at_outer_l = self._slope * self._x_outer_l + self._intercept
            y_at_inner_l = self._slope * self._x_inner_l + self._intercept
            y_at_inner_r = self._slope * self._x_inner_r + self._intercept
            y_at_outer_r = self._slope * self._x_outer_r + self._intercept

            if self._x_outer_l != self._x_inner_l:
                ax.vlines(self._x_outer_l, 0, y_at_outer_l, linestyle="dotted",
                          color="gray", lw=2,
                          label=f"x1 range: [{round(self._x_outer_l,3)}, {round(self._x_inner_l,3)}]")
                ax.vlines(self._x_inner_l, 0, y_at_inner_l, linestyle="dotted",
                          color="gray", lw=2)
            else:
                ax.vlines(self._x_inner_l, 0, y_at_inner_l, linestyle="dotted",
                          color="gray", lw=2,
                          label=f"x1 = {round(self._x_inner_l, 3)}")
            if self._x_inner_r != self._x_outer_r:
                ax.vlines(self._x_inner_r, 0, y_at_inner_r, linestyle="dotted",
                          color="C1", lw=2,
                          label=f"x2 range: [{round(self._x_inner_r,3)}, {round(self._x_outer_r,3)}]")
                ax.vlines(self._x_outer_r, 0, y_at_outer_r, linestyle="dotted",
                          color="C1", lw=2)
            else:
                ax.vlines(self._x_inner_r, 0, y_at_inner_r, linestyle="dotted",
                          color="C1", lw=2,
                          label=f"x2 = {round(self._x_inner_r, 3)}")
            ax.legend(loc="upper right")

        mplcursors.cursor(line, hover=True)
        ax.set_yscale("linear")
        ax.set_xlabel(self.spect.x_units)
        ax.set_ylabel(self.spect.y_label)
        plt.show()