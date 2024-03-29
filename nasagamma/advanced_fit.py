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
    def __init__(self, spectrum, x1, x2):
        if not isinstance(spectrum, sp.Spectrum):
            raise Exception("spectrum must be a Spectrum object")
        self.spect = spectrum
        self.x1 = x1
        self.x2 = x2
        self.ch = self.spect.channels
        self.prange = 0
        self.B = 0
        self.A = 0
        self.sigB = 0
        self.sigA = 0
        self.y_eqn = 0
        if self.spect.energies is None:
            # print("Working with channel numbers")
            self.ch1 = int(x1)
            self.ch2 = int(x2)
            # self.chrange = [round(self.xrange[0]), round(self.xrange[1])]
        else:
            # print("Working with energy values")
            erg = self.spect.energies
            self.ch1 = int(self.spect.channels[erg > x1][0])
            self.ch2 = int(self.spect.channels[erg > x2][0])
            self.e1 = erg[self.ch1]
            self.e2 = erg[self.ch2]
        self.y1 = self.spect.counts[self.ch1]
        self.y2 = self.spect.counts[self.ch2]
        self.xr = self.spect.channels[self.ch1 : self.ch2 + 1]
        self.yr = self.spect.counts[self.ch1 : self.ch2 + 1]

    def calculate_peak_area(self):
        # linear equation for background
        self.y_eqn = (self.y2 - self.y1) / (self.x2 - self.x1) * (
            self.xr - self.x1
        ) + self.y1
        self.A = self.yr.sum() - self.y_eqn.sum()
        self.B = self.y_eqn.sum()
        # print(f"Peak area based on linear background = {self.A}")
        # Errors
        sigAB = np.sqrt(self.yr.sum())
        self.sigB = np.sqrt(self.y_eqn.sum())
        self.sigA = np.sqrt(
            sigAB**2 + self.sigB**2
        )  # should there be a factor of 2?
        # print(f"Sigma area based on linear background = {self.sigA}")

    def plot(self, ax=None, areas=False):
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            fig.patch.set_alpha(0.3)  # set background transparent
            ax = fig.add_subplot()

        if areas is False:
            line = ax.plot(
                self.spect.x[self.ch1 : self.ch2],
                self.spect.counts[self.ch1 : self.ch2],
                drawstyle="steps",
            )
        else:
            if self.spect.energies is None:
                xarea = self.xr
            else:
                xarea = self.spect.energies[self.pchrange[0] : self.pchrange[1] + 1]
            line = ax.plot(self.x, self.y, drawstyle="steps")
            ax.plot(xarea, self.y_eqn, color="C1")

            ax.fill_between(
                x=xarea,
                y1=0,
                y2=self.y_eqn,
                step="pre",
                alpha=0.2,
                color="r",
                label=f"B = {round(self.B,3)}",
            )
            ax.fill_between(
                x=xarea,
                y1=self.y_eqn,
                y2=self.yr,
                step="pre",
                alpha=0.2,
                color="g",
                label=f"A = {round(self.A,3)}",
            )
            ax.vlines(
                x=self.prange[0],
                ymin=0,
                ymax=self.y1,
                linestyle="dotted",
                color="r",
                label=f"$x_1$ = {self.prange[0]}",
            )
            ax.vlines(
                x=self.prange[1],
                ymin=0,
                ymax=self.y2,
                linestyle="--",
                color="r",
                label=f"$x_2$ = {self.prange[1]}",
            )
            ax.legend(loc="upper right")
        mplcursors.cursor(line, hover=True)
        ax.set_yscale("linear")
        ax.set_xlabel(self.spect.x_units)
        ax.set_ylabel(self.spect.y_label)
        plt.show()
