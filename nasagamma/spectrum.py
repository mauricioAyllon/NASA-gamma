"""
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Spectrum:
    def __init__(
        self,
        counts=None,
        channels=None,
        energies=None,
        e_units=None,
        livetime=None,
        cps=False,
    ):
        """
        Initialize the spectrum.

        Parameters
        ----------
        counts : numpy array, pandas series, or list.
            counts per bin or count rate. This is the only
            required input parameter.
        channels : numpy array, pandas series, or list. Optional
            array of bin edges. If None, assume based on counts.
            The default is None.
        energies : numpy array, pandas series, or list. Optional
            energy values. The default is None.
        e_units : string, optional
            string of energy units e.g. "MeV". The default is None.

        Returns
        -------
        None.

        """
        self.e_units = e_units
        if counts is None:
            print("ERROR: Must specify counts")
        if channels is None:
            channels = np.arange(0, len(counts), 1)
        if energies is not None:
            self.energies = np.asarray(energies, dtype=float)
            self.x = self.energies
            if self.e_units is None:
                self.x_units = "Energy"
            else:
                self.x_units = f"Energy ({e_units})"

        else:
            self.energies = energies
            self.x = channels
            self.x_units = "Channels"

        self.counts = np.asarray(counts, dtype=float)
        self.channels = np.asarray(channels, dtype=int)
        self.livetime = livetime
        self.cps = cps
        if cps:
            self.y_label = "CPS"
        else:
            self.y_label = "Cts"
        self.plot_label = None

    def smooth(self, num=4):
        """
        Parameters
        ----------
        num : integer, optional
            number of data points for averaging. The default is 4.

        Returns
        -------
        numpy array
            moving average of counts.

        """
        df = pd.DataFrame(data=self.counts, columns=["cts"])
        mav = df.cts.rolling(window=num, center=True).mean()
        mav.fillna(method="bfill", inplace=True)
        mav.fillna(method="pad", inplace=True)
        mav.fillna(0, inplace=True)
        counts_mav = np.array(mav)
        counts_mav_scaled = counts_mav / counts_mav.sum() * self.counts.sum()
        self.__init__(
            counts=counts_mav_scaled,
            energies=self.energies,
            e_units=self.e_units,
            livetime=self.livetime,
        )

    def rebin(self, by=2):
        """
        Rebins data by adding 'by' adjacent bins at a time.
        """
        new_size = int(self.counts.shape[0] / by)
        new_cts = self.counts.reshape((new_size, -1)).sum(axis=1)
        self.counts = new_cts
        self.channels = np.arange(0, len(new_cts), 1)

        if self.energies is not None:
            new_erg = self.energies.reshape((new_size, -1)).mean(axis=1)
            self.energies = new_erg
            self.x = new_erg
        else:
            self.x = self.channels

    def gain_shift(self, by=0, energy=False):
        """
        Slide spectrum left or right by 'by' number of channels or energy values.
        If positive shift, replace low energy values by zeroes. If negative shift,
        replace high energy values by the value in the last bin.

        Parameters
        ----------
        by : integer, optional
            Number of channels or energy values to shift the spectrum by. The default is 0.
        energy : double, optional
            Set to True if shifting by energy values. The default is False.

        Returns
        -------
        None.

        """
        if energy and self.energies is not None:
            cal = np.diff(self.energies)[0]
            by = round(by / cal)
        by = int(by)
        if by > 0:
            # positive roll
            # replace rolled low energy counts with zeros
            self.counts = np.roll(self.counts, shift=by)
            self.counts[0:by] = 0
        elif by < 0:
            # negative roll
            # replace rolled high energy counts with the last high energy value
            self.counts = np.roll(self.counts, shift=by)
            self.counts[by:] = self.counts[by - 1]
        else:
            print(f"Cannot roll by {by} units")

    def replace_neg_vals(self):
        """
        Replaces negative values in spectrum with 1/10th of the minimum

        Returns
        -------
        None.

        """
        # find min greater than zero
        y0_min = np.amin(self.counts[self.counts > 0.0])
        # replace negative values and zeros by 1/10th of the minimum
        self.counts[self.counts < 0.0] = y0_min * 1e-1

    def plot(self, fig=None, ax=None, scale="log"):
        """
        Plot spectrum object using channels and energies (if not None)

        Parameters
        ----------
        scale : string, optional
            DESCRIPTION. Either 'linear' or 'log'. The default is 'log'.

        Returns
        -------
        None.

        """
        plt.rc("font", size=14)
        plt.style.use("seaborn-v0_8-darkgrid")

        if fig is None:
            fig = plt.figure(figsize=(10, 6))
        if ax is None:
            ax = fig.add_subplot()
        fig.patch.set_alpha(0.3)

        integral = round(self.counts.sum())
        if self.plot_label is None:
            if self.livetime is None:
                lt = "Livetime = N/A"
            else:
                lt = f"Livetime = {self.livetime:.3E} s"
            self.plot_label = f"Total counts = {integral:.3E}\n{lt}"

        ax.fill_between(self.x, 0, self.counts, alpha=0.2, color="C1", step="pre")
        ax.plot(
            self.x, self.counts, drawstyle="steps", alpha=0.7, label=self.plot_label
        )
        ax.set_yscale(scale)
        ax.set_xlabel(self.x_units, fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)
        ax.legend()
        plt.show()
