"""
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Spectrum:
    def __init__(self, counts=None, channels=None, energies=None, e_units=None):
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
        if counts is None:
            print("ERROR: Must specify counts")
        if channels is None:
            channels = np.arange(0, len(counts), 1)
        if energies is not None:
            self.energies = np.asarray(energies, dtype=float)
            if e_units is None:
                self.x_units = "Energy"
            else:
                self.x_units = f"Energy [{e_units}]"

        else:
            self.energies = energies
            self.x_units = "Channels"

        self.counts = np.asarray(counts, dtype=float)
        self.channels = np.asarray(channels, dtype=int)

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
        mav.fillna(0, inplace=True)
        return np.array(mav)

    def rebin(self):
        """
        Rebins data by adding two adjacent bins at a time.

        Returns
        -------
        numpy array
            Rebinned counts
        If energies are passed, returns both rebinned counts and average
        energies

        """
        arr_cts = self.counts
        if arr_cts.shape[0] % 2 != 0:
            arr_cts = arr_cts[:-1]
        y0 = arr_cts[::2]
        y1 = arr_cts[1::2]
        y = y0 + y1

        if self.energies is None:
            return y

        erg = self.energies
        if erg.shape[0] % 2 != 0:
            erg = erg[:-1]
        en0 = erg[::2]
        en1 = erg[1::2]
        en = (en0 + en1) / 2
        return en, y

    def plot(self, scale="log"):
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
        x = self.channels
        y = self.counts
        integral = round(y.sum())
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")
        plt.figure()
        plt.fill_between(x, 0, y, alpha=0.5, color="C0", step="pre")
        plt.plot(x, y, drawstyle="steps")
        plt.yscale(scale)
        plt.title(f"Raw Spectrum. Integral = {integral}")
        plt.xlabel("Channels")
        # plt.ylabel("a.u")
        plt.style.use("default")

        if self.energies is not None:
            x = self.energies
            y = self.counts
            plt.rc("font", size=14)
            plt.style.use("seaborn-darkgrid")
            plt.figure()
            plt.fill_between(x, 0, y, alpha=0.5, color="C1", step="pre")
            plt.plot(x, y, color="C1", drawstyle="steps")
            plt.yscale(scale)
            plt.title(f"Raw Spectrum. Integral = {integral}")
            plt.xlabel(self.x_units)
            # plt.ylabel("a.u")
            plt.style.use("default")
