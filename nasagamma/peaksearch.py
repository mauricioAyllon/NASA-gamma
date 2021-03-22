"""
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from . import spectrum as sp


def gaussian_second_derivate(x, mean, sigma):
    """Return the second Gaussian derivate (unnormalized).

    This is also called the Marr wavelet or Mexican hat [https://en.wikipedia.org/wiki/Mexican_hat_wavelet].

    Parameters
    ----------
    x : numpy array.
        x-values.
    mean : float or int.
        mean of Gaussian distribution.
    sigma : float or int.
        standard deviation.

    Returns
    -------
    numpy array
        second derivative of the Gaussian distribution

    """

    z = (x - mean) / sigma
    return (1 - z ** 2) * np.exp(-(z ** 2) / 2.0)


class PeakSearch:
    def __init__(self, spectrum, ref_x, ref_fwhm, fwhm_at_0=1.0, min_snr=2):
        """
        Find peaks in a Spectrum object and decompose specrum into components
        using a Gaussian kernel deconvolution technique. Most of this
        functionality was adapted from https://github.com/lbl-anp/becquerel

        Parameters
        ----------
        spectrum : Spectrum object.
            previously initialized spectrum object.
        ref_x : int
            reference x-value (in channels) corresponding to ref_fwhm.
        ref_fwhm : int or float.
            fwhm value (in channels) corresponding to ref_x.
        fwhm_at_0 : int or float, optional
            fwhm value at channel = 0. The default is 1.0.
        min_snr : int or float, optional
            minimum SNR to look for releant peaks. The default is 2.

        Raises
        ------
        Exception
            'spectrum' must be a Spectrum object.

        Returns
        -------
        None.

        """
        if not isinstance(spectrum, sp.Spectrum):
            raise Exception("'spectrum' must be a Spectrum object")
        self.ref_x = ref_x
        self.ref_fwhm = ref_fwhm
        self.fwhm_at_0 = fwhm_at_0
        self.spectrum = spectrum
        self.min_snr = min_snr
        self.kern_mat = None
        self.kern_mat_pos = None
        self.kern_mat_neg = None
        self.snr = []
        self.peak_plus_bkg = []
        self.bkg = []
        self.signal = []
        self.noise = []
        self.peaks_idx = []
        self.fwhm_guess = []
        self.calculate()

    def fwhm(self, x):
        """
        Calculate the expected FWHM at the given x value

        Parameters
        ----------
        x : numpy array
            x-values.

        Returns
        -------
        numpy array.
            expected FWHM values.

        """
        # f(x) = k * sqrt(x) + b
        # b = f(0)
        # k = f1/sqrt(x1)
        f0 = self.fwhm_at_0
        f1 = self.ref_fwhm
        x1 = self.ref_x
        # fwhm_sqr = np.sqrt(f0**2 + (f1**2 - f0**2) * (x / x1)**2)
        fwhm_sqr = (f1 / np.sqrt(x1)) * np.sqrt(x) + f0
        return fwhm_sqr

    def kernel(self, x, edges):
        """Generate the kernel for the given x value."""

        sigma = self.fwhm(x) / 2.355
        kernel = gaussian_second_derivate(edges, x, sigma)
        # normalize so that the peaks fit the data
        mask = kernel > 0
        if mask.sum():
            kernel /= kernel[mask].sum()
        return kernel

    def kernel_matrix(self, edges):
        """Build a matrix of the kernel evaluated at each x value."""
        n_channels = len(edges)
        kern = np.zeros((n_channels, n_channels))
        for i, x in enumerate(edges):
            kern[:, i] = self.kernel(x, edges)
        return kern

    def convolve(self, edges, data):
        """Convolve kernel with the data."""

        self.kern_mat = self.kernel_matrix(edges)
        self.kern_mat_pos = +1 * self.kern_mat.clip(0, np.inf)
        self.kern_mat_neg = -1 * self.kern_mat.clip(-np.inf, 0)

        self.peak_plus_bkg = self.kern_mat_pos @ data
        self.bkg = self.kern_mat_neg @ data
        self.signal = self.kern_mat @ data
        self.noise = np.sqrt((self.kern_mat ** 2) @ data)

        self.snr = np.zeros_like(self.signal)
        mask = self.noise > 0
        self.snr[mask] = self.signal[mask] / self.noise[mask]
        self.snr = self.snr.clip(0)

    def calculate(self):
        """Calculate the convolution of the spectrum with the kernel."""

        self.convolve(self.spectrum.channels, self.spectrum.counts)

        # find peak indices
        peaks_idx = find_peaks(self.snr, height=self.min_snr)[0]

        self.fwhm_guess = self.fwhm(peaks_idx)
        self.peaks_idx = peaks_idx

    def plot_kernel(self, ax=None):
        """Plot the 3D matrix of kernels evaluated across the x values."""
        # edges = self.spectrum.channels
        if self.kern_mat is None:
            print("The kernel has not been calculated yet, nothing to plot")
            return

        # at the edges, the normalization can create single values that are close to 1
        # we want to skip these for the plotting, so we pick the 99% percentile
        tmp = np.sort(np.abs(self.kern_mat).flatten())
        kern_max = tmp[int(len(tmp) * 0.99)]

        ax_orig = ax
        if ax is None:
            _, ax = plt.subplots()

        img = ax.imshow(
            self.kern_mat.T,
            cmap=plt.get_cmap("bwr"),
            vmin=-kern_max,
            vmax=kern_max,
            origin="lower",
        )
        plt.colorbar(img)
        ax.set_xlabel("Input x")
        ax.set_ylabel("Output x")
        ax.set_aspect("equal")
        ax.set_title("Kernel Matrix")
        if ax_orig is None:
            plt.show()

    def plot_peaks(self, scale="log", snrs="on", ax=None):
        """
        Plot spectrum and their found peak positions.

        Parameters
        ----------
        scale : string, optional
            Either 'log' or 'linear'. The default is 'log'.

        Returns
        -------
        None.

        """
        if self.spectrum.energies is None:
            # x = self.spectrum.channels[:-1]
            x = self.spectrum.channels
        else:
            x = self.spectrum.energies

        ax_orig = ax
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if snrs == "on":
            ax.plot(x, self.snr, label="SNR all")
        # ax.plot(x, self.spectrum.counts, label="Raw spectrum")
        self.spectrum.plot(ax=ax)
        ax.set_yscale(scale)
        for xc in self.peaks_idx:
            if self.spectrum.energies is None:
                x0 = xc
            else:
                x0 = self.spectrum.energies[xc]
            ax.axvline(x=x0, color="red", linestyle="-", alpha=0.2)
        ax.legend(loc=1)
        ax.set_title(f"SNR > {self.min_snr}")
        ax.set_xlabel(self.spectrum.x_units)
        ax.set_ylabel("Cts")
        if ax_orig is None:
            plt.show()

    def plot_components(self, scale="log", ax=None):
        """
        Plot spectrum components after decomposition.

        Parameters
        ----------
        yscale : string, optional
            Either 'log' or 'linear'. The default is 'log'.

        Returns
        -------
        None.

        """
        if self.spectrum.energies is None:
            x = self.spectrum.channels
        else:
            x = self.spectrum.energies

        ax_orig = ax
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, self.spectrum.counts, label="Raw spectrum")
        ax.plot(x, self.peak_plus_bkg.clip(1e-1), label="Peaks+Continuum")
        ax.plot(x, self.bkg.clip(1e-1), label="Continuum")
        ax.plot(x, self.signal.clip(1e-1), label="Peaks")
        ax.plot(x, self.noise.clip(1e-1), label="noise")
        ax.set_yscale(scale)

        ax.set_ylim(3e-1)
        ax.set_xlabel(self.spectrum.x_units)
        ax.set_ylabel("Cts")
        ax.legend(loc=1)
        if ax_orig is None:
            plt.show()
