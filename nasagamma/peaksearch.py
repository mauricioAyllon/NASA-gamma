"""
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from . import spectrum as sp


def gaussian(x, mean, sigma):
    """
    Gaussian function.

    Parameters
    ----------
    x : numpy array.
        x-values.
    mean : float or int.
        mean of distribution.
    sigma : float or int.
        standard deviation.

    Returns
    -------
    numpy array.
        Gaussian distribution.
    """
    z = (x - mean) / sigma
    return np.exp(-(z**2) / 2.0)


def gaussian_derivative(x, mean, sigma):
    """
    First derivative of a Gaussian.

    Parameters
    ----------
    x : numpy array.
        x-values.
    mean : float or int.
        mean of distribution.
    sigma : float or int.
        standard deviation.

    Returns
    -------
    numpy array
        first derivaive of a Gaussian.

    """
    z = x - mean
    return -1 * z * gaussian(x, mean, sigma)


class PeakSearch:
    def __init__(
        self,
        spectrum,
        ref_x,
        ref_fwhm,
        fwhm_at_0=1.0,
        min_snr=2,
        xrange=None,
        method="km",
    ):
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
        xrange : list or numpy array of shape (2,), optional
            specific x range for peak searching. The default is None.
        method : string, optional
            peak searching method including kernel method (km) and scipy
            peak finding method (scipy). The default is km.

        Raises
        ------
        Exception
            'spectrum' must be a Spectrum object.

        Returns
        -------
        None.

        """
        if not isinstance(spectrum, sp.Spectrum):
            raise Exception("spectrum must be a Spectrum object")
        if xrange is None:
            self.channel_idx = spectrum.channels
            self.xrange = xrange
        elif len(xrange) == 2 and spectrum.energies is not None:
            ixe = (spectrum.energies >= xrange[0]) & (spectrum.energies <= xrange[1])
            erange = spectrum.channels[ixe]
            self.xrange = [erange[0], erange[-1]]
        elif len(xrange) == 2 and spectrum.energies is None:
            self.xrange = xrange
        else:
            print("ERROR: check that the length of xrange is 2")

        self.ref_x = ref_x
        self.ref_fwhm = ref_fwhm
        self.fwhm_at_0 = fwhm_at_0
        self.spectrum = spectrum
        self.min_snr = min_snr
        self.method = method
        self.snr = []
        self.peak_plus_bkg = []
        self.bkg = []
        self.signal = []
        self.noise = []
        self.peaks_idx = []
        self.fwhm_guess = []
        if method == "km":
            self.calculate_km()
        if method == "scipy":
            self.calculate_scipy()

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
        fwhm1 = self.fwhm(x)
        sigma = fwhm1 / 2.355
        g1_x0 = gaussian_derivative(edges[:-1], x, sigma)
        g1_x1 = gaussian_derivative(edges[1:], x, sigma)
        kernel = g1_x0 - g1_x1
        return kernel

    def kernel_matrix(self, edges):
        """Build a matrix of the kernel evaluated at each x value."""
        n_channels = len(edges) - 1
        kern = np.zeros((n_channels, n_channels))
        for i, x in enumerate(edges[:-1]):
            kern[:, i] = self.kernel(x, edges)
        kern_pos = +1 * kern.clip(0, np.inf)
        kern_neg = -1 * kern.clip(-np.inf, 0)
        # normalize negative part to be equal to the positive part
        kern_neg *= kern_pos.sum(axis=0) / kern_neg.sum(axis=0)
        kmat = kern_pos - kern_neg
        return kmat

    def convolve(self, edges, data):
        """Convolve kernel with the data."""
        kern_mat = self.kernel_matrix(edges)
        kern_mat_pos = +1 * kern_mat.clip(0, np.inf)
        kern_mat_neg = -1 * kern_mat.clip(-np.inf, 0)
        peak_plus_bkg = kern_mat_pos @ data
        bkg = kern_mat_neg @ data
        signal = kern_mat @ data
        noise = (kern_mat**2) @ data
        # print("other")
        # noise = np.array([np.sqrt(x) for x in noise])
        noise = np.sqrt(noise)
        snr = np.zeros_like(signal)
        snr[noise > 0] = signal[noise > 0] / noise[noise > 0]
        return peak_plus_bkg, bkg, signal, noise, snr

    def calculate_km(self):
        """Calculate the convolution of the spectrum with the kernel."""

        if self.spectrum.cps and self.spectrum.livetime is not None:
            spect_cts = self.spectrum.counts * self.spectrum.livetime
        else:
            spect_cts = self.spectrum.counts

        snr = np.zeros(len(self.spectrum.counts))  # do we need this?
        if self.xrange is None:
            self.edg = np.append(self.spectrum.channels, self.spectrum.channels[-1] + 1)
            # calculate the convolution
            peak_plus_bkg, bkg, signal, noise, snr = self.convolve(self.edg, spect_cts)
        else:
            x0 = self.xrange[0]
            x1 = self.xrange[1]
            self.channel_idx = (self.spectrum.channels >= x0) & (
                self.spectrum.channels <= x1
            )
            new_ch = self.spectrum.channels[self.channel_idx]
            new_cts = spect_cts[self.channel_idx]
            self.edg = np.append(new_ch, new_ch[-1] + 1)
            peak_plus_bkg, bkg, signal, noise, snr = self.convolve(self.edg, new_cts)

        clipped_snr = snr.clip(0)

        # find peak indices
        peaks_idx = find_peaks(clipped_snr, height=self.min_snr)[0]

        # remove first and last index (not real peaks)
        # peaks_idx = peaks_idx[1:-1]

        self.fwhm_guess = self.fwhm(peaks_idx)
        self.peak_plus_bkg = peak_plus_bkg
        self.bkg = bkg
        self.signal = signal
        self.noise = noise
        self.snr = clipped_snr
        if self.xrange == None:
            self.peaks_idx = peaks_idx
        else:
            self.peaks_idx = new_ch[peaks_idx]
        # self.reset()

    def calculate_scipy(self):
        if self.xrange is None:
            peaks_idx, _ = find_peaks(
                self.spectrum.counts,
                prominence=self.min_snr,
                width=self.fwhm(self.spectrum.counts),
            )
            self.peaks_idx = peaks_idx
        else:
            x0 = self.xrange[0]
            x1 = self.xrange[1]
            self.channel_idx = (self.spectrum.channels >= x0) & (
                self.spectrum.channels <= x1
            )
            new_ch = self.spectrum.channels[self.channel_idx]
            new_cts = self.spectrum.counts[self.channel_idx]
            new_widths = self.fwhm(new_cts)
            peaks_idx, _ = find_peaks(
                new_cts, prominence=self.min_snr, width=new_widths
            )
            self.peaks_idx = new_ch[peaks_idx]

        self.fwhm_guess = self.fwhm(self.peaks_idx)
        self.peak_plus_bkg = None
        self.bkg = None
        self.signal = None
        self.noise = None
        self.snr = None

    def plot_kernel(self):
        """Plot the 3D matrix of kernels evaluated across the x values."""
        # edges = self.spectrum.channels
        edges = self.edg
        n_channels = len(edges) - 1
        kern_mat = self.kernel_matrix(edges)
        kern_min = kern_mat.min()
        kern_max = kern_mat.max()
        kern_min = min(kern_min, -1 * kern_max)
        kern_max = max(kern_max, -1 * kern_min)

        plt.figure()
        plt.imshow(
            kern_mat.T[::-1, :],
            cmap=plt.get_cmap("bwr"),
            vmin=kern_min,
            vmax=kern_max,
            extent=[n_channels, 0, 0, n_channels],
        )
        plt.colorbar()
        plt.xlabel("Input x")
        plt.ylabel("Output x")
        plt.gca().set_aspect("equal")
        plt.title("Kernel Matrix")

    def plot_peaks(self, yscale="log", snrs="on", fig=None, ax=None):
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
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")
        if self.spectrum.energies is None:
            # x = self.spectrum.channels[:-1]
            x = self.spectrum.channels[self.channel_idx]
        else:
            x = self.spectrum.energies[self.channel_idx]
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
        if ax is None:
            ax = fig.add_subplot()

        self.spectrum.plot(fig=fig, ax=ax)
        if snrs == "on" and self.method == "km":
            ax.plot(x, self.snr, label="SNR")
        # ax.plot(x, self.spectrum.counts, label="Raw spectrum")
        if yscale == "log":
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
        for xc in self.peaks_idx:
            if self.spectrum.energies is None:
                x0 = xc
            else:
                x0 = self.spectrum.energies[xc]
            ax.axvline(x=x0, color="red", linestyle="--", alpha=0.2)
        ax.legend(loc=1)
        ax.set_title(f"SNR > {self.min_snr}")
        # ax.set_ylim(1e-1)
        ax.set_ylabel(self.spectrum.y_label)
        ax.set_xlabel(self.spectrum.x_units)
        # plt.style.use("default")

    def plot_components(self, yscale="log"):
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
            x = self.spectrum.channels[self.channel_idx]
        else:
            x = self.spectrum.energies[self.channel_idx]
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.spectrum.counts[self.channel_idx], label="Raw spectrum")
        plt.plot(x, self.peak_plus_bkg.clip(1e-1), label="Peaks+Continuum")
        plt.plot(x, self.bkg.clip(1e-1), label="Continuum")
        plt.plot(x, self.signal.clip(1e-1), label="Peaks")
        plt.plot(x, self.noise.clip(1e-1), label="noise")
        if yscale == "log":
            plt.yscale("log")
        else:
            plt.yscale("linear")
        # plt.xlim(0, len(spec))
        plt.ylim(3e-1)
        plt.xlabel(self.spectrum.x_units)
        plt.ylabel(self.spectrum.y_label)
        plt.legend(loc=1)
        plt.style.use("default")
