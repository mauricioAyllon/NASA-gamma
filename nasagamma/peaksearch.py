"""
Allows to automatically find peaks in a spectrum object
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from . import spectrum as sp
import pandas as pd


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
            peak searching method including kernel method (km), fast FFT-based
            method (fast), and scipy peak finding method (scipy). The default is km.

        Raises
        ------
        Exception
            'spectrum' must be a Spectrum object.

        Returns
        -------
        None.

        """
        if not isinstance(spectrum, sp.Spectrum):
            raise TypeError(f"spectrum must be a Spectrum object, got {type(spectrum)} instead")
        
        if not isinstance(ref_x, (int, float)) or ref_x <= 0:
            raise ValueError("ref_x must be a positive number")
        
        if not isinstance(ref_fwhm, (int, float)) or ref_fwhm <= 0:
            raise ValueError("ref_fwhm must be a positive number")
        
        if not isinstance(fwhm_at_0, (int, float)) or fwhm_at_0 <= 0:
            raise ValueError("fwhm_at_0 must be a positive number")
        
        if not isinstance(min_snr, (int, float)) or min_snr <= 0:
            raise ValueError("min_snr must be a positive number")
        
        if method not in ("km", "fast", "scipy"):
            raise ValueError(f"method must be one of 'km', 'fast', or 'scipy', got '{method}'")
            
        self.xrange, self.channel_idx = self._parse_xrange(xrange, spectrum)
        self.ref_x = ref_x
        self.ref_fwhm = ref_fwhm
        self.fwhm_at_0 = fwhm_at_0
        self.spectrum = spectrum
        self.min_snr = min_snr
        self.method = method
        self.snr = None
        self.peak_plus_bkg = None
        self.bkg = None
        self.signal = None
        self.noise = None
        self.peaks_idx = None
        self.fwhm_guess = None
        self.edg = None
        if method == "km":
            self.calculate_km()
        elif method == "scipy":
            self.calculate_scipy()
        elif method == "fast":
            self.calculate_fast()
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'km', 'scipy', or 'fast'")
    
    def _parse_xrange(self, xrange, spectrum):
        """
        Parse and validate the xrange parameter.
    
        Parameters
        ----------
        xrange : list or numpy array of shape (2,), or None
            x range for peak searching.
        spectrum : Spectrum object
            previously initialized spectrum object.
    
        Returns
        -------
        xrange : list or None
            parsed xrange.
        channel_idx : numpy array of bool
            boolean mask for the channel range.
        """
        if xrange is None:
            return None, np.ones(len(spectrum.channels), dtype=bool)
        elif len(xrange) == 2 and spectrum.energies is not None:
            ixe = (spectrum.energies >= xrange[0]) & (spectrum.energies <= xrange[1])
            erange = spectrum.channels[ixe]
            return [erange[0], erange[-1]], ixe
        elif len(xrange) == 2 and spectrum.energies is None:
            channel_idx = (spectrum.channels >= xrange[0]) & (spectrum.channels <= xrange[1])
            return xrange, channel_idx
        else:
            raise ValueError("xrange must have exactly 2 elements: [x_min, x_max]")

    def metadata(self):
        """
        Return metadata of the PeakSearch instance as a dictionary.
    
        Returns
        -------
        dict
            Dictionary containing search parameters and results.
        """
        return {
            "ref_x": self.ref_x,
            "ref_fwhm": self.ref_fwhm,
            "fwhm_at_0": self.fwhm_at_0,
            "min_snr": self.min_snr,
            "method": self.method,
            "xrange": self.xrange,
            "n_peaks": len(self.peaks_idx) if self.peaks_idx is not None else 0,
            "peak_positions": self.peaks_idx,
            "fwhm_guess": self.fwhm_guess,
        }
    def peaks_in_range(self, x_min, x_max):
        """
        Return peak positions within a given range.
    
        Parameters
        ----------
        x_min : int or float
            minimum x value (channel or energy).
        x_max : int or float
            maximum x value (channel or energy).
    
        Returns
        -------
        numpy array
            peak positions within the specified range.
        """
        if self.peaks_idx is None:
            raise ValueError("No peaks found. Run calculate_km, calculate_fast, or calculate_scipy first.")
        mask = (self.peaks_idx >= x_min) & (self.peaks_idx <= x_max)
        return self.peaks_idx[mask]
    
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
            new_ch = self.spectrum.channels[self.channel_idx]
            new_cts = spect_cts[self.channel_idx]
            self.edg = np.append(new_ch, new_ch[-1] + 1)
            peak_plus_bkg, bkg, signal, noise, snr = self.convolve(self.edg, new_cts)

        clipped_snr = snr.clip(0)

        # find peak indices
        peaks_idx = find_peaks(clipped_snr, height=self.min_snr)[0]

        self.fwhm_guess = self.fwhm(peaks_idx)
        self.peak_plus_bkg = peak_plus_bkg
        self.bkg = bkg
        self.signal = signal
        self.noise = noise
        self.snr = clipped_snr
        if self.xrange is None:
            self.peaks_idx = peaks_idx
        else:
            self.peaks_idx = new_ch[peaks_idx]
        # self.reset()

    def calculate_scipy(self):
        if self.xrange is None:
            peaks_idx, params = find_peaks(
                self.spectrum.counts,
                prominence=self.min_snr,
                width=self.fwhm(self.spectrum.channels),
            )
            self.peaks_idx = peaks_idx
        else:
            new_ch = self.spectrum.channels[self.channel_idx]
            new_cts = self.spectrum.counts[self.channel_idx]
            new_widths = self.fwhm(new_ch)
            peaks_idx, params = find_peaks(
                new_cts, prominence=self.min_snr, width=new_widths
            )
            self.peaks_idx = new_ch[peaks_idx]

        self.fwhm_guess = self.fwhm(self.peaks_idx)
        self.snr = params["prominences"]
    
    def calculate_fast(self):
        """
        Fast peak finding using segmented FFT convolution with km-style
        normalization. Approximates calculate_km using overlapping segments
        with locally constant sigma. Primary output is peaks_idx and snr.
        Decomposition components are not computed.
        """
        from scipy.signal import fftconvolve
    
        if self.spectrum.cps and self.spectrum.livetime is not None:
            spect_cts = self.spectrum.counts * self.spectrum.livetime
        else:
            spect_cts = self.spectrum.counts
    
        channels = self.spectrum.channels[self.channel_idx]
        counts = spect_cts[self.channel_idx]
        n = len(counts)
    
        fwhm_vals = self.fwhm(channels)
        sigma_vals = fwhm_vals / 2.355
        sigma_min = 0.5
    
        snr = np.zeros(n)
        n_segments = 100
        segment_size = n // n_segments
    
        for seg in range(n_segments):
            i_start = seg * segment_size
            i_end = min(n, (seg + 1) * segment_size)
            mid = (i_start + i_end) // 2
            sigma = max(sigma_min, sigma_vals[mid])
    
            kernel_half_width = max(3, int(4 * sigma))
            kernel_x = np.arange(-kernel_half_width, kernel_half_width + 1, dtype=float)
    
            # Match calculate_km normalization
            kernel_raw = -kernel_x * np.exp(-0.5 * (kernel_x / sigma) ** 2)
            kern_pos = kernel_raw.clip(0)
            kern_neg = -kernel_raw.clip(-np.inf, 0)
            if kern_neg.sum() > 0:
                kern_neg *= kern_pos.sum() / kern_neg.sum()
            kernel = kern_pos - kern_neg
    
            # Convolve with full spectrum for correct boundary behavior
            conv = fftconvolve(counts, kernel, mode="same")
    
            # Noise matching calculate_km: sqrt(kernel^2 @ counts)
            noise_conv = fftconvolve(counts, kernel ** 2, mode="same")
            noise = np.sqrt(np.abs(noise_conv))
            noise[noise == 0] = 1
    
            snr[i_start:i_end] = (conv / noise)[i_start:i_end]
    
        clipped_snr = snr.clip(0)
        peaks_idx = find_peaks(clipped_snr, height=self.min_snr)[0]
    
        # Refine peak positions using local argmax within fwhm window
        refined_peaks = []
        for idx in peaks_idx:
            fwhm_local = int(self.fwhm(channels[idx]))
            i_start = max(0, idx - fwhm_local)
            i_end = min(n, idx + fwhm_local)
            local_max = np.argmax(counts[i_start:i_end]) + i_start
            refined_peaks.append(local_max)
    
        refined_peaks = np.array(refined_peaks) if refined_peaks else np.array([], dtype=int)
        self.snr = clipped_snr
        self.peaks_idx = channels[refined_peaks] if self.xrange is not None else refined_peaks
        self.fwhm_guess = self.fwhm(self.peaks_idx)
    
    def _get_x(self):
        """
        Return the x-axis values for the current channel range.
    
        Returns
        -------
        numpy array
            channels or energies depending on calibration.
        """
        if self.spectrum.energies is None:
            return self.spectrum.channels[self.channel_idx]
        else:
            return self.spectrum.energies[self.channel_idx]
        
    def to_csv(self, fileName):
        """
        Save peak positions, fwhm guesses, and SNR values at each peak
        to a CSV file.
    
        Parameters
        ----------
        fileName : string
            file name or path of where to save the file.
    
        Returns
        -------
        None.
        """
        if self.peaks_idx is None:
            raise ValueError("No peaks found. Run calculate_km, calculate_fast, or calculate_scipy first.")
        
        if self.spectrum.energies is not None:
            peak_energies = self.spectrum.energies[self.peaks_idx]
            cols = {"channel": self.peaks_idx,
                    f"energy ({self.spectrum.e_units})": peak_energies,
                    "fwhm_guess": self.fwhm_guess,
                    "snr": self.snr[self.peaks_idx]}
        else:
            cols = {"channel": self.peaks_idx,
                    "fwhm_guess": self.fwhm_guess,
                    "snr": self.snr[self.peaks_idx]}
    
        df = pd.DataFrame(data=cols)
        if fileName[-4:] == ".csv":
            df.to_csv(fileName, index=False)
        else:
            df.to_csv(f"{fileName}.csv", index=False)
            
    def plot_kernel(self, ax=None):
        """Plot the 3D matrix of kernels evaluated across the x values."""
        # edges = self.spectrum.channels
        edges = self.edg
        n_channels = len(edges) - 1
        kern_mat = self.kernel_matrix(edges)
        kern_min = kern_mat.min()
        kern_max = kern_mat.max()
        kern_min = min(kern_min, -1 * kern_max)
        kern_max = max(kern_max, -1 * kern_min)

        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        im = ax.imshow(
            kern_mat.T[::-1, :],
            cmap=plt.get_cmap("bwr"),
            vmin=kern_min,
            vmax=kern_max,
            extent=[n_channels, 0, 0, n_channels])
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Input x")
        ax.set_ylabel("Output x")
        ax.set_aspect("equal")
        ax.set_title("Kernel Matrix")

    def plot(self, yscale="linear", snrs="on", ax=None):
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
        plt.rc("font", size=12)
        plt.style.use("seaborn-v0_8-darkgrid")
        x = self._get_x()
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()

        if snrs == "on" and self.method == "km":
            ax.plot(x, self.snr, label="SNR")
        # ax.plot(x, self.spectrum.counts, label="Raw spectrum")
        if yscale == "log":
            ax.set_yscale("log")
        elif yscale == "linear":
            ax.set_yscale("linear")
        for xc in self.peaks_idx:
            if self.spectrum.energies is None:
                x0 = xc
            else:
                x0 = self.spectrum.energies[xc]
            ax.axvline(x=x0, color="red", linestyle="--", alpha=0.2)
        self.spectrum.plot(ax=ax, scale=yscale)
        ax.legend(loc=1)
        ax.set_title(f"SNR > {self.min_snr}")
        # ax.set_ylim(1e-1)
        ax.set_ylabel(self.spectrum.y_label)
        ax.set_xlabel(self.spectrum.x_units)
        # plt.style.use("default")

    def plot_components(self, yscale="log", ax=None):
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
        x = self._get_x()
        plt.rc("font", size=12)
        plt.style.use("seaborn-v0_8-darkgrid")
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.spectrum.counts[self.channel_idx], label="Raw spectrum")
        ax.plot(x, self.peak_plus_bkg.clip(1e-1), label="Peaks+Continuum")
        ax.plot(x, self.bkg.clip(1e-1), label="Continuum")
        ax.plot(x, self.signal.clip(1e-1), label="Peaks")
        ax.plot(x, self.noise.clip(1e-1), label="noise")
        ax.set_yscale(yscale)
        ax.set_ylim(3e-1)
        ax.set_xlabel(self.spectrum.x_units)
        ax.set_ylabel(self.spectrum.y_label)
        ax.legend(loc=1)
