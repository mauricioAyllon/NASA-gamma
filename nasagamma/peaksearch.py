# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:08:07 2020

@author: mauricio
"""
import numpy as np
from . import spectrum as sp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# need to check that spectrum is a spectrum object
 
def gaussian0(x, mean, sigma):
    """Gaussian function."""
    z = (x - mean) / sigma
    return np.exp(-z**2 / 2.)

def gaussian1(x, mean, sigma):
    """First derivative of a gaussian."""
    z = (x - mean)
    return -1 * z * gaussian0(x, mean, sigma)

class PeakSearchError(Exception):
    """Base class for errors in PeakSearch."""
    pass

class PeakSearch:
    
    def __init__(self, spectrum, ref_x, ref_fwhm, fwhm_at_0=1.0, min_snr=2):
        """Initialize with a spectrum object."""
        
        self.ref_x = ref_x
        self.ref_fwhm = ref_fwhm
        self.fwhm_at_0 = fwhm_at_0
        self.spectrum = spectrum
        self.min_snr = min_snr
        self.snr = []
        self.peak_plus_bkg = []
        self.bkg = []
        self.signal = []
        self.noise = []
        self.peaks_idx = []
        self.fwhm_guess = []
        self.calculate()
        #self.centroids = []
        #self.snrs = []
        #self.fwhms = []
        #self.integrals = []
        #self.backgrounds = []

    def fwhm(self, x):
        """Calculate the expected FWHM at the given x value."""
        # f(x)^2 = f0^2 + k x^2
        # f1^2 = f0^2 + k x1^2
        # k = (f1^2 - f0^2) / x1^2
        # f(x)^2 = f0^2 + (f1^2 - f0^2) (x/x1)^2
        f0 = self.fwhm_at_0
        f1 = self.ref_fwhm
        x1 = self.ref_x
        fwhm_sqr = f0**2 + (f1**2 - f0**2) * (x / x1)**2
        return np.sqrt(fwhm_sqr)
    
    def kernel(self, x, edges):
        """Generate the kernel for the given x value."""
        fwhm1 = self.fwhm(x)
        sigma = fwhm1 / 2.355
        g1_x0 = gaussian1(edges[:-1], x, sigma)
        g1_x1 = gaussian1(edges[1:], x, sigma)
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
    
    # now convolve the spectrum with the kernel
    def convolve(self, edges, data):
        """Convolve this kernel with the data."""
        kern_mat = self.kernel_matrix(edges)
        kern_mat_pos = +1 * kern_mat.clip(0, np.inf)
        kern_mat_neg = -1 * kern_mat.clip(-np.inf, 0)
        peak_plus_bkg = np.dot(kern_mat_pos, data)
        bkg = np.dot(kern_mat_neg, data)
        signal = np.dot(kern_mat, data)
        noise = np.dot(kern_mat**2, data)
        noise = np.array([np.sqrt(x) for x in noise])
        snr = np.zeros_like(signal)
        snr[noise > 0] = signal[noise > 0] / noise[noise > 0]
        return peak_plus_bkg, bkg, signal, noise, snr

    def calculate(self):
        """Calculate the convolution of the spectrum with the kernel."""
        
        # if not isinstance(self.spectrum, Spectrum):
        #     raise PeakFinderError(
        #         'Argument must be a Spectrum, not {}'.format(type(self.spectrum)))
        
        snr = np.zeros(len(self.spectrum.counts))
        chan = self.spectrum.channels
        # calculate the convolution
        peak_plus_bkg, bkg, signal, noise, snr = \
            self.convolve(chan, self.spectrum.counts)
        # find peak indices
        peaks_idx = find_peaks(snr.clip(0), height=self.min_snr)[0]
        
        #for pk in peaks_idx:
        self.fwhm_guess = self.fwhm(peaks_idx)
        
        self.peak_plus_bkg = peak_plus_bkg
        self.bkg = bkg
        self.signal = signal
        self.noise = noise
        self.snr = snr.clip(0)
        self.peaks_idx = peaks_idx
        #self.reset()
        
    def plot_kernel(self):
        """Plot the matrix of kernels evaluated across the x values."""
        edges = self.spectrum.channels
        n_channels = len(edges) - 1
        kern_mat = self.kernel_matrix(edges)
        kern_min = kern_mat.min()
        kern_max = kern_mat.max()
        kern_min = min(kern_min, -1 * kern_max)
        kern_max = max(kern_max, -1 * kern_min)
        
        plt.figure()
        plt.imshow(
            kern_mat.T[::-1, :], cmap=plt.get_cmap('bwr'),
            vmin=kern_min, vmax=kern_max,
            extent=[n_channels, 0, 0, n_channels])
        plt.colorbar()
        plt.xlabel('Input x')
        plt.ylabel('Output x')
        plt.gca().set_aspect('equal')
        plt.title("Kernel Matrix")
        
    def plot_peaks(self):
        if self.spectrum.energies is None:
            x = self.spectrum.channels[:-1]
        else:
            x = self.spectrum.energies
        plt.rc("font", size=14)  
        plt.style.use("seaborn-darkgrid")
        plt.figure("peak-search")
        plt.plot(x, self.snr, label="SNR all")
        plt.plot(x, self.spectrum.counts, label="Raw spectrum")
        plt.yscale("log")
        for xc in self.peaks_idx:
            if self.spectrum.energies is None:
                x0 = xc
            else:
                x0 = self.spectrum.energies[xc]
            plt.axvline(x=x0, color='red', linestyle='-', alpha=0.5)
        plt.legend(loc=1)
        plt.title(f"SNR > {self.min_snr}")
        plt.ylim(1e-1)
        plt.ylabel("Cts/MeV/s")
        plt.xlabel("Energy [MeV]")
        plt.style.use("default")
    
    def plot_components(self):
        if self.spectrum.energies is None:
            x = self.spectrum.channels[:-1]
        else:
            x = self.spectrum.energies
        plt.rc("font", size=14) 
        plt.style.use("seaborn-darkgrid")
        plt.figure(figsize=(10,6))
        plt.plot(x, self.spectrum.counts, label='Raw spectrum')
        plt.plot(x, self.peak_plus_bkg.clip(1e-1), label='Peaks+Continuum')
        plt.plot(x, self.bkg.clip(1e-1), label='Continuum')
        plt.plot(x, self.signal.clip(1e-1), label='Peaks')
        plt.plot(x, self.noise.clip(1e-1), label='noise')
        plt.yscale('log')
        #plt.xlim(0, len(spec))
        plt.ylim(3e-1)
        plt.xlabel(self.spectrum.x_units)
        plt.ylabel('Counts')
        plt.legend(loc=1)
        plt.style.use("default")

        
     
    
   