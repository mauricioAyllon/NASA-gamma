# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:08:07 2020

@author: mauricio
"""
import numpy as np
from spectrum import Spectrum
from scipy.signal import find_peaks

 
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
        return kern_pos - kern_neg
    
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
        
        self.peak_plus_bkg = peak_plus_bkg
        self.bkg = bkg
        self.signal = signal
        self.noise = noise
        self.snr = snr
        self.peaks_idx = peaks_idx
        #self.reset()
    
    # def peak_finder(self, min_snr=2):
    #     """Find the highest SNR peaks in the data."""
    #     idx = find_peaks(self.snr, height=min_snr)[0]
    #     return idx