# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:14:25 2020

@author: mauricio
"""
import numpy as np

class Spectrum():
    """
    Initialize a Spectrum.

    Data Attributes:
      counts: counts per bin, or cps
      channels: np.array of raw/uncalibrated bin edges
      energies: np.array of energy bin edges, if calibrated
     
    """

    def __init__(self, counts=None, channels=None, energies=None):
        """Initialize the spectrum.

        cpb is the only required input parameter

        Args:
          counts: counts per bin, or cps
          channels (optional): array of bin edges. If None, assume 
          energies (optional): an array of bin edge energies
        """
        if counts is None:
            print('ERROR: Must specify counts')
        if channels is None:
            channels = np.arange(0,len(counts)+1,1)

        self.counts = np.array(counts, dtype=float)
        self.channels = np.array(channels, dtype=int)
        self.energies = np.array(energies, dtype=float)
        
    def smooth():
        pass
    
    def rebin():
        pass
    
    def plot():
        pass
       
