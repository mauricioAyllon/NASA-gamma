# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:14:25 2020

@author: mauricio
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
          channels (optional): array of bin edges. If None, assume based on
          counts
          energies (optional): an array of bin edge energies
        """
        if counts is None:
            print("ERROR: Must specify counts")
        if channels is None:
            channels = np.arange(0,len(counts)+1,1)
        if energies is not None:
            self.energies = np.array(energies, dtype=float)
            self.x_units = "Energy"
        else:
            self.energies = energies
            self.x_units = "Channels"

        self.counts = np.array(counts, dtype=float)
        self.channels = np.array(channels, dtype=int)
        
    def smooth(self, num=4):
        '''
        Parameters
        ----------
        num : integer, optional
            number of data points for averaging. The default is 4.

        Returns
        -------
        numpy array
            moving average of counts.

        '''
        df = pd.DataFrame(data=self.counts, columns=["cts"])
        mav = df.cts.rolling(window=num, center=True).mean()
        mav.fillna(0, inplace=True)
        return np.array(mav)
    
    def rebin(self):
        '''
        Rebins data by adding two adjacent bins at a time.

        Returns
        -------
        numpy array
            Rebinned counts
        If energies are passed, returns both rebinned counts and average
        energies

        '''
        arr_cts = self.counts
        if arr_cts.shape[0] % 2 != 0:
            arr_cts = arr_cts[:-1]
        y0 = arr_cts[::2]
        y1 = arr_cts[1::2]
        y = y0 + y1
        if self.energies is None:
            return y
        else:
            erg = self.energies
            if erg.shape[0] % 2 != 0:
                erg = erg[:-1]
            en0 = erg[::2]
            en1 = erg[1::2]
            en = (en0 + en1) / 2
            return en, y
    
    def plot(self, scale='log'):
        x = self.channels[:-1]
        y = self.counts
        integral = round(y.sum())
        plt.rc("font", size=14)  
        plt.style.use("seaborn-darkgrid")
        plt.figure()
        plt.fill_between(x, 0, y, alpha=0.5, color="C0", step="pre")
        plt.plot(x,y, drawstyle="steps")
        plt.yscale(scale)
        plt.title(f"Raw Spectrum. Integral = {integral}")
        plt.xlabel("Channels")
        plt.ylabel("a.u")
        plt.style.use("default")
        
        if self.energies is not None:
            x = self.energies
            y= self.counts
            plt.rc("font", size=14)  
            plt.style.use("seaborn-darkgrid")
            plt.figure()
            plt.fill_between(x, 0, y, alpha=0.5, color="C1", step="pre")
            plt.plot(x,y, color="C1", drawstyle="steps")
            plt.yscale(scale)
            plt.title(f"Raw Spectrum. Integral = {integral}")
            plt.xlabel("Energy")
            plt.ylabel("a.u")
            plt.style.use("default")
            
        
       
