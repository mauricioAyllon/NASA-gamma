# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:46:23 2020

@author: mauricio
Peak fit class: Gaussian + Linear fit
"""
import numpy as np
from . import peaksearch as ps
import lmfit
import pandas as pd
import matplotlib.pyplot as plt

# need to check if search is a PeakSearch object

class PeakFit:
    
    def __init__(self, search, xrange, bkg='linear'):
        """Initialize with a peaksearch object."""
        
        self.search = search
        self.xrange = xrange
        self.bkg = bkg
        self.x_data = 0
        self.y_data = 0
        self.peak_info = []
        self.fit_result = 0
        if search.spectrum.energies is None:
            print("Working with channel numbers")
            self.x = search.spectrum.channels[0:-1]
            self.chan = search.spectrum.channels[0:-1]
            self.x_units = "Channels"
        else:
            print("Working with energy values")
            self.x = search.spectrum.energies
            self.chan = search.spectrum.channels[0:-1]
            self.x_units = "Energy"
        
        self.gaussians_bkg()
    
    def find_peaks_range(self):
        
        mask = mask = (self.x[self.search.peaks_idx] > self.xrange[0])*\
                (self.x[self.search.peaks_idx] < self.xrange[1])
       
        if sum(mask) == 0:
            print(f"Found 0 peaks within range {self.xrange}")
            print("Make sure the SNR is set low enough")
        else:
            print(f"Found {sum(mask)} peak(s) within range {self.xrange}")
        pidx = self.search.peaks_idx[mask]
        return mask, pidx
        
    def init_values(self):
        # chan = self.search.spectrum.channels[0:-1]
        # erg = self.search.spectrum.energies
        cts = self.search.spectrum.counts
        m = np.polyfit(self.chan, self.x, 1)[0] # energy/channel
        left = cts[np.where(self.x > self.xrange[0])[0][0]]
        right = cts[np.where(self.x > self.xrange[1])[0][0]]
        
        mask, pks_idx = self.find_peaks_range()
        erg0 = self.x[pks_idx]
        sig0 = self.search.fwhm_guess[mask]*m/2.355
        height0 = abs(cts[pks_idx] - (right+left)/2)
        amp0 = height0*sig0/0.4
        m0 = (right-left)/(self.xrange[1]-self.xrange[0])
        b0 = left - m0*self.xrange[0]
        return m0, b0, amp0, erg0, sig0
    
       
    def gaussians_bkg(self):
        
        maskx = (self.x > self.xrange[0]) * (self.x < self.xrange[1])
        m,b,amp,erg,sig = self.init_values()
        # number of peaks detected in range
        npeaks = len(erg)
        
        y0 = self.search.spectrum.counts[maskx]
        x0 = self.x[maskx]
        self.y_data = y0
        self.x_data = x0
        
        if self.bkg == 'linear':
            # here we guess the slope and the intercept
            lin_mod = lmfit.models.LinearModel()
            pars = lin_mod.make_params(slope=m, itercept=b)
            model = lin_mod
        elif self.bkg == 'quadratic':
            quad_mod = lmfit.models.QuadraticModel(prefix='quadratic')
            pars = quad_mod.guess(y0, x=x0)
            model = quad_mod
        elif self.bkg == 'exponential':
            exp_mod = lmfit.models.ExponentialModel()
            pars = exp_mod.guess(y0, x=x0)
            model = exp_mod
        else:
            # assume polynomial of degree n
            n = [int(s) for s in list(self.bkg) if s.isdigit()][0]
            poly_mod = lmfit.models.PolynomialModel(degree=n)
            pars = poly_mod.guess(y0, x=x0)
            model = poly_mod
            
        for i in range(npeaks):
            gauss0 = lmfit.models.GaussianModel(prefix=f'g{i+1}_')
            pars.update(gauss0.make_params())
            pars[f"g{i+1}_center"].set(value=erg[i])
            pars[f"g{i+1}_sigma"].set(value=sig[i])
            pars[f"g{i+1}_amplitude"].set(value=amp[i])
            model += gauss0   
        fit0 = model.fit(y0, pars, x=x0)  
        print(fit0.message)
        components = fit0.eval_components()
        self.fit_result = fit0
        
        # save some extra info
        for i in range(npeaks):
            mean0 = fit0.best_values[f"g{i+1}_center"]
            g0 = components[f"g{i+1}_"]
            area0 = g0.sum()
            fwhm0 = fit0.best_values[f"g{i+1}_sigma"] * 2.355
            dict_peak_info = {f'mean{i+1}': mean0, f'area{i+1}': area0,
                              f'fwhm{i+1}': fwhm0}
            self.peak_info.append(dict_peak_info)
            
    def plot(self, plot_type="simple", legend="on"):
        x = self.x_data
        y = self.y_data
        #init_fit = self.fit_result.init_fit
        best_fit = self.fit_result.best_fit
        res = self.fit_result
        
        comps = res.eval_components()
        
        if "poly" in self.bkg:
            n = [int(s) for s in list(self.bkg) if s.isdigit()][0]
            bkg_label = 'polynomial'
        else:
            n = "N/A"
            bkg_label = self.bkg
        
        if plot_type == "simple":
            plt.rc("font", size=14)  
            plt.style.use("seaborn-darkgrid")
            plt.figure(figsize=(10,8))
            plt.title(f"Reduced $\chi^2$ = {round(res.redchi,4)}")
            plt.plot(x,y,'bo', alpha=0.5, label="data")
            plt.plot(x, best_fit, 'r', lw=3, alpha=0.5, label="Best fit")
            for cp in range(len(comps)-1):
                plt.plot(x, comps[f'{bkg_label}'] + comps[f'g{cp+1}_'], 'k--',
                         lw=2, label=f'Gaussian {cp+1} + {bkg_label}: n={n}')
                plt.plot(x, comps[f'{bkg_label}'], 'g--', label="bkg")
            
            dely = res.eval_uncertainty(sigma=3)
            plt.fill_between(x, res.best_fit-dely, res.best_fit+dely,
                             color="#ABABAB",
                             label='3-$\sigma$ uncertainty band')
            plt.xlabel(self.x_units)
            if legend == 'on':
                plt.legend()
            plt.style.use("default")
            
        elif plot_type == 'full':
            cols = ["mean", "net_area", "fwhm"]
            mean = []
            area = []
            fwhm = []
            for i in self.peak_info:
                ls = list(i.values())
                mean.append(round(ls[0],3))
                area.append(round(ls[1],3))
                fwhm.append(round(ls[2],3))
            
            rs = np.array([mean, area, fwhm]).T
            colors = [['lightblue']*len(cols)]*len(rs)
            plt.rc("font", size=14)
            plt.style.use("seaborn-darkgrid")
            fig = plt.figure(constrained_layout=False, figsize=(14,8))
            gs = fig.add_gridspec(2, 2, width_ratios=[5,1],
                                   height_ratios=[1,4])
            f_ax1 = fig.add_subplot(gs[0, 0])
            f_ax1.plot(x, res.residual, '.', ms=10, alpha=0.5)
            f_ax1.hlines(y=0, xmin=x.min(), xmax=x.max(), lw=3)
            f_ax1.set_ylabel("Residual")
            f_ax1.set_xlim([x.min(), x.max()])
            f_ax1.set_xticks([])
            # f_ax1.set_yticks(np.arange(min(res.residual),
            #                            max(res.residual)+1, 4.0))
            f_ax2 = fig.add_subplot(gs[1, 0])
            f_ax2.set_title(f"Reduced $\chi^2$ = {round(res.redchi,4)}")
            f_ax2.plot(x,y,'bo', alpha=0.5, label="data")
            f_ax2.plot(x, best_fit, 'r', lw=3, alpha=0.5, label="Best fit")
            m = 1
            for cp in range(len(comps)-1):
                if m == 1:
                    f_ax2.plot(x, comps[f'{bkg_label}'] + comps[f'g{cp+1}_'],
                            'k--', lw=2,
                            label=f'Gaussian {cp+1} + {bkg_label}: n={n}')
                    f_ax2.plot(x, comps[f'{bkg_label}'], 'g--', label="bkg")
                else:
                    f_ax2.plot(x, comps[f'{bkg_label}'] + comps[f'g{cp+1}_'],
                            'k--', lw=2)
                    f_ax2.plot(x, comps[f'{bkg_label}'], 'g--')
                m = 0
                    
            dely = res.eval_uncertainty(sigma=3)
            f_ax2.fill_between(x, res.best_fit-dely, res.best_fit+dely,
                             color="#ABABAB",
                             label='3-$\sigma$ uncertainty band')
            f_ax2.set_xlabel(self.x_units)
            if legend == 'on':
                f_ax2.legend()
            plt.style.use("default")
            
            f_ax3 = fig.add_subplot(gs[0:, 1:])
            t = f_ax3.table(cellText=rs,colLabels=cols,loc='center',
                             cellLoc='center', 
                             colColours =["palegreen"] * len(cols),
                             cellColours=colors)
            t.scale(1.8, 4)
            t.auto_set_font_size(False)
            t.set_fontsize(14)
            f_ax3.axis('off')
        
            
class AddPeaks:
    
    def __init__(self, filename, n=0):
        # add option to save to hdf file
        """Add peak fit objects for further analysis."""
        
        self.filename = filename
        self.all_peaks = []
        self.n = n
        if n == 0:
            cols = ['x_data', 'y_data', 'mean', 'area', 'fwhm', 'best_fit',
                    'redchi', 'gauss', 'uncertainty', 'bkg', 'bkg_type']
            #cols = ['x_data', 'y_data', 'mean']
            self.df = pd.DataFrame(columns=cols)
            self.df.to_hdf(f"{filename}.hdf", key="data")
        else:
            print(f"Appending to existing file: {filename}.hdf")
            self.df = pd.read_hdf(f"{filename}.hdf", key="data")
        
    def add_peak(self, fit_obj):
        # need to check it is a fit object
        self.all_peaks.append(fit_obj)
        npeaks = len(fit_obj.peak_info)
        
        # save to pandas dataframe
        for i in range(npeaks):
            x_data = fit_obj.x_data
            self.df.loc[self.n, 'x_data'] = x_data
            y_data = fit_obj.y_data
            self.df.loc[self.n, 'y_data'] = y_data
            mean = list(fit_obj.peak_info[i].keys())[0]
            self.df.loc[self.n, 'mean'] = fit_obj.peak_info[i][mean]
            area = list(fit_obj.peak_info[i].keys())[1]
            self.df.loc[self.n, 'area'] = fit_obj.peak_info[i][area]
            fwhm = list(fit_obj.peak_info[i].keys())[2]
            self.df.loc[self.n, 'fwhm'] = fit_obj.peak_info[i][fwhm]
            best_fit = fit_obj.fit_result.best_fit
            self.df.loc[self.n, 'best_fit'] = best_fit
            redchi = fit_obj.fit_result.redchi
            self.df.loc[self.n, 'redchi'] = redchi
            bkg = list(fit_obj.fit_result.eval_components().keys())[0]
            comps = fit_obj.fit_result.eval_components()
            self.df.loc[self.n, 'bkg'] = comps[bkg]
            gauss = list(fit_obj.fit_result.eval_components().keys())[i+1]
            self.df.loc[self.n, 'gauss'] = comps[gauss]
            uncertainty = fit_obj.fit_result.eval_uncertainty()
            self.df.loc[self.n, 'uncertainty'] = uncertainty
            bkg = fit_obj.bkg
            self.df.loc[self.n, 'bkg_type'] = bkg
            self.n += 1
        self.df.to_hdf(f"{self.filename}.hdf", key="data")
            
    def reset(self):
        self.all_peaks = []
        self.n = 0
        
    def del_peak(self, pos):
        self.all_peaks.pop(pos)
  
        
def consecutive(data, stepsize=0):
    idx = np.where(np.diff(data[:,1]) != stepsize)[0]+1
    return np.split(data, idx)

def auto_range(search, fwhm_factor):
    f = fwhm_factor
    pidx =  search.peaks_idx
    chan = search.spectrum.channels[:-1]
    
    fwhm_guess = search.fwhm_guess
    density = sum((abs(xi - chan) < f*fw) for xi,fw in zip(pidx, fwhm_guess))
    
    dens2 = np.vstack((chan, density)).T
    rs = consecutive(dens2)
    ranges = []
    for arr in rs:
        if arr[:,1].sum() != 0 and np.isin(arr[:,0], pidx).sum() > 0:
            mi = arr[:,0].min()
            ma = arr[:,0].max()
            left = round(mi - 2*search.fwhm(mi))
            right = round(ma + 2*search.fwhm(ma))
            if right > pidx.max():
                right = pidx.max()
            ranges.append([int(left), int(right)])
            
    return ranges
        
def auto_scan(search, lst=None, plot=False, save_to_hdf=False):
    #TODO: optimize bkg, save to hdf (using AddPeaks),
    # add checks (fit message, positive area, etc)
    fits = []
    if lst is None:
        ranges = auto_range(search, 2)
    else:
        ranges = lst
    
    bkgs = ["poly1", "poly2"]
    for rg in ranges:
        redchi = 1e10
        for bk in bkgs:
            fit0 = PeakFit(search, rg, bkg=bk)
            if "Fit succeeded." != fit0.fit_result.message:
                next
            elif fit0.fit_result.redchi < redchi:
                fitx = fit0
                redchi = fitx.fit_result.redchi 
            else:
                fitx = 0
            
        if plot and fitx != 0:
            fitx.plot(plot_type="full")
        fits.append(fitx)
    return fits
   

    
    
    
    
    
    
    
    