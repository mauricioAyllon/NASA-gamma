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


class PeakFit:
    
    def __init__(self, search, xrange, bkg='linear'):
        '''
        Use the package LMfit to fit and plot peaks with Gaussians for a given
        background. It must be initialized with a PeakSearch object.

        Parameters
        ----------
        search : PeakSearch object.
            from peaksearch.py.
        xrange : list
            range of 2 x-values where the fit is attempted. Can be either in
            energy values or channels depending on how the search object
            was initialized.
        bkg : string, optional
            Can be 'linear', 'quadratic', 'exponential' or 'polyn', where n
            is any degree polynomial. The default is 'linear'.

        Raises
        ------
        Exception
            'search' must be a PeakSearch object.

        Returns
        -------
        None.

        '''
        
        if type(search) != ps.PeakSearch:
            raise Exception("'search' must be a PeakSearch object")
        self.search = search
        self.xrange = xrange
        self.bkg = bkg
        self.x_data = 0
        self.y_data = 0
        self.peak_info = []
        self.peak_err = []
        self.fit_result = 0
        self.x_units = search.spectrum.x_units
        self.chan = search.spectrum.channels
        if search.spectrum.energies is None:
            print("Working with channel numbers")
            self.x = search.spectrum.channels
              
        else:
            print("Working with energy values")
            self.x = search.spectrum.energies
          
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
        epar = fit0.params
        for i in range(npeaks):
            mean0 = fit0.best_values[f"g{i+1}_center"]
            g0 = components[f"g{i+1}_"]
            area0 = g0.sum()
            fwhm0 = fit0.best_values[f"g{i+1}_sigma"] * 2.355
            dict_peak_info = {f'mean{i+1}': mean0, f'area{i+1}': area0,
                              f'fwhm{i+1}': fwhm0}
            self.peak_info.append(dict_peak_info)
            # errors
            mean_err = epar[f"g{i+1}_center"].stderr
            area_err = np.sqrt(area0)
            fwhm_err = epar[f"g{i+1}_fwhm"].stderr
            dict_peak_err = {f'mean_err{i+1}': mean_err,
                             f'area_err{i+1}': area_err,
                             f'fwhm_err{i+1}': fwhm_err}
            self.peak_err.append(dict_peak_err)
            
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
            fig = plt.figure(constrained_layout=False, figsize=(16,8))
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

def ecalibration(mean_vals, erg, channels, n=1, e_units="keV", plot=False):
    '''
    Perform energy calibration with LMfit. Polynomial degree = n.

    Parameters
    ----------
    mean_vals : list or numpy array
        mean values of fitted peaks (in channel numbers).
    erg : list or numpy array
        energy values corresponding to mean_vals.
    channels : list or numpy array
        channel values.
    n : integer, optional
        polynomial degree. The default is 1.
    e_units : string, optional
        energy units. The default is "keV".
    plot : bool, optional
        plot calibration curve and calibration points. The default is False.

    Returns
    -------
    predicted : numpy array
        predicted energy values spanning all channel values.
    fit : LMfit object.
        curve fit.

    '''
    y0 = np.array(erg, dtype=float)
    x0 = np.array(mean_vals, dtype=float)
    poly_mod = lmfit.models.PolynomialModel(degree=n)
    pars = poly_mod.guess(y0, x=x0)
    model = poly_mod
    fit = model.fit(y0, params=pars, x=x0)
    predicted = fit.eval(x=channels)
    ye = fit.eval_uncertainty()
    if plot:
        x_offset = 100
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")
        fig = plt.figure(constrained_layout=False, figsize=(12,8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1,4])
        f_ax1 = fig.add_subplot(gs[0, 0])
        f_ax1.plot(mean_vals, fit.residual, '.', ms=15, alpha=0.5, color="red")
        f_ax1.hlines(y=0, xmin=min(channels)-x_offset,
                     xmax=max(channels), lw=3)
        f_ax1.set_ylabel("Residual")
        f_ax1.set_xlim([min(channels)-x_offset, max(channels)])
        f_ax1.set_xticks([])
        f_ax2 = fig.add_subplot(gs[1, 0])
        f_ax2.set_title("Energy calibration")
        f_ax2.errorbar(mean_vals, erg, yerr=ye, ecolor='red', elinewidth=5,
                       capsize=12, capthick=3, marker="s", mfc="black",
                       mec="black", markersize=7, ls="--", lw=3,
                       label="Best fit")
        f_ax2.plot(channels, predicted, ls="--", lw=3, color="green",
                         label="Predicted")
        f_ax2.set_xlim([min(channels)-x_offset, max(channels)])
        f_ax2.set_xlabel("Channels")
        f_ax2.set_ylabel(f"Energy [{e_units}]")
        plt.legend()
        plt.style.use("default")
    return predicted, fit


class GaussianComponents:
    
    def __init__(self, fit_obj_lst=None, df_peak=None):
        '''
        Extract background subtracted Gaussian components and plot them
        together with a table of relevant values and uncertainties.

        Parameters
        ----------
        fit_obj_lst : list, optional
            list of FitPeak objects. The default is None.
        df_peak : pandas dataframe, optional
            dataframe of peak info as saved by using the class AddPEaks.
            The default is None.

        Returns
        -------
        None.

        '''
        self.fit_obj_lst = fit_obj_lst
        self.df_peak = df_peak
        self.npeaks = 0
        self.mean = []
        self.area = []
        self.fwhm = []
        self.gauss = []
        self.x_data = []
        self.peak_err = []
        if fit_obj_lst is not None:
            self.x_units = fit_obj_lst[0].x_units
            self.gauss_peakfit()
        elif df_peak is not None:
            self.x_units = df_peak.loc[0,"x_units"]
            self.gauss_df()
            
    def gauss_peakfit(self):
        for fit_obj in self.fit_obj_lst:
            npeaks = len(fit_obj.peak_info)
            self.npeaks += npeaks
            comps = fit_obj.fit_result.eval_components()
            for i in range(npeaks):
                self.peak_err.append(fit_obj.peak_err[i])
                x_data = fit_obj.x_data
                self.x_data.append(x_data)
                m = list(fit_obj.peak_info[i].keys())[0]
                a = list(fit_obj.peak_info[i].keys())[1]
                f = list(fit_obj.peak_info[i].keys())[2]
                mean = fit_obj.peak_info[i][m]
                area = fit_obj.peak_info[i][a]
                fwhm = fit_obj.peak_info[i][f]
                g = list(fit_obj.fit_result.eval_components().keys())[i+1]
                gauss = comps[g]
                self.mean.append(mean)
                self.area.append(area)
                self.fwhm.append(fwhm)
                self.gauss.append(gauss)
        
    def gauss_df(self):
        self.npeaks = self.df_peak.index.shape[0]
        for i in self.df_peak.index:
            self.x_data.append(self.df_peak.loc[i,"x_data"])
            self.gauss.append(self.df_peak.loc[i, "gauss"])
            self.mean.append( self.df_peak.loc[i,"mean"])
            self.area.append(self.df_peak.loc[i,"area"])
            self.fwhm.append(self.df_peak.loc[i,"fwhm"])
            dict_err = {}
            dict_err["mean_err"] = self.df_peak.loc[i, 'mean_err']
            dict_err["area_err"] = self.df_peak.loc[i, 'area_err']
            dict_err["fwhm_err"] = self.df_peak.loc[i, 'fwhm_err']
            self.peak_err.append(dict_err)
            
            
                
    def plot_gauss(self, plot_type="simple", table_scale=[1,3]):
        if plot_type == "simple":
            plt.rc("font", size=14)
            plt.style.use("seaborn-darkgrid")
            plt.figure(figsize=(12,8))
            for i in range(self.npeaks):
                x = self.x_data[i]
                y = self.gauss[i]
                plt.fill_between(x, 0, y, alpha=0.5) 
                x0 = round(self.mean[i], 2)
                y0 = y.max()
                a = round(self.area[i], 2)
                str0 = f"mean={x0} \narea={a}"
                plt.text(x0,y0, str0)
            plt.xlabel(self.x_units)
            plt.ylabel("Cts")
            plt.style.use("default")
            
        elif plot_type == "full":
            cols = ["Mean", "Net_Area", "FWHM"]
            mean = self.mean
            area = self.area
            fwhm = self.fwhm
            rs = np.array([mean, area, fwhm]).T
            rs = np.round(rs, decimals=2)
            plt.rc("font", size=14)
            plt.style.use("seaborn-darkgrid")
            fig = plt.figure(constrained_layout=False, figsize=(18,8))
            gs = fig.add_gridspec(1, 2, width_ratios=[3,2.5]) # 5 , 3
            f_ax1 = fig.add_subplot(gs[0, 0])
            for i in range(self.npeaks):
                            x = self.x_data[i]
                            y = self.gauss[i]
                            f_ax1.fill_between(x, 0, y, alpha=0.5) 
                            x0 = mean[i]
                            y0 = y.max()
                            a = area[i]
                            f_ax1.text(x0,y0, int(i+1),
                                       bbox=dict(facecolor='red', alpha=0.1),
                                       weight="bold")
            f_ax1.set_xlabel(self.x_units)
            f_ax1.set_ylabel("Cts")
            f_ax1.set_title("Gaussian Components")
            
            f_ax2 = fig.add_subplot(gs[0:, 1:])
            cols.insert(0,"Peak_N")
            colors = [['lightblue']*len(cols)]*len(rs)
            
            # errors
            vals_lst = []
            for element in self.peak_err:
                vals_lst.append(list(element.values()))
            maf = [float(item) for sublist in vals_lst for item in sublist]
            
            lst1 = []
            n = 1
            n2 = 0
            for i in rs:
                lst0 = []
                for j in i:
                    str0 = f"{round(j,2)} +/- {round(maf[n2],2)}"
                    lst0.append(str0)
                    n2 += 1
                lst0.insert(0,n)
                lst1.append(lst0)
                n += 1
            
            df = pd.DataFrame(lst1, columns=cols)
            
            t = f_ax2.table(cellText=df.values, colLabels=cols,loc='center',
                              cellLoc='center', 
                              colWidths=[1/8,1/3,1/3,1/3],
                              colColours =["palegreen"] * len(cols),
                              cellColours=colors)
            t.scale(table_scale[0], table_scale[1])
            t.auto_set_font_size(False)
            t.set_fontsize(12)
            f_ax2.axis('off')
            plt.style.use("default")
            
        
class AddPeaks:
    
    def __init__(self, filename, n=0):
        '''
        Save peak fit objects to a pandas dataframe for further analysis.

        Parameters
        ----------
        filename : string.
            filename to save peak info as hdf.
        n : integer, optional
            Add peaks to an existing hdf file with n being the row number
            to append peaks. The default is 0.

        Returns
        -------
        None.

        '''
        
        self.filename = filename
        self.all_peaks = []
        self.n = n
        if n == 0:
            cols = ['x_data', 'y_data', 'mean', 'area', 'fwhm', 'best_fit',
                    'redchi', 'gauss', 'uncertainty', 'bkg', 'bkg_type',
                    'mean_err', 'area_err', 'fwhm_err', 'x_units']
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
        x_data = fit_obj.x_data
        y_data = fit_obj.y_data
        best_fit = fit_obj.fit_result.best_fit
        redchi = fit_obj.fit_result.redchi
        bkg = list(fit_obj.fit_result.eval_components().keys())[0]
        comps = fit_obj.fit_result.eval_components()
        uncertainty = fit_obj.fit_result.eval_uncertainty()
        bkg_type = fit_obj.bkg
        for i in range(npeaks):
            self.df.loc[self.n, 'x_data'] = x_data
            self.df.loc[self.n, 'y_data'] = y_data
            mean = list(fit_obj.peak_info[i].keys())[0]
            self.df.loc[self.n, 'mean'] = fit_obj.peak_info[i][mean]
            area = list(fit_obj.peak_info[i].keys())[1]
            self.df.loc[self.n, 'area'] = fit_obj.peak_info[i][area]
            fwhm = list(fit_obj.peak_info[i].keys())[2]
            self.df.loc[self.n, 'fwhm'] = fit_obj.peak_info[i][fwhm]
            self.df.loc[self.n, 'best_fit'] = best_fit
            self.df.loc[self.n, 'redchi'] = redchi
            self.df.loc[self.n, 'bkg'] = comps[bkg]
            gauss = list(fit_obj.fit_result.eval_components().keys())[i+1]
            self.df.loc[self.n, 'gauss'] = comps[gauss]
            self.df.loc[self.n, 'uncertainty'] = uncertainty
            self.df.loc[self.n, 'bkg_type'] = bkg_type
            mean_err = list(fit_obj.peak_err[i].keys())[0]
            self.df.loc[self.n, 'mean_err'] = fit_obj.peak_err[i][mean_err]
            area_err = list(fit_obj.peak_err[i].keys())[1]
            self.df.loc[self.n, 'area_err'] = fit_obj.peak_err[i][area_err]
            fwhm_err = list(fit_obj.peak_err[i].keys())[2]
            self.df.loc[self.n, 'fwhm_err'] = fit_obj.peak_err[i][fwhm_err]
            self.df.loc[self.n, 'x_units'] = fit_obj.x_units
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
        
def auto_scan(search, xlst=None, bkglst=None, plot=False, save_to_hdf=False):
    '''
    scan a PeakSearch object either automatically or with given xrange and bkg
    lists

    Parameters
    ----------
    search : PeakSearch object
        from peaksearch.py.
    xlst : list, optional
        list of x ranges either in channels or energies
        (as defined by the search object). The default is None.
    bkglst : list, optional
        list of strings specifying the background type for each xrange in
        xlst (e.g. poly3). The default is None.
    plot : bool, optional
        plot each PeakFit object. The default is False.
    save_to_hdf : bool, optional
        Save to an hdf file (not yet implemented). The default is False.

    Returns
    -------
    fits : list
        PeakFit objects.

    '''
    #TODO: optimize bkg, save to hdf (using AddPeaks),
    # add checks (fit message, positive area, etc)
    fits = []
    if xlst is None and bkglst is None:
        ranges = auto_range(search, 2)
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
    elif xlst is not None and bkglst is not None:
        ranges = xlst
        bkgs = bkglst
        for rg, bk in zip(ranges, bkgs):
            fit0 = PeakFit(search, rg, bkg=bk)
            
            if plot:
                fit0.plot(plot_type="full")
            fits.append(fit0)
    return fits
   
    
    
    
    
    
    