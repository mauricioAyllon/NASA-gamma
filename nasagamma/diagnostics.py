"""
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader
import glob


class Diagnostics:
    def __init__(self, folder_path=None):
        plt.rc("font", size=14)
        plt.style.use("seaborn-v0_8-darkgrid")
        self.folder_path = folder_path
        self.load_data()

    def load_data(self):
        files = glob.glob(f"{self.folder_path}/*")
        self.spects, count_rates, tot_counts, livetimes, realtimes = [], [], [], [], []
        for file in files:
            if file.lower()[-8:] == ".pha.txt":
                f = file_reader.ReadMultiscanPHA(file)
                self.spects.append(f.spect)
                count_rates.append(f.count_rate)
                tot_counts.append(f.counts)
                livetimes.append(f.live_time)
                realtimes.append(f.real_time)
            elif file.lower()[-3:] == "spe":
                spe = file_reader.ReadSPE(file)
                spect = sp.Spectrum(counts=spe.counts)
                self.spects.append(spect)
                cr = spe.counts.sum() / spe.live_time
                count_rates.append(cr)
                tot_counts.append(spe.counts.sum())
                livetimes.append(spe.live_time)
                realtimes.append(spe.real_time)
            else:
                continue
        self.runs = np.arange(0, len(count_rates), 1)
        self.count_rates = np.array(count_rates)
        self.tot_counts = np.array(tot_counts)
        self.livetimes = np.array(livetimes)
        self.realtimes = np.array(realtimes)

    def fit_peaks(self, xmid, width):
        means, areas, fwhms, maxims = [], [], [], []
        for i, spect in enumerate(self.spects):
            if spect.energies is not None:
                self.e_flag = True
                xnew = np.where(spect.energies >= xmid)[0][0]
            else:
                self.e_flag = False
                xnew = np.where(spect.channels >= xmid)[0][0]
            search = ps.PeakSearch(spect, 420, 20, min_snr=1e6, method="scipy")
            search.peaks_idx = np.append(search.peaks_idx, xnew)
            fwhm_guess_new = search.fwhm(xnew)
            search.fwhm_guess = np.append(search.fwhm_guess, fwhm_guess_new)
            fit = pf.PeakFit(search=search, xrange=[xmid - width, xmid + width])
            if fit.fit_result.message == "Fit succeeded." and len(fit.peak_info) > 0:
                info = fit.peak_info[0]
                mean = info["mean1"]
                area = info["area1"]
                fwhm = info["fwhm1"]
            else:
                print(f"No fit available for run number {i}")
                mean, area, fwhm = 0, 0, 0
            mx = np.argmax(spect.counts[xnew - width : xnew + width])
            if spect.energies is not None:
                mx = np.where(spect.energies >= mx)[0][0]
            maxims.append(mx)
            means.append(mean)
            areas.append(area)
            fwhms.append(fwhm)
        self.xmid = xmid
        self.width = width
        self.means = np.array(means)
        self.areas = np.array(areas)
        self.fwhms = np.array(fwhms)
        self.maxims = np.array(maxims)

    def combine_spects(self):
        counts = 0
        livetime = 0
        for s in self.spects:
            counts += s.counts
            livetime += s.livetime
        spe = sp.Spectrum(
            counts=counts,
            energies=s.energies,
            e_units=s.e_units,
            livetime=livetime,
            cps=s.cps,
        )
        return spe

    def plot_counts(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.tot_counts, "o--", color="grey")
        ax.errorbar(
            self.runs,
            self.tot_counts,
            yerr=np.sqrt(self.tot_counts),
            ecolor="black",
            elinewidth=3,
            capsize=12,
            capthick=2,
            marker="s",
            markersize=5,
            mfc="C1",
            mec="C1",
            ls=" ",
            lw=2,
            label="Total counts",
        )
        ax.set_xlabel("Run number", fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax.legend()
        plt.show()

    def plot_count_rates(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.count_rates, "o--", color="grey")
        ax.errorbar(
            self.runs,
            self.count_rates,
            yerr=np.sqrt(self.count_rates * self.livetimes) / self.livetimes,
            ecolor="black",
            elinewidth=3,
            capsize=12,
            capthick=2,
            marker="s",
            markersize=5,
            mfc="C2",
            mec="C2",
            ls=" ",
            lw=2,
            label="Count rates",
        )
        ax.set_xlabel("Run number", fontsize=14)
        ax.set_ylabel("Count rate (cts/s)", fontsize=14)
        ax.legend()
        plt.show()

    def plot_spectra(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        for s in self.spects:
            s.plot(ax=ax)
        ax.legend().set_visible(False)

    def plot_measurement_times(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.realtimes, "o-", label="Real time")
        ax.plot(self.runs, self.livetimes, "o-", label="Live time")
        ax.set_xlabel("Run number")
        ax.set_ylabel("Time (s)")
        ax.legend()

    def plot_livetime_percent(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(
            self.runs,
            self.livetimes / self.realtimes * 100,
            "o-",
            color="C2",
            label="% Live time",
        )
        ax.set_xlabel("Run number")
        ax.set_ylabel("% livetime")
        ax.legend()

    def plot_peaks(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        xrange = [int(self.xmid - self.width), int(self.xmid + self.width)]
        if self.e_flag:
            idx0 = np.where(self.spects[0].energies >= xrange[0])[0][0]
            idx1 = np.where(self.spects[0].energies >= xrange[1])[0][0]
            x = self.spects[0].energies[idx0:idx1]
        else:
            idx0 = xrange[0]
            idx1 = xrange[1]
            x = self.spects[0].channels[idx0:idx1]
        for s in self.spects:
            ax.plot(x, s.counts[idx0:idx1])
            ax.set_xlabel(s.x_units)
            ax.set_ylabel("Counts")

    def plot_fit_counts(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.areas, "o--", color="grey")
        ax.errorbar(
            self.runs,
            self.areas,
            yerr=np.sqrt(self.areas),
            ecolor="black",
            elinewidth=3,
            capsize=12,
            capthick=2,
            marker="s",
            markersize=5,
            mfc="C2",
            mec="C2",
            ls=" ",
            lw=2,
            label="Counts",
        )
        ax.set_xlabel("Run number", fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax.legend()
        plt.show()

    def plot_fit_count_rates(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.areas / self.livetimes, "o--", color="grey")
        ax.errorbar(
            self.runs,
            self.areas / self.livetimes,
            yerr=np.sqrt(self.areas) / self.livetimes,
            ecolor="black",
            elinewidth=3,
            capsize=12,
            capthick=2,
            marker="s",
            markersize=5,
            mfc="C3",
            mec="C3",
            ls=" ",
            lw=2,
            label="Count rates",
        )
        ax.set_xlabel("Run number", fontsize=14)
        ax.set_ylabel("Count rate (cts/s)", fontsize=14)
        ax.legend()
        plt.show()

    def plot_centroid(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.means, "o--", color="C1", label="Centroid")
        ax.set_xlabel("Run number")
        ax.set_ylabel("Centroid")
        ax.legend()

    def plot_max(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(self.runs, self.maxims, "o--", color="C2", label="Maximum channel")
        ax.set_xlabel("Run number")
        ax.set_ylabel("Max channel")
        ax.legend()

    def plot_fwhm(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(
            self.runs, self.fwhms / self.means * 100, "o--", color="C0", label="FWHM"
        )
        ax.set_xlabel("Run number")
        ax.set_ylabel("% FWHM")
        ax.legend()


# d = Diagnostics(r"C:\Users\mayllonu\Documents\NASA-GSFC\Technical\experiments\2022-09-DraGNS-Goddard\2022-09-23\HPGe-activation")
# d.plot_counts()
# d.plot_count_rates()
# #d.plot_spectra()
# d.plot_measurement_times()
# d.plot_livetime_percent()
# d.fit_peaks(xmid=2755, width=10)
# d.plot_peaks()
# d.plot_max()
