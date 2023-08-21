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
import natsort


class Diagnostics:
    def __init__(self, folder_path=None):
        plt.rc("font", size=14)
        plt.style.use("seaborn-v0_8-darkgrid")
        self.folder_path = folder_path
        self.load_data()
        self.means = []
        self.areas = []
        self.maxims = []
        self.fwhms = []
        self.area_errors = []

    def load_data(self):
        files = glob.glob(f"{self.folder_path}/*")
        files = natsort.natsorted(files)
        self.spects, count_rates, tot_counts, livetimes, realtimes = [], [], [], [], []
        for file in files:
            if file.lower()[-8:] == ".pha.txt":
                f = file_reader.ReadMultiscanPHA(file)
                self.spects.append(f.spect)
                count_rates.append(f.count_rate)
                tot_counts.append(f.counts)
                livetimes.append(f.live_time)
                realtimes.append(f.real_time)
            elif file.lower()[-4:] == ".txt" and file.lower()[-8:] != ".pha.txt":
                spect = file_reader.read_txt(file)
                self.spects.append(spect)
                cr = spect.counts.sum() / spect.livetime
                count_rates.append(cr)
                tot_counts.append(spect.counts.sum())
                livetimes.append(spect.livetime)
                realtimes.append(spect.realtime)
            elif file.lower()[-4:] == ".spe":
                spe = file_reader.ReadSPE(file)
                spect = sp.Spectrum(
                    counts=spe.counts, livetime=spe.live_time, realtime=spe.real_time
                )
                self.spects.append(spect)
                cr = spe.counts.sum() / spe.live_time
                count_rates.append(cr)
                tot_counts.append(spe.counts.sum())
                livetimes.append(spe.live_time)
                realtimes.append(spe.real_time)
            else:
                continue
        if self.spects[0].energies is None:
            self.e_flag = False
        else:
            self.e_flag = True
        self.runs = np.arange(0, len(count_rates), 1)
        self.count_rates = np.array(count_rates)
        self.tot_counts = np.array(tot_counts)
        self.livetimes = np.array(livetimes)
        self.realtimes = np.array(realtimes)
        t_abs = self.livetimes.cumsum()
        self.absolute_time = t_abs - t_abs[0]

    def calculate_integral(self, xmid, width):
        areas, area_errors, maxims = [], [], []
        for i, spect in enumerate(self.spects):
            if spect.energies is not None:
                ixe = (spect.energies >= xmid - width) & (
                    spect.energies <= xmid + width
                )
                erange = spect.channels[ixe]
                xrange = [erange[0], erange[-1]]
            else:
                xrange = [int(xmid - width), int(xmid + width)]
            # maximum channel (index)
            mx = xrange[0] + np.argmax(spect.counts[xrange[0] : xrange[1]])
            if self.e_flag:
                mx_e = spect.energies[mx]
                maxims.append(mx_e)
            else:
                maxims.append(mx)
            area = spect.counts[xrange[0] : xrange[1]].sum()
            areas.append(area)
            area_errors.append(np.sqrt(area))
        self.xmid = xmid
        self.width = width
        self.areas = np.array(areas)
        self.area_errors = np.array(area_errors)
        self.maxims = np.array(maxims)
        self.means = np.zeros(len(areas))
        self.fwhms = np.zeros(len(areas))

    @staticmethod
    def single_search(spect, xrange):
        search = None
        for rf in range(1, 20, 1):
            # print(f"Reference FWHM = {rf}")
            search = ps.PeakSearch(
                spectrum=spect,
                ref_x=420,
                ref_fwhm=rf,
                min_snr=2,
                xrange=[xrange[0], xrange[1]],
                method="km",
            )
            check_single = search.peaks_idx.shape[0]
            if check_single == 1:
                break
            else:
                search = None
        if search == None:
            for msnr in np.arange(0.1, 2, 0.2):
                # print(f"Minimum SNR = {msnr}")
                search = ps.PeakSearch(
                    spectrum=spect,
                    ref_x=420,
                    ref_fwhm=4,
                    min_snr=msnr,
                    xrange=[xrange[0], xrange[1]],
                    method="km",
                )
                check_single = search.peaks_idx.shape[0]
                if check_single == 1:
                    break
                else:
                    search = None
        return search

    def fit_peaks(self, xmid, width):
        means, areas, area_errors, fwhms, maxims = [], [], [], [], []
        for i, spect in enumerate(self.spects):
            mean, area, area_err, fwhm = 0, 0, 0, 0
            if spect.energies is not None:
                xnew = np.where(spect.energies >= xmid)[0][0]
            else:
                xnew = np.where(spect.channels >= xmid)[0][0]

            try:
                search = self.single_search(spect, xrange=[xmid - width, xmid + width])
            except:
                search = None
            if search is None:
                print(f"SEARCH: No fit available for run number {i}")
            else:
                try:
                    fit = pf.PeakFit(search=search, xrange=[xmid - width, xmid + width])
                    if (
                        fit.fit_result.message == "Fit succeeded."
                        and len(fit.peak_info) > 0
                        and fit.peak_info[0]["area1"] > 0
                    ):
                        info = fit.peak_info[0]
                        mean = info["mean1"]
                        area = info["area1"]
                        area_err = fit.peak_err[0]["area_err1"]
                        fwhm = info["fwhm1"]
                except:
                    print(f"FIT: No fit available for run number {i}")

            # max val
            mx = int(xnew - width) + np.argmax(
                spect.counts[xnew - width : xnew + width]
            )
            if spect.energies is not None:
                mx = np.where(spect.energies >= mx)[0][0]
            maxims.append(mx)
            means.append(mean)
            areas.append(area)
            area_errors.append(area_err)
            fwhms.append(fwhm)
        self.xmid = xmid
        self.width = width
        self.means = np.array(means)
        self.area_errors = np.array(area_errors)
        self.areas = np.array(areas)
        self.fwhms = np.array(fwhms)
        self.maxims = np.array(maxims)

    def combine_spects(self):
        counts = 0
        livetime = 0
        realtime = 0
        for i, s in enumerate(self.spects):
            counts += s.counts
            if s.livetime is not None:
                livetime += s.livetime
            else:
                print(f"No live time available for file number {i}")
            if s.realtime is not None:
                realtime += s.realtime
            else:
                print(f"No real time available for file number {i}")
        spe = sp.Spectrum(
            counts=counts,
            energies=s.energies,
            e_units=s.e_units,
            realtime=realtime,
            livetime=livetime,
            cps=s.cps,
        )
        return spe

    def create_data_frame(self):
        if (
            len(self.means) != 0
            and len(self.areas) != 0
            and len(self.maxims) != 0
            and len(self.fwhms) != 0
        ):
            units = self.spects[0].e_units
            CR = self.areas / self.livetimes
            CR_err = self.area_errors / self.livetimes
            data_np = np.array(
                [
                    self.absolute_time,
                    self.means,
                    self.areas,
                    CR,
                    CR_err,
                    self.maxims,
                    self.fwhms,
                ]
            ).T
            df = pd.DataFrame(
                columns=[
                    "Time (s)",
                    f"Mean ({units})",
                    "Counts",
                    "Count Rate (cps)",
                    "Count Rate Error",
                    "Max",
                    "FWHM",
                ],
                data=data_np,
            )
            return df

    def plot_counts(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.tot_counts, "o--", color="grey")
        ax.errorbar(
            x,
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
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        # ax.legend()
        plt.show()

    def plot_count_rates(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.count_rates, "o--", color="grey")
        ax.errorbar(
            x,
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
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel("Count rate (cts/s)", fontsize=14)
        # ax.legend()
        plt.show()

    def plot_spectra(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        for s in self.spects:
            s.plot(ax=ax)
        ax.legend().set_visible(False)

    def plot_measurement_times(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.realtimes, "o-", label="Real time")
        ax.plot(x, self.livetimes, "o-", label="Live time")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        ax.legend()

    def plot_livetime_percent(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(
            x,
            self.livetimes / self.realtimes * 100,
            "o-",
            color="C2",
            label="% Live time",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("% livetime")
        # ax.legend()

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

    def plot_fit_counts(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.areas, "o--", color="grey")
        ax.errorbar(
            x,
            self.areas,
            yerr=self.area_errors,
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
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        # ax.legend()
        plt.show()

    def plot_fit_count_rates(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.areas / self.livetimes, "o--", color="grey")
        ax.errorbar(
            x,
            self.areas / self.livetimes,
            yerr=self.area_errors / self.livetimes,
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
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel("Count rate (cts/s)", fontsize=14)
        # ax.legend()
        plt.show()

    def plot_centroid(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.means, "o--", color="C1", label="Centroid")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Centroid")
        # ax.legend()

    def plot_max(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        ax.plot(x, self.maxims, "o--", color="C2", label="Maximum channel")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Max channel")
        # ax.legend()

    def plot_fwhm(self, time=False, ax=None):
        if time:
            x = self.absolute_time
            x_label = "Time (s)"
        else:
            x = self.runs
            x_label = "Run number"
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()
        if np.any(self.means == 0):
            ax.plot(x, self.fwhms, "o--", color="C0", label="FWHM")
        else:
            ax.plot(
                x,
                self.fwhms / self.means * 100,
                "o--",
                color="C0",
                label="FWHM",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("% FWHM")
        # ax.legend()


# d = Diagnostics(r"C:\Users\mayllonu\Documents\NASA-GSFC\Technical\experiments\2022-09-DraGNS-Goddard\2022-09-23\HPGe-activation")
# d.plot_counts()
# d.plot_count_rates()
# #d.plot_spectra()
# d.plot_measurement_times()
# d.plot_livetime_percent()
# d.fit_peaks(xmid=2755, width=10)
# xmid = 847.2
# width = 10
# d.fit_peaks(xmid=xmid, width=width)
# d.plot_peaks()
# d.plot_max()
# d.plot_fit_counts()
# d.plot_centroid()


# def single_search(spect, xrange):
#     search = None
#     for rf in range(1,20,1):
#         #print(f"Reference FWHM = {rf}")
#         search = ps.PeakSearch(spectrum=spect, ref_x=420, ref_fwhm=rf,
#                                 min_snr=2, xrange=[xrange[0], xrange[1]],
#                                 method="km")
#         check_single = search.peaks_idx.shape[0]
#         if check_single == 1:
#             break
#         else:
#             search = None
#     if search == None:
#         for msnr in np.arange(0.1,2,0.2):
#             #print(f"Minimum SNR = {msnr}")
#             search = ps.PeakSearch(spectrum=spect, ref_x=420, ref_fwhm=4,
#                                     min_snr=msnr, xrange=[xrange[0], xrange[1]],
#                                     method="km")
#             check_single = search.peaks_idx.shape[0]
#             if check_single == 1:
#                 break
#             else:
#                 search = None
#     return search

# s1 = single_search(d.spects[0], xrange=[xmid-width, xmid+width])

# xrange = [xmid-width, xmid + width]
# search = ps.PeakSearch(spectrum=d.spects[0], ref_x=420, ref_fwhm=3,
#                         min_snr=5, xrange=[xrange[0], xrange[1]],
#                         method="km")
# search.plot_peaks()
