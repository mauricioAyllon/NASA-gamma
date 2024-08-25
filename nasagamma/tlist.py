"""
Tlist (event-by-event) reader for MultiScan, simple 2-column text file, and CAEN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Tlist:
    def __init__(self, df_raw, period):
        max_e_lst = np.array([2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16])
        self.df_raw = df_raw
        self.period = period
        self.trange = [0, period]
        self.energy_flag = False
        self.erange = None
        self.xd = None
        self.yd = None
        self.yerrd = None
        self.data = None
        self.df = None
        self.e_max = max_e_lst[max_e_lst > df_raw.channel.max()].min()
        self.dt_bins = 100
        self.tbins = 200
        self.ebins = 4096
        self.stack_data()
        # plt.rc("font", size=14)
        # plt.style.use("seaborn-v0_8-darkgrid")

    def stack_data(self):
        data0 = np.array(self.df_raw)
        dt0 = np.mod(data0[:, 1], self.period * 10) / 10
        dt = dt0.reshape((dt0.shape[0], 1))
        self.data = np.hstack((data0, dt))
        self.df = pd.DataFrame(data=self.data, columns=["channel", "ts", "dt"])

    def filter_tdata(self, trange, restore_df=True):
        if restore_df:
            self.restore_df()
        self.df = self.df[(self.df["dt"] > trange[0]) & (self.df["dt"] < trange[1])]
        self.trange = trange

    def filter_edata(self, erange):
        self.restore_df()
        self.df = self.df[
            (self.df["channel"] > erange[0]) & (self.df["channel"] < erange[1])
        ]
        self.erange = erange

    def restore_df(self):
        if self.energy_flag:
            self.df = pd.DataFrame(
                data=self.data, columns=["energy", "channel", "ts", "dt"]
            )
        else:
            self.df = pd.DataFrame(data=self.data, columns=["channel", "ts", "dt"])

    def change_period(self, new_period):
        dt_new = np.mod(self.data[:, 1], new_period * 10) / 10
        self.data[:, 2] = dt_new
        self.restore_df()
        self.period = new_period

    def hist_erg(self):
        if self.energy_flag:
            keyword = "energy"
        else:
            keyword = "channel"
        spect, edg = np.histogram(
            self.df[keyword], bins=self.ebins, range=[0, self.e_max]
        )
        x = (edg[1:] + edg[:-1]) / 2
        self.x = x
        self.spect = spect

    def hist_time(self):
        y, edg = np.histogram(self.df["dt"], bins=self.tbins)
        y_err = np.sqrt(y)
        x = (edg[1:] + edg[:-1]) / 2
        self.xd = x
        self.yd = y
        self.yerrd = y_err

    def plot_time_hist(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        ax.hist(
            # self.data[:, 2],
            self.df["dt"],
            bins=self.tbins,
            edgecolor="black",
            alpha=0.5,
            label=f"Total counts = {self.df.shape[0]}",
        )
        ax.set_xlabel("dt (us)")
        ax.set_ylabel("Counts")
        ax.legend()

    def plot_die_away(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        ax.errorbar(
            self.xd,
            self.yd,
            self.yerrd,
            fmt="o",
            linewidth=2,
            capsize=6,
            label=f"time range: {self.trange} us",
        )
        # ax.plot(x, y, label=f"time range: {self.trange} us")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Counts")
        ax.legend()

    def plot_vlines_t(self, color="red", ax=None):
        ax.axvspan(xmin=self.trange[0], xmax=self.trange[1], alpha=0.3, color=color)

    def plot_vlines_e(self, color="red", ax=None):
        ax.axvspan(xmin=self.erange[0], xmax=self.erange[1], alpha=0.3, color=color)

    def plot_spect_erg_all(self, x, y, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        ax.plot(x, y, label=f"time range: {self.trange} us")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Counts")
        ax.legend()

    def plot_spect_erg_range(
        self,
        plot_pulse=False,
        ax=None,
        pulse_width=100,
        pulse_delay=20,
        scale_pulse=2500,
    ):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        if self.erange is not None:
            ax.hist(self.df["dt"], bins=self.dt_bins, edgecolor="black")
            ax.set_title(f"Channel range: {self.erange}")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Counts")

            # include square pulse
            if plot_pulse:
                stp = np.ones(self.period)
                stp[pulse_width:] = 0
                stp_shift = np.roll(stp, pulse_delay)
                t_shift = np.linspace(0, self.period, num=stp_shift.shape[0])
                ax.plot(t_shift, stp_shift * scale_pulse, label="neutron pulse")
                ax.legend(loc="upper right")
