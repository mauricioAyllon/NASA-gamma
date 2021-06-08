# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:39:42 2021

@author: mauricio
Tlist reader
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Tlist:
    def __init__(self, fname, period):
        ebin_lst = [2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]
        self.period = period
        self.tbins = 200
        self.trange = [0, period]
        self.erange = None
        self.data = self.load_data(fname)
        self.ebins = min(ebin_lst, key=lambda x: abs(x - self.data[:, 0].max()))
        self.dt_bins = 100
        self.df = pd.DataFrame(data=self.data, columns=["channel", "ts", "dt"])
        self.spect_cts = 0
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")

    def load_data(self, fname):
        data0 = np.genfromtxt(fname)
        dt0 = np.mod(data0[:, 1], self.period * 10) / 10
        dt = dt0.reshape((dt0.shape[0], 1))
        data = np.hstack((data0, dt))
        return data

    def filter_data(self, trange):
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
        self.df = pd.DataFrame(data=self.data, columns=["channel", "ts", "dt"])

    def change_period(self, new_period):
        dt_new = np.mod(self.data[:, 1], new_period * 10) / 10
        self.data[:, 2] = dt_new
        self.restore_df()
        self.period = new_period

    def plot_hist(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        ax.hist(self.data[:, 2], bins=self.tbins, edgecolor="black")
        ax.set_xlabel("dt [us]")

    def plot_vlines(self, ax=None):
        ax.axvspan(xmin=self.trange[0], xmax=self.trange[1], alpha=0.1, color="red")

    def plot_1(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        self.spect_cts, edg = np.histogram(
            self.df["channel"], bins=self.ebins, range=[0, self.ebins]
        )
        ax.plot(self.spect_cts, label=f"time range: {self.trange} us")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Counts")
        ax.legend()

    def plot_spect_e(
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
            ax.set_xlabel("Time [us]")
            ax.set_ylabel("Counts")

            # include square pulse
            if plot_pulse:
                stp = np.ones(self.period)
                stp[pulse_width:] = 0
                stp_shift = np.roll(stp, pulse_delay)
                t_shift = np.linspace(0, self.period, num=stp_shift.shape[0])
                ax.plot(t_shift, stp_shift * scale_pulse, label="neutron pulse")
                ax.legend(loc="upper right")
