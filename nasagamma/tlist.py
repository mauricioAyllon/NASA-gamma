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
    def __init__(self, fname, period, ebins=4096):
        self.period = period
        self.ebins = ebins
        self.tbins = 200
        self.trange = [0, period]
        self.data = self.load_data(fname)
        self.df = pd.DataFrame(data=self.data, columns=["channel", "ts", "dt"])
        plt.rc("font", size=14)
        plt.style.use("seaborn-darkgrid")

    def load_data(self, fname):
        data0 = np.genfromtxt(fname)
        dt0 = np.mod(data0[:, 1], self.period * 10) / 10
        dt = dt0.reshape((dt0.shape[0], 1))
        data = np.hstack((data0, dt))
        return data

    def filter_data(self, trange):
        self.df = self.df[(self.df["dt"] > trange[0]) & (self.df["dt"] < trange[1])]
        self.trange = trange

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
        # h, edg = np.histogram(self.data[:,2], bins=self.tbins)
        # ax.vlines(x=self.trange[0], ymin=0, ymax=h.max(), linestyles="dotted",
        #           lw=3, color="red")
        # ax.vlines(x=self.trange[1], ymin=0, ymax=h.max(), linestyles="dotted",
        #           lw=3, color="red")

    def plot_1(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        h, edg = np.histogram(
            self.df["channel"], bins=self.ebins, range=[0, self.ebins]
        )
        ax.plot(h, label=f"time range: {self.trange} us")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Counts")
        ax.legend()
