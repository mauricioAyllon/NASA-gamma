"""
Tools to display and interact with gamma ray spectra.
The Spectrum class is used for all subsequent spectrum manipulations 
including peak fitting, calibration, and line identification.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


class Spectrum:
    def __init__(
        self,
        counts=None,
        counts_err=None,
        energies=None,
        e_units=None,
        realtime=None,
        livetime=None,
        cps=False,
        acq_date=None,
        energy_cal=None,
        description=None,
        label=None,
    ):
        """
        Initialize the spectrum.

        Parameters
        ----------
        counts : numpy array, pandas series, or list.
            counts per bin or count rate. This is the only
            required input parameter.
        counts_err : numpy array, pandas series, or list, optional.
            1-sigma uncertainty per bin. If None, Poisson errors
            sqrt(max(counts, 1)) are assumed. The default is None.
        energies : numpy array, pandas series, or list. Optional
            energy values. The default is None.
        e_units : string, optional
            string of energy units e.g. "MeV". The default is None.
        realtime : int or float, optional
            real time of the measurement. The default is None.
        livetime : int or float, optional
            live time of the measurement. The default is None.
        cps : bool, optional
            if counts per second are used instead of counts,
            set this to True. The default is False.
        acq_date : string, optional
            aqcquisition date for record keeping. The default is None.
        energy_cal : string, optional
            energy calibration equation for record keeping. The default is None.
        description : string, optional
            experiment description for record keeping. The default is None.
        label : string, optional
            label experiment for plotting and record keeping. The default is None.

        Returns
        -------
        None.

        """
        self.e_units = e_units
        if counts is None:
            raise ValueError("counts must be specified and cannot be None")
        channels = np.arange(0, len(counts), 1)
        if energies is not None:
            self.energies = np.asarray(energies, dtype=float)
            self.x = self.energies
            if self.e_units is None:
                self.x_units = "Energy"
            else:
                self.x_units = f"Energy ({e_units})"
        else:
            self.energies = energies
            self.x = channels
            self.x_units = "Channels"

        self.counts = np.asarray(counts, dtype=float)
        self.channels = np.asarray(channels, dtype=int)

        # Per-bin uncertainty: use supplied array or default to Poisson sqrt(N),
        # with a floor of 1 to avoid zero weights in the fitter.
        if counts_err is None:
            self.counts_err = np.sqrt(np.maximum(self.counts, 1.0))
        else:
            self.counts_err = np.asarray(counts_err, dtype=float)

        self.realtime = realtime
        self.livetime = livetime
        self.cps = cps
        self.acq_date = acq_date
        self.energy_cal = energy_cal
        self.description = description
        if cps:
            self.y_label = "CPS"
        else:
            self.y_label = "Cts"
        self.label = label
        
    def __repr__(self):
        lt = f"{self.livetime:.3E} s" if self.livetime is not None else "N/A"
        return (
            f"Spectrum("
            f"label={self.label!r}, "
            f"channels={len(self.counts)}, "
            f"counts={self.counts.sum():.3E}, "
            f"livetime={lt}, "
            f"calibrated={self.energies is not None})"
        )

    def metadata(self):
        """
        Return metadata of the Spectrum instance as a dictionary.

        Returns
        -------
        dict
            Dictionary containing all metadata fields.
        """
        return {
            "label": self.label,
            "description": self.description,
            "acq_date": self.acq_date,
            "realtime": self.realtime,
            "livetime": self.livetime,
            "cps": self.cps,
            "e_units": self.e_units,
            "energy_cal": self.energy_cal,
            "n_channels": len(self.counts),
            "total_counts": self.counts.sum(),
        }

    def copy(self):
        """
        Return a deep copy of the Spectrum object.

        Returns
        -------
        Spectrum
            A new Spectrum object with copied data and metadata.
        """
        return Spectrum(
            counts=self.counts.copy(),
            counts_err=self.counts_err.copy(),
            energies=self.energies.copy() if self.energies is not None else None,
            e_units=self.e_units,
            realtime=self.realtime,
            livetime=self.livetime,
            cps=self.cps,
            acq_date=self.acq_date,
            energy_cal=self.energy_cal,
            description=self.description,
            label=self.label,
        )

    def smooth(self, num=4):
        """
        Parameters
        ----------
        num : integer, optional
            number of data points for averaging. The default is 4.

        Returns
        -------
        numpy array
            moving average of counts. Modifies spectrum in place.

        """
        df = pd.DataFrame(data=self.counts, columns=["cts"])
        mav = df.cts.rolling(window=num, center=True).mean()
        mav = mav.bfill().ffill().fillna(0)
        counts_mav = np.array(mav)
        counts_mav_scaled = counts_mav / counts_mav.sum() * self.counts.sum()
        self.counts = counts_mav_scaled
        # Recompute default Poisson errors after smoothing
        self.counts_err = np.sqrt(np.maximum(self.counts, 1.0))

    def rebin(self, by=2):
        """
        Rebins data by adding 'by' adjacent bins at a time.
        Errors are propagated in quadrature.
        """
        new_size = int(self.counts.shape[0] / by)
        new_cts = self.counts.reshape((new_size, -1)).sum(axis=1)
        # errors add in quadrature across combined bins
        new_err = np.sqrt((self.counts_err**2).reshape((new_size, -1)).sum(axis=1))
        self.counts = new_cts
        self.counts_err = new_err
        self.channels = np.arange(0, len(new_cts), 1)

        if self.energies is not None:
            new_erg = self.energies.reshape((new_size, -1)).mean(axis=1)
            self.energies = new_erg
            self.x = new_erg
        else:
            self.x = self.channels

    def gain_shift(self, by=0, energy=False):
        """
        Slide spectrum left or right by 'by' number of channels or energy values.
        If positive shift, replace low energy values by zeroes. If negative shift,
        replace high energy values by the value in the last bin.

        Parameters
        ----------
        by : integer, optional
            Number of channels or energy values to shift the spectrum by. The default is 0.
        energy : double, optional
            Set to True if shifting by energy values. The default is False.

        Returns
        -------
        None.

        """
        if energy and self.energies is not None:
            cal = np.diff(self.energies)[0]
            by = round(by / cal)
        by = int(by)
        if by > 0:
            # positive roll — replace rolled low-energy counts with zeros
            self.counts = np.roll(self.counts, shift=by)
            self.counts[0:by] = 0
            self.counts_err = np.roll(self.counts_err, shift=by)
            self.counts_err[0:by] = 1.0
        elif by < 0:
            # negative roll — replace rolled high-energy counts with last value
            self.counts = np.roll(self.counts, shift=by)
            self.counts[by:] = self.counts[by - 1]
            self.counts_err = np.roll(self.counts_err, shift=by)
            self.counts_err[by:] = self.counts_err[by - 1]
        else:
            print("No shift applied")
            
    def normalize(self, by="counts"):
        """
        Normalize spectrum in place.
 
        Parameters
        ----------
        by : str, optional
            Normalization method. Either 'counts' to divide by total counts,
            or 'livetime' to divide by livetime. The default is 'counts'.
 
        Returns
        -------
        None.
 
        """
        if by == "counts":
            total = self.counts.sum()
            if total == 0:
                raise ValueError("Cannot normalize: total counts is zero.")
            self.counts = self.counts / total
            self.counts_err = self.counts_err / total
        elif by == "livetime":
            if self.livetime is None:
                raise ValueError("Cannot normalize by livetime: livetime is not set.")
            self.counts = self.counts / self.livetime
            self.counts_err = self.counts_err / self.livetime
        else:
            raise ValueError(f"Unknown normalization method '{by}'. Use 'counts' or 'livetime'.")
            
    def roi_counts(self, x1, x2):
        """
        Sum counts between two x values (channels or energies).
 
        Parameters
        ----------
        x1 : float
            Lower bound of the region of interest.
        x2 : float
            Upper bound of the region of interest.
 
        Returns
        -------
        dict with keys:
            'sum'         : total counts in the ROI.
            'uncertainty' : 1-sigma uncertainty propagated in quadrature.
            'n_bins'      : number of bins in the ROI.
            'x1'          : actual lower bound used.
            'x2'          : actual upper bound used.
        """
        if x1 >= x2:
            raise ValueError("x1 must be less than x2.")
        mask = (self.x >= x1) & (self.x <= x2)
        if not mask.any():
            raise ValueError(f"No bins found in range [{x1}, {x2}].")
        roi_counts = self.counts[mask]
        roi_err = self.counts_err[mask]
        return {
            "sum": roi_counts.sum(),
            "uncertainty": np.sqrt((roi_err**2).sum()),
            "n_bins": int(mask.sum()),
            "x1": self.x[mask][0],
            "x2": self.x[mask][-1],
        }

    def replace_neg_vals(self):
        """
        Replaces negative values in spectrum with 1/10th of the minimum
        positive count value.

        Returns
        -------
        None.

        """
        y0_min = np.amin(self.counts[self.counts > 0.0])
        self.counts[self.counts < 0.0] = y0_min * 1e-1

    def gaussian_energy_broadening(self, fwhm_func, nsigmas=3, random_seed=None):
          """
          Apply Gaussian energy broadening with preserved Poisson noise characteristics.
     
          Parameters
          ----------
          fwhm_func : callable
              Function taking energy and returning FWHM at that energy.
          nsigmas : int, optional
              Number of sigma to span in the Gaussian kernel. The default is 3.
          random_seed : int or None
              Optional seed for reproducibility.
     
          Returns
          -------
          None.
     
          """
          rng = np.random.default_rng(random_seed)
          counts = self.counts
          x = self.x
          broadened_counts = np.zeros_like(counts, dtype=float)
          for i, count in enumerate(counts):
              if count <= 0:
                  continue
              sampled_count = rng.poisson(count)
              if sampled_count == 0:
                  continue
              E_i = x[i]
              fwhm = fwhm_func(E_i)
              sigma = fwhm / 2.355
              if sigma <= 0:
                  broadened_counts[i] += sampled_count
                  continue
              E_min = E_i - nsigmas * sigma
              E_max = E_i + nsigmas * sigma
              mask = (x >= E_min) & (x <= E_max)
              E_window = x[mask]
              idx_window = np.where(mask)[0]
              kernel = np.exp(-0.5 * ((E_window - E_i) / sigma) ** 2)
              kernel /= kernel.sum()
              redistributed = rng.multinomial(sampled_count, kernel)
              broadened_counts[idx_window] += redistributed
          self.counts = broadened_counts
          self.counts_err = np.sqrt(np.maximum(self.counts, 1.0))

    @staticmethod
    def fwhm_HPGe_example(E):
        """FWHM example for HPGe in keV."""
        return 0.05 * np.sqrt(E) + 0.001 * E

    @staticmethod
    def fwhm_LaBr_example(E):
        """FWHM example for LaBr in MeV."""
        a = -0.02
        b = 0.044
        c = 0.117
        return a + b * np.sqrt(E + c * E ** 2)

    def remove_calibration(self):
        """
        Remove energy calibration and reinitialize Spectrum object.

        Returns
        -------
        None.

        """
        self.__init__(
            counts=self.counts,
            counts_err=self.counts_err,
            energies=None,
            e_units=None,
            realtime=self.realtime,
            livetime=self.livetime,
            cps=self.cps,
            acq_date=self.acq_date,
            energy_cal=None,
            description=self.description,
            label=self.label,
        )

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    def _check_compatible(self, other):
        """
        Check that two spectra can be combined bin-by-bin.

        Raises
        ------
        ValueError
            If spectra have different number of bins or mismatched x-axes.
        """
        if len(self.counts) != len(other.counts):
            raise ValueError(
                f"Spectra have different number of bins: {len(self.counts)} vs {len(other.counts)}"
            )
        if not np.array_equal(self.x, other.x):
            raise ValueError("Spectra have mismatched x-axes and cannot be combined")

    # ------------------------------------------------------------------
    # Arithmetic operators  — errors propagated throughout
    # ------------------------------------------------------------------

    def __add__(self, other):
        """
        Add two spectra. Returns a new Spectrum object.
        Counts and errors are added in quadrature.
        Livetimes and realtimes are summed if both are present.
        """
        self._check_compatible(other)
        new_counts = self.counts + other.counts
        new_err = np.sqrt(self.counts_err**2 + other.counts_err**2)
        new_livetime = (
            self.livetime + other.livetime
            if (self.livetime is not None and other.livetime is not None)
            else None
        )
        new_realtime = (
            self.realtime + other.realtime
            if (self.realtime is not None and other.realtime is not None)
            else None
        )
        return Spectrum(
            counts=new_counts,
            counts_err=new_err,
            energies=self.energies.copy() if self.energies is not None else None,
            e_units=self.e_units,
            realtime=new_realtime,
            livetime=new_livetime,
            cps=self.cps,
        )

    def __sub__(self, other):
        """
        Subtract two spectra. Returns a new Spectrum object.
        Errors are propagated in quadrature: σ₃ = sqrt(σ₁² + σ₂²).

        For background subtraction with different livetimes, scale the
        background first:
            scale = spe1.livetime / spe2.livetime
            spe3 = spe1 - spe2 * scale
        The scaled __mul__ will carry the scaled error automatically,
        giving σ₃ᵢ = sqrt(N1ᵢ + scale² · N2ᵢ) per bin.
        """
        self._check_compatible(other)
        new_counts = self.counts - other.counts
        new_err = np.sqrt(self.counts_err**2 + other.counts_err**2)
        return Spectrum(
            counts=new_counts,
            counts_err=new_err,
            energies=self.energies.copy() if self.energies is not None else None,
            e_units=self.e_units,
            cps=self.cps,
        )

    def __mul__(self, scalar_or_array):
        """
        Multiply spectrum counts by a scalar or numpy array.
        Returns a new Spectrum object.
        Error scales by the same factor: σ_new = |k| · σ.
        """
        new_counts = self.counts * scalar_or_array
        new_err = self.counts_err * np.abs(scalar_or_array)
        return Spectrum(
            counts=new_counts,
            counts_err=new_err,
            energies=self.energies.copy() if self.energies is not None else None,
            e_units=self.e_units,
            realtime=self.realtime,
            livetime=self.livetime,
            cps=self.cps,
        )

    def __rmul__(self, scalar_or_array):
        """
        Right multiply — allows expressions like 2 * spectrum.
        """
        return self.__mul__(scalar_or_array)

    def __truediv__(self, scalar_or_array):
        """
        Divide spectrum counts by a scalar or numpy array.
        Returns a new Spectrum object.
        Error scales inversely: σ_new = σ / |k|.
        """
        new_counts = self.counts / scalar_or_array
        new_err = self.counts_err / np.abs(scalar_or_array)
        return Spectrum(
            counts=new_counts,
            counts_err=new_err,
            energies=self.energies.copy() if self.energies is not None else None,
            e_units=self.e_units,
            realtime=self.realtime,
            livetime=self.livetime,
            cps=self.cps,
        )

    def __radd__(self, other):
        """
        Right add — allows use of sum() on a list of Spectrum objects.
        Handles the sum() initialization value of 0.
        """
        if other == 0:
            return self.copy()
        return self.__add__(other)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_csv(self, fileName):
        """
        Save spectrum to a .csv file. This file format does not include metadata.

        Parameters
        ----------
        fileName : string
            file name or path of where to save the file.

        Returns
        -------
        None.

        """
        if self.energies is not None:
            cols = ["counts", "counts_err", f"{self.x_units}"]
            data = np.array((self.counts, self.counts_err, self.x)).T
        else:
            cols = ["counts", "counts_err"]
            data = np.column_stack((self.counts, self.counts_err))

        df = pd.DataFrame(data=data, columns=cols)
        if fileName[-4:] == ".csv":
            df.to_csv(f"{fileName}", index=False)
        else:
            df.to_csv(f"{fileName}.csv", index=False)

    def to_txt(self, fileName):
        """
        Save spectrum to a .txt file. This file format includes metadata
        as headers.

        Parameters
        ----------
        fileName : string
            file name or path of where to save the file.

        Returns
        -------
        None.

        """
        if self.acq_date is None:
            self.acq_date = datetime.date.today()
        if fileName[-4:] == ".txt":
            file_txt = fileName
        else:
            file_txt = fileName + ".txt"
        with open(file_txt, "w") as f:
            f.write(f"Description: {self.description}\n")
            f.write(f"Label: {self.label}\n")
            f.write(f"Date created: {self.acq_date}\n")
            f.write(f"Real time (s): {self.realtime}\n")
            f.write(f"Live time (s): {self.livetime}\n")
            f.write(f"Energy calibration: {self.energy_cal}\n")
            if self.energies is None:
                f.write("counts,counts_err\n")
                for cts, err in zip(self.counts, self.counts_err):
                    f.write(f"{cts},{err}\n")
            else:
                f.write(f"counts,counts_err,{self.x_units}\n")
                for cts, err, erg in zip(self.counts, self.counts_err, self.energies):
                    f.write(f"{cts},{err},{erg}\n")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None, scale="log", fontsize=14):
        """
        Plot spectrum object using channels and energies (if not None).
 
        Parameters
        ----------
        scale : string, optional
            Either 'linear' or 'log'. The default is 'log'.
        ax : matplotlib Axes, optional
            Axes to plot on. The default is None (creates new figure).
        fontsize : int, optional
            Font size. The default is 14.
 
        Returns
        -------
        matplotlib Axes
 
        """
        plt.rc("font", size=fontsize)
        plt.style.use("seaborn-v0_8-darkgrid")
 
        if ax is None:
           fig = plt.figure(figsize=(10, 6))
           fig.patch.set_alpha(0.3)  # set background transparent
           ax = fig.add_subplot()
 
        integral = round(self.counts.sum())
        if self.label is None:
             if self.livetime is None:
                 lt = "Livetime = N/A"
             else:
                 lt = f"Livetime = {self.livetime:.3E} s"
             label = f"Total counts = {integral:.3E}\n{lt}"
        else:
             label = self.label
       
        ax.fill_between(self.x, 0, self.counts, alpha=0.2, color="C1", step="pre")
        ax.plot(self.x, self.counts, drawstyle="steps", alpha=0.7, label=label)
        ax.set_yscale(scale)
        ax.set_xlabel(self.x_units, fontsize=fontsize)
        ax.set_ylabel(self.y_label, fontsize=fontsize)
        ax.legend()
        return ax


def plot_overlay(spectra, scale="log", fontsize=14, ax=None, colors=None):
    """
    Plot multiple Spectrum objects on the same axes.
 
    Parameters
    ----------
    spectra : list of Spectrum
        Spectra to overlay.
    scale : string, optional
        Either 'linear' or 'log'. The default is 'log'.
    fontsize : int, optional
        Font size. The default is 14.
    ax : matplotlib Axes, optional
        Axes to plot on. The default is None.
    colors : list of str, optional
        Colors for each spectrum. The default is None (uses matplotlib defaults).
 
    Returns
    -------
    matplotlib Axes
 
    """
    if not spectra:
        raise ValueError("spectra list cannot be empty")
    plt.rc("font", size=fontsize)
    plt.style.use("seaborn-v0_8-darkgrid")
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_alpha(0.3)
        ax = fig.add_subplot()
    for i, spec in enumerate(spectra):
        color = colors[i] if colors is not None else f"C{i}"
        label = spec.label if spec.label is not None else f"Spectrum {i + 1}"
        ax.fill_between(spec.x, 0, spec.counts, alpha=0.2, color=color, step="pre")
        ax.plot(spec.x, spec.counts, drawstyle="steps", alpha=0.7,
                label=label, color=color)
    ax.set_yscale(scale)
    ax.set_xlabel(spectra[0].x_units, fontsize=fontsize)
    ax.set_ylabel(spectra[0].y_label, fontsize=fontsize)
    ax.legend()  
    return ax