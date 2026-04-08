"""
Pytest tests for the PeakSearch class.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nasagamma.spectrum import Spectrum
from nasagamma.peaksearch import PeakSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spectrum(n_channels, peak_positions, peak_heights, fwhm_func, 
                  bkg_intercept=50, bkg_slope=0.05, with_energy=False, seed=42):
    """
    Create a synthetic spectrum with Gaussian peaks on a linear background.

    Parameters
    ----------
    n_channels : int
        number of channels.
    peak_positions : list of int
        true peak centroids in channels.
    peak_heights : list of float
        peak amplitudes.
    fwhm_func : callable
        function returning FWHM at a given channel.
    bkg_intercept : float
        background intercept.
    bkg_slope : float
        background slope per channel.
    with_energy : bool
        if True, attach a linear energy calibration.
    seed : int
        random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    channels = np.arange(n_channels, dtype=float)
    background = bkg_intercept + bkg_slope * channels
    counts = background.copy()
    for pos, height in zip(peak_positions, peak_heights):
        fwhm = fwhm_func(pos)
        sigma = fwhm / 2.355
        counts += height * np.exp(-0.5 * ((channels - pos) / sigma) ** 2)
    counts = rng.poisson(counts).astype(float)
    if with_energy:
        energies = 0.5 + 0.72 * channels  # keV
        return Spectrum(counts=counts, energies=energies, e_units="keV")
    return Spectrum(counts=counts)


# ---------------------------------------------------------------------------
# FWHM functions
# ---------------------------------------------------------------------------

def fwhm_hpge(x):
    """HPGe resolution ~0.3% at high energy."""
    return 0.5 + 0.065 * np.sqrt(x + 1)


def fwhm_scint(x):
    """Scintillator resolution ~3% at high energy."""
    return 1.0 + 0.5 * np.sqrt(x + 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PEAK_POSITIONS = [100, 250, 400]
PEAK_HEIGHTS = [5000, 8000, 3000]

@pytest.fixture
def spec_hpge():
    """HPGe spectrum without energy calibration."""
    return make_spectrum(
        n_channels=512,
        peak_positions=PEAK_POSITIONS,
        peak_heights=PEAK_HEIGHTS,
        fwhm_func=fwhm_hpge,
    )


@pytest.fixture
def spec_hpge_cal():
    """HPGe spectrum with energy calibration."""
    return make_spectrum(
        n_channels=512,
        peak_positions=PEAK_POSITIONS,
        peak_heights=PEAK_HEIGHTS,
        fwhm_func=fwhm_hpge,
        with_energy=True,
    )


@pytest.fixture
def spec_scint():
    """Scintillator spectrum without energy calibration."""
    return make_spectrum(
        n_channels=512,
        peak_positions=PEAK_POSITIONS,
        peak_heights=PEAK_HEIGHTS,
        fwhm_func=fwhm_scint,
    )


@pytest.fixture
def spec_scint_cal():
    """Scintillator spectrum with energy calibration."""
    return make_spectrum(
        n_channels=512,
        peak_positions=PEAK_POSITIONS,
        peak_heights=PEAK_HEIGHTS,
        fwhm_func=fwhm_scint,
        with_energy=True,
    )


@pytest.fixture
def ps_hpge_km(spec_hpge):
    """PeakSearch on HPGe spectrum using km method."""
    return PeakSearch(
        spectrum=spec_hpge,
        ref_x=250,
        ref_fwhm=fwhm_hpge(250),
        fwhm_at_0=0.5,
        min_snr=3,
        method="km",
    )


@pytest.fixture
def ps_hpge_fast(spec_hpge):
    """PeakSearch on HPGe spectrum using fast method."""
    return PeakSearch(
        spectrum=spec_hpge,
        ref_x=250,
        ref_fwhm=fwhm_hpge(250),
        fwhm_at_0=0.5,
        min_snr=3,
        method="fast",
    )


@pytest.fixture
def ps_scint_km(spec_scint):
    """PeakSearch on scintillator spectrum using km method."""
    return PeakSearch(
        spectrum=spec_scint,
        ref_x=250,
        ref_fwhm=fwhm_scint(250),
        fwhm_at_0=1.0,
        min_snr=3,
        method="km",
    )


@pytest.fixture
def ps_scint_fast(spec_scint):
    """PeakSearch on scintillator spectrum using fast method."""
    return PeakSearch(
        spectrum=spec_scint,
        ref_x=250,
        ref_fwhm=fwhm_scint(250),
        fwhm_at_0=1.0,
        min_snr=3,
        method="fast",
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_spectrum_type(self, spec_hpge):
        with pytest.raises(TypeError, match="spectrum must be a Spectrum object"):
            PeakSearch(spectrum=np.array([1, 2, 3]), ref_x=100, ref_fwhm=5)

    def test_invalid_ref_x_negative(self, spec_hpge):
        with pytest.raises(ValueError, match="ref_x must be a positive number"):
            PeakSearch(spectrum=spec_hpge, ref_x=-10, ref_fwhm=5)

    def test_invalid_ref_x_zero(self, spec_hpge):
        with pytest.raises(ValueError, match="ref_x must be a positive number"):
            PeakSearch(spectrum=spec_hpge, ref_x=0, ref_fwhm=5)

    def test_invalid_ref_fwhm(self, spec_hpge):
        with pytest.raises(ValueError, match="ref_fwhm must be a positive number"):
            PeakSearch(spectrum=spec_hpge, ref_x=100, ref_fwhm=-1)

    def test_invalid_fwhm_at_0(self, spec_hpge):
        with pytest.raises(ValueError, match="fwhm_at_0 must be a positive number"):
            PeakSearch(spectrum=spec_hpge, ref_x=100, ref_fwhm=5, fwhm_at_0=-1)

    def test_invalid_min_snr(self, spec_hpge):
        with pytest.raises(ValueError, match="min_snr must be a positive number"):
            PeakSearch(spectrum=spec_hpge, ref_x=100, ref_fwhm=5, min_snr=-1)

    def test_invalid_method(self, spec_hpge):
        with pytest.raises(ValueError, match="method must be one of"):
            PeakSearch(spectrum=spec_hpge, ref_x=100, ref_fwhm=5, method="invalid")

    def test_invalid_xrange_length(self, spec_hpge):
        with pytest.raises(ValueError, match="xrange must have exactly 2 elements"):
            PeakSearch(spectrum=spec_hpge, ref_x=100, ref_fwhm=5, xrange=[50, 200, 300])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_attributes_initialized_none(self, ps_hpge_km):
        """All decomposition attributes should be set after km run."""
        assert ps_hpge_km.peaks_idx is not None
        assert ps_hpge_km.snr is not None
        assert ps_hpge_km.fwhm_guess is not None

    def test_decomposition_none_for_fast(self, ps_hpge_fast):
        """Decomposition components should be None for fast method."""
        assert ps_hpge_fast.peak_plus_bkg is None
        assert ps_hpge_fast.bkg is None
        assert ps_hpge_fast.signal is None
        assert ps_hpge_fast.noise is None

    def test_channel_idx_full_without_xrange(self, ps_hpge_km, spec_hpge):
        assert ps_hpge_km.channel_idx.sum() == len(spec_hpge.channels)

    def test_channel_idx_restricted_with_xrange(self, spec_hpge):
        ps = PeakSearch(
            spectrum=spec_hpge,
            ref_x=250,
            ref_fwhm=fwhm_hpge(250),
            min_snr=3,
            xrange=[50, 350],
        )
        assert ps.channel_idx.sum() < len(spec_hpge.channels)


# ---------------------------------------------------------------------------
# Peak detection — km method
# ---------------------------------------------------------------------------

class TestPeakDetectionKm:
    def test_finds_all_peaks_hpge(self, ps_hpge_km):
        """All three true peaks should be found within tolerance."""
        for true_pos in PEAK_POSITIONS:
            assert any(abs(ps_hpge_km.peaks_idx - true_pos) <= 5), \
                f"Peak at {true_pos} not found in {ps_hpge_km.peaks_idx}"

    def test_finds_all_peaks_scint(self, ps_scint_km):
        """All three true peaks should be found within tolerance."""
        for true_pos in PEAK_POSITIONS:
            assert any(abs(ps_scint_km.peaks_idx - true_pos) <= 5), \
                f"Peak at {true_pos} not found in {ps_scint_km.peaks_idx}"

    def test_snr_positive(self, ps_hpge_km):
        assert np.all(ps_hpge_km.snr >= 0)

    def test_snr_length_matches_spectrum(self, ps_hpge_km, spec_hpge):
        assert len(ps_hpge_km.snr) == len(spec_hpge.counts)

    def test_fwhm_guess_length_matches_peaks(self, ps_hpge_km):
        assert len(ps_hpge_km.fwhm_guess) == len(ps_hpge_km.peaks_idx)

    def test_decomposition_computed(self, ps_hpge_km):
        assert ps_hpge_km.peak_plus_bkg is not None
        assert ps_hpge_km.bkg is not None
        assert ps_hpge_km.signal is not None
        assert ps_hpge_km.noise is not None

    def test_with_xrange(self, spec_hpge):
        ps = PeakSearch(
            spectrum=spec_hpge,
            ref_x=250,
            ref_fwhm=fwhm_hpge(250),
            min_snr=3,
            xrange=[50, 350],
            method="km",
        )
        assert len(ps.peaks_idx) > 0
        assert all(50 <= p <= 350 for p in ps.peaks_idx)

    def test_with_energy_calibration(self, spec_hpge_cal):
        ps = PeakSearch(
            spectrum=spec_hpge_cal,
            ref_x=250,
            ref_fwhm=fwhm_hpge(250),
            min_snr=3,
            method="km",
        )
        assert len(ps.peaks_idx) > 0


# ---------------------------------------------------------------------------
# Peak detection — fast method
# ---------------------------------------------------------------------------

class TestPeakDetectionFast:
    def test_finds_all_peaks_hpge(self, ps_hpge_fast):
        """All three true peaks should be found within tolerance."""
        for true_pos in PEAK_POSITIONS:
            assert any(abs(ps_hpge_fast.peaks_idx - true_pos) <= 5), \
                f"Peak at {true_pos} not found in {ps_hpge_fast.peaks_idx}"

    def test_finds_all_peaks_scint(self, ps_scint_fast):
        """All three true peaks should be found within tolerance."""
        for true_pos in PEAK_POSITIONS:
            assert any(abs(ps_scint_fast.peaks_idx - true_pos) <= 5), \
                f"Peak at {true_pos} not found in {ps_scint_fast.peaks_idx}"

    def test_snr_positive(self, ps_hpge_fast):
        assert np.all(ps_hpge_fast.snr >= 0)

    def test_snr_length_matches_spectrum(self, ps_hpge_fast, spec_hpge):
        assert len(ps_hpge_fast.snr) == len(spec_hpge.counts)

    def test_fwhm_guess_length_matches_peaks(self, ps_hpge_fast):
        assert len(ps_hpge_fast.fwhm_guess) == len(ps_hpge_fast.peaks_idx)

    def test_with_xrange(self, spec_hpge):
        ps = PeakSearch(
            spectrum=spec_hpge,
            ref_x=250,
            ref_fwhm=fwhm_hpge(250),
            min_snr=3,
            xrange=[50, 350],
            method="fast",
        )
        assert len(ps.peaks_idx) > 0
        assert all(50 <= p <= 350 for p in ps.peaks_idx)

    def test_with_energy_calibration(self, spec_hpge_cal):
        ps = PeakSearch(
            spectrum=spec_hpge_cal,
            ref_x=250,
            ref_fwhm=fwhm_hpge(250),
            min_snr=3,
            method="fast",
        )
        assert len(ps.peaks_idx) > 0


# ---------------------------------------------------------------------------
# fwhm()
# ---------------------------------------------------------------------------

class TestFwhm:
    def test_fwhm_at_ref_x(self, ps_hpge_km):
        """fwhm at ref_x should equal ref_fwhm + fwhm_at_0."""
        computed = ps_hpge_km.fwhm(ps_hpge_km.ref_x)
        expected = (ps_hpge_km.ref_fwhm / np.sqrt(ps_hpge_km.ref_x)) * np.sqrt(ps_hpge_km.ref_x) + ps_hpge_km.fwhm_at_0
        assert np.isclose(computed, expected)

    def test_fwhm_increases_with_x(self, ps_hpge_km):
        x = np.arange(1, 512)
        fwhm_vals = ps_hpge_km.fwhm(x)
        assert np.all(np.diff(fwhm_vals) >= 0)

    def test_fwhm_at_0_equals_fwhm_at_0_param(self, ps_hpge_km):
        assert np.isclose(ps_hpge_km.fwhm(0), ps_hpge_km.fwhm_at_0)


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_returns_dict(self, ps_hpge_km):
        assert isinstance(ps_hpge_km.metadata(), dict)

    def test_metadata_contains_keys(self, ps_hpge_km):
        meta = ps_hpge_km.metadata()
        for key in ["ref_x", "ref_fwhm", "fwhm_at_0", "min_snr", "method",
                    "xrange", "n_peaks", "peak_positions", "fwhm_guess"]:
            assert key in meta

    def test_metadata_n_peaks_correct(self, ps_hpge_km):
        meta = ps_hpge_km.metadata()
        assert meta["n_peaks"] == len(ps_hpge_km.peaks_idx)

    def test_metadata_method_correct(self, ps_hpge_km):
        assert ps_hpge_km.metadata()["method"] == "km"


# ---------------------------------------------------------------------------
# peaks_in_range()
# ---------------------------------------------------------------------------

class TestPeaksInRange:
    def test_returns_peaks_within_range(self, ps_hpge_km):
        result = ps_hpge_km.peaks_in_range(50, 300)
        assert np.all(result >= 50)
        assert np.all(result <= 300)

    def test_returns_empty_for_empty_range(self, ps_hpge_km):
        result = ps_hpge_km.peaks_in_range(450, 500)
        assert len(result) == 0

    def test_all_peaks_in_full_range(self, ps_hpge_km, spec_hpge):
        result = ps_hpge_km.peaks_in_range(0, len(spec_hpge.counts))
        assert len(result) == len(ps_hpge_km.peaks_idx)


# ---------------------------------------------------------------------------
# to_csv()
# ---------------------------------------------------------------------------

class TestToCsv:
    def test_csv_written_without_energies(self, ps_hpge_km, tmp_path):
        path = tmp_path / "peaks.csv"
        ps_hpge_km.to_csv(str(path))
        df = pd.read_csv(path)
        assert "channel" in df.columns
        assert "fwhm_guess" in df.columns
        assert "snr" in df.columns
        assert len(df) == len(ps_hpge_km.peaks_idx)

    def test_csv_written_with_energies(self, spec_hpge_cal, tmp_path):
        ps = PeakSearch(
            spectrum=spec_hpge_cal,
            ref_x=250,
            ref_fwhm=fwhm_hpge(250),
            min_snr=3,
            method="km",
        )
        path = tmp_path / "peaks.csv"
        ps.to_csv(str(path))
        df = pd.read_csv(path)
        assert "channel" in df.columns
        assert any("energy" in col.lower() for col in df.columns)
        assert len(df) == len(ps.peaks_idx)

    def test_csv_appends_extension(self, ps_hpge_km, tmp_path):
        path = tmp_path / "peaks"
        ps_hpge_km.to_csv(str(path))
        assert (tmp_path / "peaks.csv").exists()


# ---------------------------------------------------------------------------
# Plotting
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPlotPeaks:
    def test_plot_peaks_runs_without_error(self, ps_hpge_km):
        ps_hpge_km.plot()
        plt.close("all")

    def test_plot_peaks_returns_none(self, ps_hpge_km):
        result = ps_hpge_km.plot()
        assert result is None
        plt.close("all")

    def test_plot_peaks_with_ax(self, ps_hpge_km):
        fig, ax = plt.subplots()
        ps_hpge_km.plot(ax=ax)
        plt.close("all")

    def test_plot_peaks_snr_off(self, ps_hpge_km):
        ps_hpge_km.plot(snrs="off")
        plt.close("all")

@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPlotComponents:
    def test_plot_components_runs_without_error(self, ps_hpge_km):
        ps_hpge_km.plot_components()
        plt.close("all")

    def test_plot_components_with_ax(self, ps_hpge_km):
        fig, ax = plt.subplots()
        ps_hpge_km.plot_components(ax=ax)
        plt.close("all")

@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPlotKernel:
    def test_plot_kernel_runs_without_error(self, ps_hpge_km):
        ps_hpge_km.plot_kernel()
        plt.close("all")

    def test_plot_kernel_with_ax(self, ps_hpge_km):
        fig, ax = plt.subplots()
        ps_hpge_km.plot_kernel(ax=ax)
        plt.close("all")
