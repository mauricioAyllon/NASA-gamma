"""
Pytest tests for the Spectrum class and plot_overlay function.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before importing pyplot
import matplotlib.pyplot as plt
from nasagamma.spectrum import Spectrum, plot_overlay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def counts():
    """Simple synthetic counts array with a Gaussian peak."""
    channels = np.arange(0, 256)
    return (
        np.exp(-0.5 * ((channels - 128) / 10) ** 2) * 1000 +
        np.random.default_rng(42).poisson(10, size=len(channels))
    ).astype(float)


@pytest.fixture
def energies():
    """Simple linear energy axis in keV."""
    return np.linspace(0, 1000, 256)


@pytest.fixture
def spec_no_cal(counts):
    """Spectrum without energy calibration."""
    return Spectrum(counts=counts, realtime=100.0, livetime=95.0, label="Test Spectrum")


@pytest.fixture
def spec_with_cal(counts, energies):
    """Spectrum with energy calibration."""
    return Spectrum(
        counts=counts,
        energies=energies,
        e_units="keV",
        realtime=100.0,
        livetime=95.0,
        label="Calibrated Spectrum",
    )


@pytest.fixture
def spec_background(counts):
    """A second spectrum to use as background for arithmetic tests."""
    bg_counts = counts * 0.1
    return Spectrum(counts=bg_counts, realtime=100.0, livetime=94.0, label="Background")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_counts_none_raises(self):
        with pytest.raises(ValueError, match="counts must be specified"):
            Spectrum(counts=None)

    def test_channels_set_without_energies(self, spec_no_cal):
        assert len(spec_no_cal.channels) == len(spec_no_cal.counts)
        assert spec_no_cal.x_units == "Channels"

    def test_energies_set_correctly(self, spec_with_cal, energies):
        assert np.array_equal(spec_with_cal.energies, energies)
        assert spec_with_cal.x_units == "Energy (keV)"

    def test_x_units_energy_no_e_units(self, counts, energies):
        spec = Spectrum(counts=counts, energies=energies)
        assert spec.x_units == "Energy"

    def test_y_label_counts(self, spec_no_cal):
        assert spec_no_cal.y_label == "Cts"

    def test_y_label_cps(self, counts):
        spec = Spectrum(counts=counts, cps=True)
        assert spec.y_label == "CPS"

    def test_metadata_stored(self, spec_no_cal):
        assert spec_no_cal.realtime == 100.0
        assert spec_no_cal.livetime == 95.0
        assert spec_no_cal.label == "Test Spectrum"


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_returns_new_object(self, spec_no_cal):
        copy = spec_no_cal.copy()
        assert copy is not spec_no_cal

    def test_copy_counts_equal(self, spec_no_cal):
        copy = spec_no_cal.copy()
        assert np.array_equal(copy.counts, spec_no_cal.counts)

    def test_copy_is_independent(self, spec_no_cal):
        copy = spec_no_cal.copy()
        copy.counts[0] = -999
        assert spec_no_cal.counts[0] != -999

    def test_copy_energies_independent(self, spec_with_cal):
        copy = spec_with_cal.copy()
        copy.energies[0] = -999
        assert spec_with_cal.energies[0] != -999

    def test_copy_metadata_preserved(self, spec_no_cal):
        copy = spec_no_cal.copy()
        assert copy.realtime == spec_no_cal.realtime
        assert copy.livetime == spec_no_cal.livetime
        assert copy.label == spec_no_cal.label


# ---------------------------------------------------------------------------
# smooth()
# ---------------------------------------------------------------------------

class TestSmooth:
    def test_smooth_preserves_total_counts(self, spec_no_cal):
        total_before = spec_no_cal.counts.sum()
        spec_no_cal.smooth(num=4)
        assert np.isclose(spec_no_cal.counts.sum(), total_before)

    def test_smooth_preserves_length(self, spec_no_cal):
        length_before = len(spec_no_cal.counts)
        spec_no_cal.smooth(num=4)
        assert len(spec_no_cal.counts) == length_before


# ---------------------------------------------------------------------------
# rebin()
# ---------------------------------------------------------------------------

class TestRebin:
    def test_rebin_reduces_bins(self, spec_no_cal):
        original_length = len(spec_no_cal.counts)
        spec_no_cal.rebin(by=2)
        assert len(spec_no_cal.counts) == original_length // 2

    def test_rebin_preserves_total_counts(self, spec_no_cal):
        total_before = spec_no_cal.counts.sum()
        spec_no_cal.rebin(by=2)
        assert np.isclose(spec_no_cal.counts.sum(), total_before)

    def test_rebin_updates_energies(self, spec_with_cal):
        original_length = len(spec_with_cal.energies)
        spec_with_cal.rebin(by=2)
        assert len(spec_with_cal.energies) == original_length // 2

    def test_rebin_updates_channels_without_cal(self, spec_no_cal):
        spec_no_cal.rebin(by=2)
        assert np.array_equal(spec_no_cal.x, spec_no_cal.channels)


# ---------------------------------------------------------------------------
# gain_shift()
# ---------------------------------------------------------------------------

class TestGainShift:
    def test_positive_shift_zero_fills_low_end(self, spec_no_cal):
        spec_no_cal.gain_shift(by=10)
        assert np.all(spec_no_cal.counts[:10] == 0)

    def test_negative_shift_fills_high_end(self, spec_no_cal):
        original_last = spec_no_cal.counts[-1]
        spec_no_cal.gain_shift(by=-10)
        assert np.all(spec_no_cal.counts[-10:] == spec_no_cal.counts[-11])

    def test_zero_shift_does_nothing(self, spec_no_cal, capsys):
        counts_before = spec_no_cal.counts.copy()
        spec_no_cal.gain_shift(by=0)
        assert np.array_equal(spec_no_cal.counts, counts_before)
        captured = capsys.readouterr()
        assert "No shift applied" in captured.out

    def test_energy_shift(self, spec_with_cal):
        counts_before = spec_with_cal.counts.copy()
        spec_with_cal.gain_shift(by=10.0, energy=True)
        assert not np.array_equal(spec_with_cal.counts, counts_before)


# ---------------------------------------------------------------------------
# replace_neg_vals()
# ---------------------------------------------------------------------------

class TestReplaceNegVals:
    def test_no_negative_values_after(self, spec_no_cal):
        spec_no_cal.counts[5] = -50.0
        spec_no_cal.replace_neg_vals()
        assert np.all(spec_no_cal.counts >= 0)


# ---------------------------------------------------------------------------
# gaussian_energy_broadening()
# ---------------------------------------------------------------------------

class TestGaussianEnergyBroadening:
    def test_broadening_preserves_total_counts_approx(self, spec_with_cal):
        total_before = spec_with_cal.counts.sum()
        spec_with_cal.gaussian_energy_broadening(
            fwhm_func=Spectrum.fwhm_HPGe_example, random_seed=42
        )
        # Poisson sampling means totals won't be exact, allow 5% tolerance
        assert abs(spec_with_cal.counts.sum() - total_before) / total_before < 0.05

    def test_broadening_output_length_unchanged(self, spec_with_cal):
        length_before = len(spec_with_cal.counts)
        spec_with_cal.gaussian_energy_broadening(
            fwhm_func=Spectrum.fwhm_HPGe_example, random_seed=42
        )
        assert len(spec_with_cal.counts) == length_before


# ---------------------------------------------------------------------------
# remove_calibration()
# ---------------------------------------------------------------------------

class TestRemoveCalibration:
    def test_energies_none_after(self, spec_with_cal):
        spec_with_cal.remove_calibration()
        assert spec_with_cal.energies is None

    def test_x_units_channels_after(self, spec_with_cal):
        spec_with_cal.remove_calibration()
        assert spec_with_cal.x_units == "Channels"


# ---------------------------------------------------------------------------
# to_csv()
# ---------------------------------------------------------------------------

class TestToCsv:
    def test_csv_written_without_energies(self, spec_no_cal, tmp_path):
        path = tmp_path / "test.csv"
        spec_no_cal.to_csv(str(path))
        df = pd.read_csv(path)
        assert "counts" in df.columns
        assert len(df) == len(spec_no_cal.counts)

    def test_csv_written_with_energies(self, spec_with_cal, tmp_path):
        path = tmp_path / "test.csv"
        spec_with_cal.to_csv(str(path))
        df = pd.read_csv(path)
        assert "counts" in df.columns
        assert len(df) == len(spec_with_cal.counts)

    def test_csv_appends_extension(self, spec_no_cal, tmp_path):
        path = tmp_path / "test"
        spec_no_cal.to_csv(str(path))
        assert (tmp_path / "test.csv").exists()


# ---------------------------------------------------------------------------
# to_txt()
# ---------------------------------------------------------------------------

class TestToTxt:
    def test_txt_written_without_energies(self, spec_no_cal, tmp_path):
        path = tmp_path / "test.txt"
        spec_no_cal.to_txt(str(path))
        content = path.read_text()
        assert "counts" in content
        assert "Real time" in content

    def test_txt_written_with_energies(self, spec_with_cal, tmp_path):
        path = tmp_path / "test.txt"
        spec_with_cal.to_txt(str(path))
        content = path.read_text()
        assert "counts" in content
        assert "Energy" in content

    def test_txt_appends_extension(self, spec_no_cal, tmp_path):
        path = tmp_path / "test"
        spec_no_cal.to_txt(str(path))
        assert (tmp_path / "test.txt").exists()


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add_counts(self, spec_no_cal, spec_background):
        result = spec_no_cal + spec_background
        expected = spec_no_cal.counts + spec_background.counts
        assert np.allclose(result.counts, expected)

    def test_add_sums_livetimes(self, spec_no_cal, spec_background):
        result = spec_no_cal + spec_background
        assert result.livetime == spec_no_cal.livetime + spec_background.livetime

    def test_add_sums_realtimes(self, spec_no_cal, spec_background):
        result = spec_no_cal + spec_background
        assert result.realtime == spec_no_cal.realtime + spec_background.realtime

    def test_add_mismatched_lengths_raises(self, spec_no_cal, counts):
        short_spec = Spectrum(counts=counts[:100])
        with pytest.raises(ValueError, match="different number of bins"):
            spec_no_cal + short_spec

    def test_sub_counts(self, spec_no_cal, spec_background):
        result = spec_no_cal - spec_background
        expected = spec_no_cal.counts - spec_background.counts
        assert np.allclose(result.counts, expected)

    def test_sub_mismatched_lengths_raises(self, spec_no_cal, counts):
        short_spec = Spectrum(counts=counts[:100])
        with pytest.raises(ValueError, match="different number of bins"):
            spec_no_cal - short_spec

    def test_mul_scalar(self, spec_no_cal):
        result = spec_no_cal * 2.0
        assert np.allclose(result.counts, spec_no_cal.counts * 2.0)

    def test_rmul_scalar(self, spec_no_cal):
        result = 2.0 * spec_no_cal
        assert np.allclose(result.counts, spec_no_cal.counts * 2.0)

    def test_mul_array(self, spec_no_cal):
        arr = np.linspace(0.8, 1.2, len(spec_no_cal.counts))
        result = spec_no_cal * arr
        assert np.allclose(result.counts, spec_no_cal.counts * arr)

    def test_truediv_scalar(self, spec_no_cal):
        result = spec_no_cal / 2.0
        assert np.allclose(result.counts, spec_no_cal.counts / 2.0)

    def test_truediv_array(self, spec_no_cal):
        arr = np.linspace(0.8, 1.2, len(spec_no_cal.counts))
        result = spec_no_cal / arr
        assert np.allclose(result.counts, spec_no_cal.counts / arr)

    def test_radd_via_sum(self, spec_no_cal, spec_background):
        result = sum([spec_no_cal, spec_background])
        expected = spec_no_cal.counts + spec_background.counts
        assert np.allclose(result.counts, expected)


# ---------------------------------------------------------------------------
# plot() and plot_overlay()
# ---------------------------------------------------------------------------
# plt.show() does nothing in a non-interactive backend
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPlot:
    def test_plot_runs_without_error(self, spec_no_cal):
        spec_no_cal.plot()
        plt.close("all")

    def test_plot_with_cal_runs_without_error(self, spec_with_cal):
        spec_with_cal.plot()
        plt.close("all")

    def test_plot_does_not_mutate_label(self, spec_no_cal):
        spec_no_cal.label = None
        spec_no_cal.plot()
        assert spec_no_cal.label is None
        plt.close("all")

    def test_plot_returns_ax(self, spec_no_cal):
        result = spec_no_cal.plot()
        assert isinstance(result, plt.Axes)
        plt.close("all")

@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPlotOverlay:
    def test_plot_overlay_runs_without_error(self, spec_no_cal, spec_background):
        ax = plot_overlay([spec_no_cal, spec_background])
        plt.close("all")

    def test_plot_overlay_returns_ax(self, spec_no_cal, spec_background):
        ax = plot_overlay([spec_no_cal, spec_background])
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_overlay_empty_list_raises(self):
        with pytest.raises(ValueError, match="spectra list cannot be empty"):
            plot_overlay([])

    def test_plot_overlay_with_colors(self, spec_no_cal, spec_background):
        ax = plot_overlay(
            [spec_no_cal, spec_background],
            colors=["steelblue", "tomato"]
        )
        plt.close("all")
