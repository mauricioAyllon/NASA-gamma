"""
Pytest tests for EnergyCalibration class and smart_calibration function.
"""

import pytest
import numpy as np

from nasagamma.energy_calibration import EnergyCalibration, smart_calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_calibration_data(
    n_points=10,
    slope=0.72,
    intercept=0.5,
    noise=0.1,
    n_channels=512,
    seed=42,
):
    """
    Generate synthetic calibration data with known linear coefficients.

    Parameters
    ----------
    n_points : int
        number of calibration points.
    slope : float
        true slope (keV/channel).
    intercept : float
        true intercept (keV).
    noise : float
        standard deviation of Gaussian noise added to energies.
    n_channels : int
        total number of channels.
    seed : int
        random seed for reproducibility.

    Returns
    -------
    mean_vals : numpy array
        channel positions of calibration peaks.
    erg : numpy array
        corresponding energy values with noise.
    channels : numpy array
        full channel array.
    """
    rng = np.random.default_rng(seed)
    channels = np.arange(n_channels, dtype=float)
    mean_vals = np.linspace(50, n_channels - 50, n_points)
    erg = intercept + slope * mean_vals + rng.normal(0, noise, n_points)
    return mean_vals, erg, channels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRUE_SLOPE = 0.72
TRUE_INTERCEPT = 0.5
TOLERANCE = 0.01  # 1%

@pytest.fixture
def cal_data():
    """Standard synthetic calibration data."""
    return make_calibration_data(
        n_points=10,
        slope=TRUE_SLOPE,
        intercept=TRUE_INTERCEPT,
        noise=0.05,
    )


@pytest.fixture
def cal_data_noiseless():
    """Noiseless calibration data for exact coefficient checks."""
    return make_calibration_data(
        n_points=10,
        slope=TRUE_SLOPE,
        intercept=TRUE_INTERCEPT,
        noise=0.0,
    )


@pytest.fixture
def ecal(cal_data):
    """EnergyCalibration instance from standard data."""
    mean_vals, erg, channels = cal_data
    return EnergyCalibration(mean_vals, erg, channels, n=1)


@pytest.fixture
def ecal_noiseless(cal_data_noiseless):
    """EnergyCalibration instance from noiseless data."""
    mean_vals, erg, channels = cal_data_noiseless
    return EnergyCalibration(mean_vals, erg, channels, n=1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_mismatched_lengths_raises(self):
        mean_vals = [100, 200, 300]
        erg = [72.0, 144.0]
        channels = np.arange(512)
        with pytest.raises(ValueError, match="same length"):
            EnergyCalibration(mean_vals, erg, channels)

    def test_insufficient_points_for_degree_raises(self):
        mean_vals = [100, 200]
        erg = [72.0, 144.0]
        channels = np.arange(512)
        with pytest.raises(ValueError, match="at least"):
            EnergyCalibration(mean_vals, erg, channels, n=2)

    def test_minimum_points_linear(self):
        """Two points should be sufficient for a linear fit."""
        mean_vals = [100, 400]
        erg = [72.0, 288.0]
        channels = np.arange(512)
        cal = EnergyCalibration(mean_vals, erg, channels, n=1)
        assert cal.fit is not None


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_fit_not_none(self, ecal):
        assert ecal.fit is not None

    def test_predicted_not_none(self, ecal):
        assert ecal.predicted is not None

    def test_predicted_length(self, ecal, cal_data):
        _, _, channels = cal_data
        assert len(ecal.predicted) == len(channels)

    def test_e_units_default(self, ecal):
        assert ecal.e_units == "keV"

    def test_e_units_custom(self, cal_data):
        mean_vals, erg, channels = cal_data
        cal = EnergyCalibration(mean_vals, erg, channels, e_units="MeV")
        assert cal.e_units == "MeV"

    def test_n_stored(self, ecal):
        assert ecal.n == 1

    def test_arrays_converted_to_float(self, ecal):
        assert ecal.mean_vals.dtype == float
        assert ecal.erg.dtype == float
        assert ecal.channels.dtype == float


# ---------------------------------------------------------------------------
# Fit quality
# ---------------------------------------------------------------------------

class TestFitQuality:
    def test_slope_within_tolerance(self, ecal_noiseless):
        fitted_slope = ecal_noiseless.fit.best_values["c1"]
        assert abs(fitted_slope - TRUE_SLOPE) / TRUE_SLOPE <= TOLERANCE

    def test_intercept_within_tolerance(self, ecal_noiseless):
        fitted_intercept = ecal_noiseless.fit.best_values["c0"]
        assert abs(fitted_intercept - TRUE_INTERCEPT) / TRUE_INTERCEPT <= TOLERANCE

    def test_predicted_values_within_tolerance(self, ecal_noiseless, cal_data_noiseless):
        mean_vals, erg, channels = cal_data_noiseless
        true_predicted = TRUE_INTERCEPT + TRUE_SLOPE * channels
        assert np.allclose(ecal_noiseless.predicted, true_predicted, rtol=TOLERANCE)

    def test_redchi_reasonable(self, ecal):
        """Reduced chi-squared should be close to 1 for a good fit."""
        assert ecal.fit.redchi < 10.0

    def test_fit_succeeds(self, ecal):
        assert "succeeded" in ecal.fit.message.lower()


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_returns_dict(self, ecal):
        assert isinstance(ecal.metadata(), dict)

    def test_contains_keys(self, ecal):
        meta = ecal.metadata()
        for key in ["n", "e_units", "mean_vals", "erg", "redchi", "coefficients"]:
            assert key in meta

    def test_n_correct(self, ecal):
        assert ecal.metadata()["n"] == 1

    def test_e_units_correct(self, ecal):
        assert ecal.metadata()["e_units"] == "keV"

    def test_coefficients_length(self, ecal):
        # degree n polynomial has n+1 coefficients
        assert len(ecal.metadata()["coefficients"]) == ecal.n + 1

    def test_redchi_positive(self, ecal):
        assert ecal.metadata()["redchi"] > 0


# ---------------------------------------------------------------------------
# smart_calibration
# ---------------------------------------------------------------------------

class TestSmartCalibration:
    def test_returns_dict(self):
        channels = [100, 200, 300, 400, 500]
        energies = [72.0, 144.0, 216.0, 288.0, 360.0]
        result = smart_calibration(channels, energies)
        assert isinstance(result, dict)

    def test_contains_keys(self):
        channels = [100, 200, 300, 400, 500]
        energies = [72.0, 144.0, 216.0, 288.0, 360.0]
        result = smart_calibration(channels, energies)
        for key in ["c0", "c1", "r2", "channels", "energies"]:
            assert key in result

    def test_r2_close_to_1_for_linear_data(self):
        """R² should be very close to 1 for perfectly linear data."""
        channels = np.array([100, 200, 300, 400, 500], dtype=float)
        energies = 0.5 + 0.72 * channels
        result = smart_calibration(channels.tolist(), energies.tolist())
        assert result["r2"] > 0.999

    def test_slope_within_tolerance(self):
        channels = np.array([100, 200, 300, 400, 500], dtype=float)
        energies = TRUE_INTERCEPT + TRUE_SLOPE * channels
        result = smart_calibration(channels.tolist(), energies.tolist())
        assert abs(result["c1"] - TRUE_SLOPE) / TRUE_SLOPE <= TOLERANCE

    def test_intercept_within_tolerance(self):
        channels = np.array([100, 200, 300, 400, 500], dtype=float)
        energies = TRUE_INTERCEPT + TRUE_SLOPE * channels
        result = smart_calibration(channels.tolist(), energies.tolist())
        assert abs(result["c0"] - TRUE_INTERCEPT) / TRUE_INTERCEPT <= TOLERANCE

    def test_unequal_lengths_more_channels(self):
        """More channels than energies — should find best subset."""
        channels = np.array([100, 200, 250, 300, 400, 500], dtype=float)
        energies = TRUE_INTERCEPT + TRUE_SLOPE * np.array([100, 200, 300, 400, 500])
        result = smart_calibration(channels.tolist(), energies.tolist())
        assert result["r2"] > 0.999

    def test_unequal_lengths_more_energies(self):
        """More energies than channels — should find best subset."""
        channels = np.array([100, 200, 300, 400, 500], dtype=float)
        energies = TRUE_INTERCEPT + TRUE_SLOPE * np.array([100, 200, 250, 300, 400, 500])
        result = smart_calibration(channels.tolist(), energies.tolist())
        assert result["r2"] > 0.999

    def test_require_positive_slope_accepted(self):
        """require_positive_slope parameter should be accepted without error."""
        channels = np.array([100, 200, 300, 400, 500], dtype=float)
        energies = TRUE_INTERCEPT + TRUE_SLOPE * channels
        result = smart_calibration(
            channels.tolist(), energies.tolist(), require_positive_slope=True
        )
        assert result["c1"] > 0

    def test_insufficient_channels_raises(self):
        with pytest.raises(ValueError, match="at least"):
            smart_calibration([100, 200], [72.0, 144.0, 216.0], min_points=3)

    def test_insufficient_energies_raises(self):
        with pytest.raises(ValueError, match="at least"):
            smart_calibration([100, 200, 300], [72.0, 144.0], min_points=3)

    def test_max_combinations_raises(self):
        """Large input should raise when combinations exceed max_combinations."""
        channels = list(range(0, 200, 10))   # 20 channels
        energies = list(range(0, 100, 10))   # 10 energies
        # comb(20, 10) = 184,756 which exceeds max_combinations=100
        with pytest.raises(ValueError, match="Too many combinations"):
            smart_calibration(channels, energies, max_combinations=100)

    def test_max_combinations_custom(self):
        """Should succeed when max_combinations is set high enough."""
        channels = np.array([100, 200, 300, 400, 500], dtype=float)
        energies = TRUE_INTERCEPT + TRUE_SLOPE * channels
        result = smart_calibration(
            channels.tolist(),
            energies.tolist(),
            max_combinations=1_000_000,
        )
        assert result["r2"] > 0.999
