"""
Pytest tests for the PeakFit class and related functions.
"""

import pytest
import numpy as np
import pandas as pd

from nasagamma.spectrum import Spectrum
from nasagamma.peaksearch import PeakSearch
from nasagamma.peakfit import PeakFit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spectrum_with_peaks(
    n_channels=512,
    peak_positions=None,
    peak_heights=None,
    peak_sigmas=None,
    bkg_intercept=50,
    bkg_slope=-0.05,
    with_energy=False,
    seed=42,
):
    """
    Create a synthetic spectrum with overlapping Gaussian peaks
    on a linear background.

    Parameters
    ----------
    n_channels : int
        number of channels.
    peak_positions : list of int
        true peak centroids in channels.
    peak_heights : list of float
        peak amplitudes.
    peak_sigmas : list of float
        standard deviations of each peak in channels.
    bkg_intercept : float
        background intercept.
    bkg_slope : float
        background slope per channel.
    with_energy : bool
        if True, attach a linear energy calibration.
    seed : int
        random seed for reproducibility.
    """
    if peak_positions is None:
        peak_positions = [200, 230]
    if peak_heights is None:
        peak_heights = [5000, 3000]
    if peak_sigmas is None:
        peak_sigmas = [6, 6]

    rng = np.random.default_rng(seed)
    channels = np.arange(n_channels, dtype=float)
    background = bkg_intercept + bkg_slope * channels
    counts = background.copy()
    for pos, height, sigma in zip(peak_positions, peak_heights, peak_sigmas):
        counts += height * np.exp(-0.5 * ((channels - pos) / sigma) ** 2)
    counts = rng.poisson(counts).astype(float)

    if with_energy:
        energies = 0.5 + 0.72 * channels
        return Spectrum(counts=counts, energies=energies, e_units="keV")
    return Spectrum(counts=counts)


def make_peaksearch(spec, ref_x=215, ref_fwhm=14.0, fwhm_at_0=1.0, min_snr=3):
    """Create a PeakSearch object from a spectrum."""
    return PeakSearch(
        spectrum=spec,
        ref_x=ref_x,
        ref_fwhm=ref_fwhm,
        fwhm_at_0=fwhm_at_0,
        min_snr=min_snr,
        method="km",
    )


# ---------------------------------------------------------------------------
# Fixtures — single isolated peak
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_single():
    """Spectrum with a single isolated Gaussian peak on linear background."""
    return make_spectrum_with_peaks(
        peak_positions=[250],
        peak_heights=[8000],
        peak_sigmas=[6],
    )


@pytest.fixture
def search_single(spec_single):
    """PeakSearch on single peak spectrum."""
    return make_peaksearch(spec_single, ref_x=250, ref_fwhm=14.0)


@pytest.fixture
def spec_single_cal():
    """Spectrum with a single isolated peak and energy calibration."""
    return make_spectrum_with_peaks(
        peak_positions=[250],
        peak_heights=[8000],
        peak_sigmas=[6],
        with_energy=True,
    )


@pytest.fixture
def search_single_cal(spec_single_cal):
    """PeakSearch on single peak calibrated spectrum."""
    return make_peaksearch(spec_single_cal, ref_x=250, ref_fwhm=14.0)


# ---------------------------------------------------------------------------
# Fixtures — overlapping peaks
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_overlap():
    """
    Spectrum with two overlapping Gaussian peaks on linear background.
    Peaks at 200 and 230 with sigma=6 are separated by ~5 sigma,
    creating significant overlap.
    """
    return make_spectrum_with_peaks(
        peak_positions=[200, 230],
        peak_heights=[5000, 3000],
        peak_sigmas=[6, 6],
    )


@pytest.fixture
def search_overlap(spec_overlap):
    """PeakSearch on overlapping peaks spectrum."""
    return make_peaksearch(spec_overlap, ref_x=215, ref_fwhm=14.0)


@pytest.fixture
def spec_overlap_cal():
    """Overlapping peaks spectrum with energy calibration."""
    return make_spectrum_with_peaks(
        peak_positions=[200, 230],
        peak_heights=[5000, 3000],
        peak_sigmas=[6, 6],
        with_energy=True,
    )


@pytest.fixture
def search_overlap_cal(spec_overlap_cal):
    """PeakSearch on overlapping peaks calibrated spectrum."""
    return make_peaksearch(spec_overlap_cal, ref_x=215, ref_fwhm=14.0)


# ---------------------------------------------------------------------------
# Fixtures — PeakFit objects for each background type
# ---------------------------------------------------------------------------

@pytest.fixture
def fit_poly1(search_single):
    """PeakFit with poly1 background on single peak."""
    return PeakFit(search_single, xrange=[220, 280], bkg="poly1")


@pytest.fixture
def fit_poly2(search_single):
    """PeakFit with poly2 background on single peak."""
    return PeakFit(search_single, xrange=[220, 280], bkg="poly2")


@pytest.fixture
def fit_linear(search_single):
    """PeakFit with linear background on single peak."""
    return PeakFit(search_single, xrange=[220, 280], bkg="linear")


@pytest.fixture
def fit_quadratic(search_single):
    """PeakFit with quadratic background on single peak."""
    return PeakFit(search_single, xrange=[220, 280], bkg="quadratic")


@pytest.fixture
def fit_exponential(search_single):
    """PeakFit with exponential background on single peak."""
    return PeakFit(search_single, xrange=[220, 280], bkg="exponential")


@pytest.fixture
def fit_overlap(search_overlap):
    """PeakFit with poly1 background on overlapping peaks."""
    return PeakFit(search_overlap, xrange=[170, 260], bkg="poly1")


@pytest.fixture
def fit_overlap_cal(search_overlap_cal):
    """PeakFit with poly1 background on overlapping calibrated peaks."""
    return PeakFit(search_overlap_cal, xrange=[140, 172], bkg="poly1")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_search_type(self, spec_single):
        with pytest.raises(TypeError, match="search must be a PeakSearch object"):
            PeakFit(search=spec_single, xrange=[220, 280])

    def test_no_peaks_in_range_raises(self, search_single):
        with pytest.raises(ValueError, match="No peaks found within range"):
            PeakFit(search_single, xrange=[10, 50])

    def test_invalid_bkg_still_attempts_fit(self, search_single):
        """Invalid bkg string that contains no digit should raise an error."""
        with pytest.raises(Exception):
            PeakFit(search_single, xrange=[220, 280], bkg="invalid")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_fit_result_not_none(self, fit_poly1):
        assert fit_poly1.fit_result is not None

    def test_x_data_not_none(self, fit_poly1):
        assert fit_poly1.x_data is not None

    def test_y_data_not_none(self, fit_poly1):
        assert fit_poly1.y_data is not None

    def test_continuum_not_none(self, fit_poly1):
        assert fit_poly1.continuum is not None

    def test_peak_info_length(self, fit_poly1):
        assert len(fit_poly1.peak_info) == 1

    def test_peak_err_length(self, fit_poly1):
        assert len(fit_poly1.peak_err) == 1

    def test_bkg_key_set(self, fit_poly1):
        assert fit_poly1.bkg_key is not None

    def test_x_units_set(self, fit_poly1):
        assert fit_poly1.x_units == "Channels"

    def test_x_units_with_cal(self, fit_overlap_cal):
        assert fit_overlap_cal.x_units == "Energy (keV)"

    def test_overlap_finds_two_peaks(self, fit_overlap):
        assert len(fit_overlap.peak_info) == 2


# ---------------------------------------------------------------------------
# Background types
# ---------------------------------------------------------------------------

class TestBackgroundTypes:
    def test_poly1_bkg_key(self, fit_poly1):
        assert fit_poly1.bkg_key == "polynomial"

    def test_poly2_bkg_key(self, fit_poly2):
        assert fit_poly2.bkg_key == "polynomial"

    def test_linear_bkg_key(self, fit_linear):
        assert fit_linear.bkg_key == "linear"

    def test_quadratic_bkg_key(self, fit_quadratic):
        assert fit_quadratic.bkg_key == "quadratic"

    def test_exponential_bkg_key(self, fit_exponential):
        assert fit_exponential.bkg_key == "exponential"

    def test_poly1_fit_succeeds(self, fit_poly1):
        assert "succeeded" in fit_poly1.fit_result.message.lower()

    def test_poly2_fit_succeeds(self, fit_poly2):
        assert "succeeded" in fit_poly2.fit_result.message.lower()

    def test_linear_fit_succeeds(self, fit_linear):
        assert "succeeded" in fit_linear.fit_result.message.lower()

    def test_quadratic_fit_succeeds(self, fit_quadratic):
        assert "succeeded" in fit_quadratic.fit_result.message.lower()

    def test_exponential_fit_succeeds(self, fit_exponential):
        assert "succeeded" in fit_exponential.fit_result.message.lower()


# ---------------------------------------------------------------------------
# Fit quality — single peak
# ---------------------------------------------------------------------------

class TestFitQualitySingle:
    MEAN_TOLERANCE = 2    # channels
    AREA_TOLERANCE = 0.10  # 10%

    def test_mean_within_tolerance_poly1(self, fit_poly1):
        mean = fit_poly1.peak_info[0]["mean"]
        assert abs(mean - 250) <= self.MEAN_TOLERANCE

    def test_mean_within_tolerance_linear(self, fit_linear):
        mean = fit_linear.peak_info[0]["mean"]
        assert abs(mean - 250) <= self.MEAN_TOLERANCE

    def test_mean_within_tolerance_quadratic(self, fit_quadratic):
        mean = fit_quadratic.peak_info[0]["mean"]
        assert abs(mean - 250) <= self.MEAN_TOLERANCE

    def test_mean_within_tolerance_exponential(self, fit_exponential):
        mean = fit_exponential.peak_info[0]["mean"]
        assert abs(mean - 250) <= self.MEAN_TOLERANCE

    def test_area_within_tolerance_poly1(self, fit_poly1):
        area = fit_poly1.peak_info[0]["area"]
        # true area = height * sigma * sqrt(2*pi)
        true_area = 8000 * 6 * np.sqrt(2 * np.pi)
        assert abs(area - true_area) / true_area <= self.AREA_TOLERANCE

    def test_fwhm_within_tolerance_poly1(self, fit_poly1):
        fwhm = fit_poly1.peak_info[0]["fwhm"]
        true_fwhm = 6 * 2.355
        assert abs(fwhm - true_fwhm) / true_fwhm <= self.AREA_TOLERANCE

    def test_redchi_reasonable(self, fit_poly1):
        """Reduced chi-squared should be close to 1 for a good fit."""
        assert 0.5 <= fit_poly1.fit_result.redchi <= 2.0

    def test_continuum_positive(self, fit_poly1):
        assert fit_poly1.continuum > 0


# ---------------------------------------------------------------------------
# Fit quality — overlapping peaks
# ---------------------------------------------------------------------------

class TestFitQualityOverlap:
    MEAN_TOLERANCE = 2    # channels
    AREA_TOLERANCE = 0.15  # 15% — slightly looser for overlapping peaks

    def test_two_peaks_found(self, fit_overlap):
        assert len(fit_overlap.peak_info) == 2

    def test_mean_peak1_within_tolerance(self, fit_overlap):
        means = sorted([fit_overlap.peak_info[i]["mean"] for i in range(2)])
        assert abs(means[0] - 200) <= self.MEAN_TOLERANCE

    def test_mean_peak2_within_tolerance(self, fit_overlap):
        means = sorted([fit_overlap.peak_info[i]["mean"] for i in range(2)])
        assert abs(means[1] - 230) <= self.MEAN_TOLERANCE

    def test_area_peak1_within_tolerance(self, fit_overlap):
        means = [fit_overlap.peak_info[i]["mean"] for i in range(2)]
        idx = int(np.argmin(means))
        area = fit_overlap.peak_info[idx]["area"]
        true_area = 5000 * 6 * np.sqrt(2 * np.pi)
        assert abs(area - true_area) / true_area <= self.AREA_TOLERANCE

    def test_area_peak2_within_tolerance(self, fit_overlap):
        means = [fit_overlap.peak_info[i]["mean"] for i in range(2)]
        idx = int(np.argmax(means))
        area = fit_overlap.peak_info[idx]["area"]
        true_area = 3000 * 6 * np.sqrt(2 * np.pi)
        assert abs(area - true_area) / true_area <= self.AREA_TOLERANCE

    def test_redchi_reasonable(self, fit_overlap):
        assert 0.5 <= fit_overlap.fit_result.redchi <= 2.0

    def test_overlap_with_cal(self, fit_overlap_cal):
        assert len(fit_overlap_cal.peak_info) == 2


# ---------------------------------------------------------------------------
# peak_info and peak_err structure
# ---------------------------------------------------------------------------

class TestPeakInfoStructure:
    def test_peak_info_has_mean_key(self, fit_poly1):
        assert "mean" in fit_poly1.peak_info[0]

    def test_peak_info_has_area_key(self, fit_poly1):
        assert "area" in fit_poly1.peak_info[0]

    def test_peak_info_has_fwhm_key(self, fit_poly1):
        assert "fwhm" in fit_poly1.peak_info[0]

    def test_peak_err_has_mean_err_key(self, fit_poly1):
        assert "mean_err" in fit_poly1.peak_err[0]

    def test_peak_err_has_area_err_key(self, fit_poly1):
        assert "area_err" in fit_poly1.peak_err[0]

    def test_peak_err_has_fwhm_err_key(self, fit_poly1):
        assert "fwhm_err" in fit_poly1.peak_err[0]

    def test_errors_are_positive_or_nan(self, fit_poly1):
        for key, val in fit_poly1.peak_err[0].items():
            assert np.isnan(val) or val > 0, f"{key} should be positive or nan"

    def test_overlap_two_peak_info_dicts(self, fit_overlap):
        assert len(fit_overlap.peak_info) == 2
        for info in fit_overlap.peak_info:
            assert "mean" in info
            assert "area" in info
            assert "fwhm" in info


# ---------------------------------------------------------------------------
# Skewed Gaussian
# ---------------------------------------------------------------------------

class TestSkewedGaussian:
    def test_skew_fit_succeeds(self, search_single):
        fit = PeakFit(search_single, xrange=[220, 280], bkg="poly1", skew=True)
        assert fit.fit_result is not None
        assert len(fit.peak_info) == 1

    def test_skew_mean_within_tolerance(self, search_single):
        fit = PeakFit(search_single, xrange=[220, 280], bkg="poly1", skew=True)
        mean = fit.peak_info[0]["mean"]
        assert abs(mean - 250) <= 2


# ---------------------------------------------------------------------------
# find_peaks_range
# ---------------------------------------------------------------------------

class TestFindPeaksRange:
    def test_returns_mask_and_pidx(self, fit_poly1):
        mask, pidx = fit_poly1.find_peaks_range()
        assert isinstance(mask, np.ndarray)
        assert isinstance(pidx, np.ndarray)

    def test_pidx_within_xrange(self, fit_poly1):
        mask, pidx = fit_poly1.find_peaks_range()
        x = fit_poly1.x
        assert all(fit_poly1.xrange[0] < x[p] < fit_poly1.xrange[1] for p in pidx)

    def test_overlap_returns_two_peaks(self, fit_overlap):
        mask, pidx = fit_overlap.find_peaks_range()
        assert len(pidx) == 2


