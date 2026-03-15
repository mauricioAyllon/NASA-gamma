"""
Pytest tests for file_reader module.
Uses example files from NASA-gamma/examples/data.
"""

import pytest
import numpy as np
from pathlib import Path

from nasagamma.spectrum import Spectrum
from nasagamma.file_reader import (
    process_df,
    read_csv,
    read_txt,
    read_mca,
    read_spe,
    read_cnf,
)

# ---------------------------------------------------------------------------
# Path to example data files
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "examples" / "data"

CSV_NO_CAL   = str(DATA_DIR / "gui_test_data_cebr.csv")
CSV_WITH_CAL = str(DATA_DIR / "gui_test_data_cebr_cal.csv")
TXT_FILE     = str(DATA_DIR / "gui_test_data_hpge_NH3.txt")
MCA_FILE     = str(DATA_DIR / "gui_test_data_3He.mca")
SPE_FILE     = str(DATA_DIR / "gui_test_data_hpge_Cu.Spe")
CNF_FILE     = str(DATA_DIR / "gui_test_data_lab_sources.cnf")


# ---------------------------------------------------------------------------
# process_df
# ---------------------------------------------------------------------------

class TestProcessDf:
    def test_counts_column_detected(self):
        import pandas as pd
        df = pd.DataFrame({"counts": [1, 2, 3], "Energy [keV]": [10, 20, 30]})
        unit, cts_col, erg = process_df(df)
        assert cts_col == "counts"

    def test_cts_column_detected(self):
        import pandas as pd
        df = pd.DataFrame({"cts": [1, 2, 3]})
        unit, cts_col, erg = process_df(df)
        assert cts_col == "cts"

    def test_energy_column_detected(self):
        import pandas as pd
        df = pd.DataFrame({"counts": [1, 2, 3], "energy[keV]": [10, 20, 30]})
        unit, cts_col, erg = process_df(df)
        assert erg is not None

    def test_kev_unit_detected(self):
        import pandas as pd
        df = pd.DataFrame({"counts": [1, 2, 3], "energy[keV]": [10, 20, 30]})
        unit, cts_col, erg = process_df(df)
        assert unit == "keV"

    def test_mev_unit_detected(self):
        import pandas as pd
        df = pd.DataFrame({"counts": [1, 2, 3], "energy[MeV]": [0.1, 0.2, 0.3]})
        unit, cts_col, erg = process_df(df)
        assert unit == "MeV"

    def test_no_counts_column_returns_none(self):
        import pandas as pd
        df = pd.DataFrame({"channel": [1, 2, 3], "energy": [10, 20, 30]})
        unit, cts_col, erg = process_df(df)
        assert cts_col is None

    def test_no_energy_column_returns_none(self):
        import pandas as pd
        df = pd.DataFrame({"counts": [1, 2, 3]})
        unit, cts_col, erg = process_df(df)
        assert erg is None


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------

class TestReadCsv:
    def test_returns_spectrum(self):
        spect = read_csv(CSV_NO_CAL)
        assert isinstance(spect, Spectrum)

    def test_no_cal_has_no_energies(self):
        spect = read_csv(CSV_NO_CAL)
        assert spect.energies is None

    def test_no_cal_x_units_channels(self):
        spect = read_csv(CSV_NO_CAL)
        assert spect.x_units == "Channels"

    def test_with_cal_has_energies(self):
        spect = read_csv(CSV_WITH_CAL)
        assert spect.energies is not None

    def test_with_cal_x_units_energy(self):
        spect = read_csv(CSV_WITH_CAL)
        assert "Energy" in spect.x_units

    def test_counts_positive_length(self):
        spect = read_csv(CSV_NO_CAL)
        assert len(spect.counts) > 0

    def test_counts_are_float(self):
        spect = read_csv(CSV_NO_CAL)
        assert spect.counts.dtype == float

    def test_invalid_file_no_counts_column_raises(self, tmp_path):
        import pandas as pd
        bad_file = tmp_path / "bad.csv"
        pd.DataFrame({"channel": [1, 2, 3]}).to_csv(bad_file, index=False)
        with pytest.raises(ValueError, match="No counts column found"):
            read_csv(bad_file)


# ---------------------------------------------------------------------------
# read_txt
# ---------------------------------------------------------------------------

class TestReadTxt:
    def test_returns_spectrum(self):
        spect = read_txt(TXT_FILE)
        assert isinstance(spect, Spectrum)

    def test_has_energies(self):
        spect = read_txt(TXT_FILE)
        assert spect.energies is not None

    def test_livetime_parsed(self):
        spect = read_txt(TXT_FILE)
        assert spect.livetime == 28800.0

    def test_realtime_parsed(self):
        spect = read_txt(TXT_FILE)
        assert spect.realtime == 36228.0

    def test_description_parsed(self):
        spect = read_txt(TXT_FILE)
        assert spect.description is not None

    def test_label_parsed(self):
        spect = read_txt(TXT_FILE)
        assert spect.label is not None

    def test_counts_positive_length(self):
        spect = read_txt(TXT_FILE)
        assert len(spect.counts) == 16384

    def test_malformed_file_raises(self, tmp_path):
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("Description: test\nLabel: test\n")
        with pytest.raises(ValueError, match="Could not find 'Energy calibration:'"):
            read_txt(bad_file)


# ---------------------------------------------------------------------------
# read_mca
# ---------------------------------------------------------------------------

class TestReadMca:
    def test_returns_spectrum(self):
        spect = read_mca(MCA_FILE)
        assert isinstance(spect, Spectrum)

    def test_counts_positive_length(self):
        spect = read_mca(MCA_FILE)
        assert len(spect.counts) > 0

    def test_livetime_parsed(self):
        spect = read_mca(MCA_FILE)
        assert spect.livetime == 1200.0

    def test_realtime_parsed(self):
        spect = read_mca(MCA_FILE)
        assert spect.realtime == 1203.56

    def test_no_energies(self):
        spect = read_mca(MCA_FILE)
        assert spect.energies is None

    def test_wrong_extension_raises(self):
        with pytest.raises(ValueError, match="Expected a .mca file"):
            read_mca("test.txt")


# ---------------------------------------------------------------------------
# read_spe
# ---------------------------------------------------------------------------

class TestReadSpe:
    def test_returns_spectrum(self):
        spect = read_spe(SPE_FILE)
        assert isinstance(spect, Spectrum)

    def test_counts_positive_length(self):
        spect = read_spe(SPE_FILE)
        assert len(spect.counts) > 0

    def test_livetime_parsed(self):
        spect = read_spe(SPE_FILE)
        assert spect.livetime == 60.0

    def test_realtime_parsed(self):
        spect = read_spe(SPE_FILE)
        assert spect.realtime == 62.0

    def test_no_energies(self):
        """SPE file without energy calibration should have no energies."""
        spect = read_spe(SPE_FILE)
        assert spect.energies is None

    def test_wrong_extension_raises(self):
        with pytest.raises(ValueError, match="Expected a .Spe file"):
            read_spe("test.txt")


# ---------------------------------------------------------------------------
# read_cnf
# ---------------------------------------------------------------------------

class TestReadCnf:
    def test_returns_spectrum(self):
        spect = read_cnf(CNF_FILE)
        assert isinstance(spect, Spectrum)

    def test_counts_positive_length(self):
        spect = read_cnf(CNF_FILE)
        assert len(spect.counts) == 4096

    def test_livetime_parsed(self):
        spect = read_cnf(CNF_FILE)
        assert spect.livetime == pytest.approx(800.0, rel=1e-3)

    def test_realtime_parsed(self):
        spect = read_cnf(CNF_FILE)
        assert spect.realtime == pytest.approx(811.2, rel=1e-3)

    def test_has_energies(self):
        spect = read_cnf(CNF_FILE)
        assert spect.energies is not None

    def test_energy_units_kev(self):
        spect = read_cnf(CNF_FILE)
        assert spect.e_units == "keV"

    def test_total_counts(self):
        spect = read_cnf(CNF_FILE)
        assert spect.counts.sum() > 0
