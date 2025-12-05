"""Curiosity DAN fetch-by-date utilities.

Refactored from dan_vis2_IsWorkingVer_backup.py to be used as an importable
module inside NASA-gamma. Provides helpers to query the PDS index by Earth
date, download matching DAN RDR products, decode CTN/CETN spectra, and
optionally compute count rates vs time.
"""

from __future__ import annotations

from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
#import os
import re
import requests

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
#LBL_NAME   = "dna_398471667rac00110030000_______p1.lbl"
#DAT_NAME   = "dna_398471667rac00110030000_______p1.dat"
FMT_NAME   = "dan_rdr_derived_activ_mod.fmt"
TBINS_NAME = "tbins_us.txt"

# ---------------------------------------------------------------------
# PDS RDR CONFIG (for querying by date)
# ---------------------------------------------------------------------
PDS_VOLUME_ROOT = (
    "https://pds-geosciences.wustl.edu/"
    "msl/msl-m-dan-3_4-rdr-v1/msldan_1xxx"
)
PDS_INDEX_URL   = f"{PDS_VOLUME_ROOT}/index/index.tab"
PDS_INDEX_CACHE = BASE_DIR / "index_cached.tab"

# Byte positions from the DAN RDR INDEX.LBL (1-based START_BYTE, BYTES)
# If something is off, you can tweak these slice widths.
PDS_INDEX_COLS = {
    "VOLUME_ID":          (2,   12),
    "PATH_NAME":          (17,  13),
    "FILE_NAME":          (33,  40),
    "PRODUCT_ID":         (76,  40),
    "PRODUCT_VERSION_ID": (119, 12),
    "PRODUCT_TYPE":       (134, 12),
    "PRODUCT_CREATION":   (148, 23),
    "START_TIME":         (172, 23),
    "STOP_TIME":          (196, 23),
    "SCLK_START":         (221, 16),
    "SCLK_STOP":          (240, 16),
    "PLANET_DAY_NUMBER":  (258, 4),
    "RELEASE_ID":         (264, 4),
}


# ---------------------------------------------------------------------
# Label & FMT parsing
# ---------------------------------------------------------------------
def parse_lbl_for_record_bytes(lbl_path: Path) -> int:
    """Parse RECORD_BYTES from a simple PDS3 label."""
    record_bytes = None
    with lbl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("RECORD_BYTES"):
                _, rhs = line.split("=", 1)
                record_bytes = int(rhs.strip())
                break
    if record_bytes is None:
        raise RuntimeError(f"Could not find RECORD_BYTES in {lbl_path}")
    return record_bytes


def read_fmt_columns(fmt_path: Path):
    """
    Parse a PDS-style .FMT file defining columns.

    Returns
    -------
    columns : list of dict
        Each dict has NAME, DATA_TYPE, START_BYTE, BYTES, ITEMS, ITEM_BYTES (if present).
    """
    columns = []
    current = None

    with fmt_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if line.upper().startswith("OBJECT") and "COLUMN" in line.upper():
                current = {}
            elif line.upper().startswith("END_OBJECT") and "COLUMN" in line.upper():
                if current is not None:
                    columns.append(current)
                    current = None
            elif current is not None and "=" in line:
                key, val = [s.strip() for s in line.split("=", 1)]
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                current[key.upper()] = val

    # Normalize numeric things and sort by START_BYTE
    for col in columns:
        for k in ["START_BYTE", "BYTES", "ITEMS", "ITEM_BYTES"]:
            if k in col:
                try:
                    col[k] = int(col[k])
                except Exception:
                    pass

    columns.sort(key=lambda c: c.get("START_BYTE", 0))

    # Ensure NAME exists
    for col in columns:
        if "NAME" not in col:
            raise RuntimeError(f"Column missing NAME: {col}")

    return columns


def column_struct_for(col):
    """
    Build a struct format string for a single column.

    Uses DATA_TYPE, BYTES, ITEMS, ITEM_BYTES.
    """
    dtype = col.get("DATA_TYPE", "").upper()
    total_bytes = int(col.get("BYTES", 0))
    n_items = int(col.get("ITEMS", 1))
    item_bytes = col.get("ITEM_BYTES")
    if not item_bytes:
        item_bytes = total_bytes // max(n_items, 1)

    if "MSB_UNSIGNED_INTEGER" in dtype:
        if item_bytes == 1:
            code = "B"
        elif item_bytes == 2:
            code = "H"
        elif item_bytes == 4:
            code = "I"
        else:
            raise ValueError(f"Unsupported unsigned int bytes: {item_bytes}")
    elif "MSB_INTEGER" in dtype:
        if item_bytes == 1:
            code = "b"
        elif item_bytes == 2:
            code = "h"
        elif item_bytes == 4:
            code = "i"
        else:
            raise ValueError(f"Unsupported signed int bytes: {item_bytes}")
    elif "IEEE_REAL" in dtype:
        if item_bytes == 4:
            code = "f"
        elif item_bytes == 8:
            code = "d"
        else:
            raise ValueError(f"Unsupported float bytes: {item_bytes}")
    elif "CHARACTER" in dtype:
        code = f"{item_bytes}s"
        n_items = 1
        total_bytes = item_bytes
    else:
        raise ValueError(f"Unsupported DATA_TYPE: {dtype}")

    fmt = ">" + code * n_items
    return fmt, n_items, total_bytes


# ---------------------------------------------------------------------
# Binary table decoding
# ---------------------------------------------------------------------
def decode_dan_table(dat_path: Path, columns, record_bytes: int):
    """
    Decode the DAN EDR binary table using column-by-column slicing.

    Returns
    -------
    data : dict
        Keys: column names
        Values: numpy arrays of shape (n_records, ITEMS)
    """
    with dat_path.open("rb") as f:
        blob = f.read()

    n_records = len(blob) // record_bytes
    #print(f"Found {n_records} records in {dat_path.name}")

    # Precompute per-column formats
    for col in columns:
        fmt, n_items, total_bytes = column_struct_for(col)
        col["FMT"] = fmt
        col["NVAL"] = n_items
        col["TOTBYTES"] = total_bytes

    names = [c["NAME"] for c in columns]
    out = {name: [] for name in names}

    for i in range(n_records):
        rec_start = i * record_bytes
        rec = blob[rec_start : rec_start + record_bytes]

        for col in columns:
            name = col["NAME"]
            start_byte = int(col["START_BYTE"])
            total_bytes = col["TOTBYTES"]
            fmt = col["FMT"]
            n_items = col["NVAL"]

            offset = start_byte - 1
            seg = rec[offset : offset + total_bytes]
            vals = struct.unpack(fmt, seg)

            if "CHARACTER" in col.get("DATA_TYPE", "").upper():
                v = vals[0]
            else:
                if n_items == 1:
                    v = vals[0]
                else:
                    v = np.array(vals)

            out[name].append(v)

    for name in out:
        arr = out[name]
        if isinstance(arr[0], np.ndarray):
            out[name] = np.stack(arr, axis=0)
        else:
            out[name] = np.array(arr)

    return out


# ---------------------------------------------------------------------
# Time bins + CTN/CETN helpers
# ---------------------------------------------------------------------
def load_tbins(tbin_path: Path) -> np.ndarray:
    """Load DAN time-bin upper edges from tbins_us.txt."""
    return np.loadtxt(tbin_path, usecols=1)


def find_ctn_cetn_names(column_names):
    """Guess CTN and CETN columns by name, ignoring background (BKGD)."""
    u = [n.upper() for n in column_names]

    def find(substr, exclude="BKGD"):
        for name, uu in zip(column_names, u):
            if substr in uu and (exclude is None or exclude not in uu):
                return name
        raise KeyError(f"Could not find column with '{substr}' in {column_names}")

    ctn_name = find("CTN")
    cetn_name = find("CETN")
    return ctn_name, cetn_name


def debug_spectra(table):
    """Print some sanity-check info about CTN/CETN so you can see if decoding worked."""
    column_names = list(table.keys())
    ctn_name, cetn_name = find_ctn_cetn_names(column_names)

    ctn = table[ctn_name]
    cetn = table[cetn_name]

    # print("\n=== CTN/CETN DEBUG ===")
    # print("CTN shape:", ctn.shape, "CETN shape:", cetn.shape)
    # print("CTN non-zero?", np.any(ctn != 0), "CETN non-zero?", np.any(cetn != 0))


def find_first_nonzero_record(table):
    """
    Find the first record index where either CTN or CETN spectrum
    has any non-zero counts.
    """
    column_names = list(table.keys())
    ctn_name, cetn_name = find_ctn_cetn_names(column_names)

    ctn = table[ctn_name]
    cetn = table[cetn_name]

    n_records = ctn.shape[0]
    for i in range(n_records):
        if np.any(ctn[i] != 0) or np.any(cetn[i] != 0):
            return i
    # If everything is zero, just return 0
    return 0


# ---------------------------------------------------------------------
# PDS RDR helpers: query DAN active data by Earth date
# ---------------------------------------------------------------------
def _slice_field(line: str, start_byte: int, n_bytes: int) -> str:
    """
    Slice a fixed-width field from INDEX.TAB.

    INDEX.LBL uses 1-based START_BYTE; we convert to 0-based.
    """
    start = start_byte - 1
    end = start + n_bytes
    return line[start:end].strip()


def _parse_index_line(line: str) -> dict:
    """
    Parse a single line of the DAN RDR INDEX.TAB into a dict.
    """
    f = {}
    for name, (start_byte, n_bytes) in PDS_INDEX_COLS.items():
        f[name] = _slice_field(line, start_byte, n_bytes)

    # Parse times (PRODUCT_CREATION, START_TIME, STOP_TIME)
    for tkey in ("PRODUCT_CREATION", "START_TIME", "STOP_TIME"):
        t = f[tkey]
        if t:
            # formats like: 2013-04-04T12:34:56.789
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    f[tkey] = dt.datetime.strptime(t, fmt)
                    break
                except ValueError:
                    continue
            else:
                f[tkey] = None
        else:
            f[tkey] = None

    # SOL
    try:
        f["PLANET_DAY_NUMBER"] = int(f["PLANET_DAY_NUMBER"])
    except ValueError:
        f["PLANET_DAY_NUMBER"] = None

    # Strip whitespace from key fields
    f["PRODUCT_TYPE"] = f["PRODUCT_TYPE"].strip()
    f["PATH_NAME"]    = f["PATH_NAME"].strip()
    f["FILE_NAME"]    = f["FILE_NAME"].strip()

    return f


def _get_index_lines_cached():
    """
    Return all lines of INDEX.TAB from a local cache if available,
    otherwise download once from PDS and cache it.
    """
    if PDS_INDEX_CACHE.exists():
        with PDS_INDEX_CACHE.open("r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
        return

    # Cache miss: download and save
    resp = requests.get(PDS_INDEX_URL, stream=True)
    resp.raise_for_status()

    with PDS_INDEX_CACHE.open("w", encoding="utf-8") as f:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.rstrip("\n")
            f.write(line + "\n")
            yield line


def _iter_index_rows():
    """
    Stream the DAN RDR index table (from cache or PDS) and yield parsed rows.
    """
    for line in _get_index_lines_cached():
        if not line.strip():
            continue
        yield _parse_index_line(line)


def _label_uses_derived_active(label_text: str) -> bool:
    """
    Check if a label references the DAN_RDR_DERIVED_ACTIV.FMT structure.
    """
    return "DAN_RDR_DERIVED_ACTIV.FMT" in label_text.upper()


def _get_label_and_data_urls(row: dict,
                             enforce_derived_activ: bool = True):
    """
    For one INDEX row, construct label URL, fetch it, and (if valid)
    return (lbl_url, data_url, lbl_text).

    If enforce_derived_activ is True, we only accept labels that
    reference DAN_RDR_DERIVED_ACTIV.FMT.
    """
    # PATH_NAME is like "/data/sol0xxx/...", relative to the volume root
    rel_dir = row["PATH_NAME"].lstrip("/")
    lbl_url = f"{PDS_VOLUME_ROOT}/{rel_dir}/{row['FILE_NAME']}"

    # Pull the label
    lbl_resp = requests.get(lbl_url)
    lbl_resp.raise_for_status()
    lbl_text = lbl_resp.text

    if enforce_derived_activ and not _label_uses_derived_active(lbl_text):
        return None, None, None

    # Find the referenced .DAT/.TAB file in the label
    m = re.search(r'"([^"]+\.(?:DAT|TAB))"', lbl_text, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not find data file reference in {lbl_url}")

    data_fname = m.group(1)
    data_url   = f"{PDS_VOLUME_ROOT}/{rel_dir}/{data_fname}"

    return lbl_url, data_url, lbl_text


def find_dan_active_by_date(start_date: dt.date,
                            end_date: dt.date,
                            product_types=("DAN_RDR_AC",),
                            enforce_derived_activ: bool = True):
    """
    Generator over DAN active products whose START_TIME date lies in
    [start_date, end_date].

    Critically: we only download the label when we already know
    the row passes the date + PRODUCT_TYPE filters.
    """
    for row in _iter_index_rows():
        pt = row["PRODUCT_TYPE"]
        if pt not in product_types:
            continue

        start_time = row["START_TIME"]
        if start_time is None:
            continue

        d = start_time.date()
        if d < start_date or d > end_date:
            continue

        # At this point we know: correct type + date range.
        # Now (and only now) hit the label.
        lbl_url, data_url, lbl_text = _get_label_and_data_urls(
            row,
            enforce_derived_activ=enforce_derived_activ,
        )
        if lbl_url is None:
            continue

        yield {
            "row": row,
            "label_url": lbl_url,
            "data_url": data_url,
            "label_text": lbl_text,
        }


def download_product_from_pds(label_url: str,
                              data_url: str,
                              out_dir: Path | None = None):
    """
    Download label + data files into a local directory and return their Paths.
    """
    if out_dir is None:
        out_dir = BASE_DIR / "pds_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []

    for url in (label_url, data_url):
        fname = url.split("/")[-1]
        out_path = out_dir / fname
        if not out_path.exists():
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        local_paths.append(out_path)

    return tuple(local_paths)


def load_from_pds_by_earth_date(
    date_str: str,
    product_types=("DAN_RDR_AC",),
    enforce_derived_activ: bool = True,
    fallback: str | None = None,
):
    """
    Find a DAN active product for a given Earth date, download it,
    and return (lbl_path, dat_path).

    Parameters
    ----------
    date_str : "YYYY-MM-DD"
    fallback : {"nearest", "next", "previous", None}
    """
    target_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

    # 1) Try exact date first — this only hits labels for matching rows
    for prod in find_dan_active_by_date(
        target_date,
        target_date,
        product_types=product_types,
        enforce_derived_activ=enforce_derived_activ,
    ):
        lbl_url  = prod["label_url"]
        data_url = prod["data_url"]
        return download_product_from_pds(lbl_url, data_url)

    # 2) No exact match → handle fallback
    if fallback is None:
        raise RuntimeError(
            f"No DAN active products found on {date_str} with "
            f"PRODUCT_TYPE in {product_types} and "
            f"derived-active FMT={enforce_derived_activ}"
        )

    # This is cheap now: uses only INDEX.TAB from cache
    active_dates = list_active_dates(product_types=product_types)

    if not active_dates:
        raise RuntimeError(
            "No DAN active dates found at all for the given constraints "
            f"(PRODUCT_TYPE in {product_types})."
        )

    if fallback not in ("nearest", "next", "previous"):
        raise ValueError(f"Unknown fallback mode: {fallback}")

    if fallback == "nearest":
        diffs = [(abs((d - target_date).days), d) for d in active_dates]
        diffs.sort()
        chosen_date = diffs[0][1]

    elif fallback == "next":
        future = [d for d in active_dates if d >= target_date]
        if not future:
            raise RuntimeError(
                f"No future DAN active dates found after {target_date}."
            )
        chosen_date = future[0]

    else:  # "previous"
        past = [d for d in active_dates if d <= target_date]
        if not past:
            raise RuntimeError(
                f"No past DAN active dates found before {target_date}."
            )
        chosen_date = past[-1]

    # Now do a second pass restricted to chosen_date
    for prod in find_dan_active_by_date(
        chosen_date,
        chosen_date,
        product_types=product_types,
        enforce_derived_activ=enforce_derived_activ,
    ):
        lbl_url  = prod["label_url"]
        data_url = prod["data_url"]
        return download_product_from_pds(lbl_url, data_url)

    raise RuntimeError(
        f"Unexpectedly found no DAN active products on chosen date {chosen_date}."
    )


def plot_single_spectrum(table, tbins_us, record_index=0, max_time_ms=10.0):
    """Plot CTN and CETN count rate vs time, restricted to the first max_time_ms."""
    column_names = list(table.keys())
    ctn_name, cetn_name = find_ctn_cetn_names(column_names)
    ctn_spec = table[ctn_name][record_index]
    cetn_spec = table[cetn_name][record_index]

    # Convert bin edges from microseconds to milliseconds
    tbins_ms = tbins_us / 1000.0
    t_edges_ms = tbins_ms

    # Bin centers & widths
    t_centers_ms = 0.5 * (t_edges_ms[:-1] + t_edges_ms[1:])
    dt_ms = np.diff(t_edges_ms)

    # Match array sizes
    n_bins = min(len(t_centers_ms), len(dt_ms), len(ctn_spec), len(cetn_spec))
    t_centers_ms = t_centers_ms[:n_bins]
    dt_ms       = dt_ms[:n_bins]
    ctn_spec    = ctn_spec[:n_bins]
    cetn_spec   = cetn_spec[:n_bins]

    # Convert counts → count rate
    ctn_rate  = ctn_spec  / dt_ms
    cetn_rate = cetn_spec / dt_ms

    # Restrict to first max_time_ms
    mask = t_centers_ms <= max_time_ms
    t_centers_ms = t_centers_ms[mask]
    ctn_rate = ctn_rate[mask]
    cetn_rate = cetn_rate[mask]

    # Plot
    plt.figure()
    plt.step(t_centers_ms, ctn_rate, where="mid", label="Thermal")
    plt.step(t_centers_ms, cetn_rate, where="mid", label="Epithermal")
    plt.xlabel("Time after pulse [ms]")
    plt.ylabel("Count rate [counts/ms]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_from_pds(date_str: str,
                 record_index: int | None = None,
                 max_time_ms: float = 0.25):
    """
    High-level convenience entry point:

    - look up DAN active RDR on the given Earth date (UTC),
    - download label + data,
    - decode using your local FMT (dan_rdr_derived_activ_mod.fmt),
    - plot CTN/CETN count rate vs time for one record.
    """
    # 1) Pull matching product from PDS
    lbl_path, dat_path = load_from_pds_by_earth_date(date_str)

    fmt_path   = BASE_DIR / FMT_NAME
    tbins_path = BASE_DIR / TBINS_NAME

    record_bytes = parse_lbl_for_record_bytes(lbl_path)
    columns      = read_fmt_columns(fmt_path)
    table        = decode_dan_table(dat_path, columns, record_bytes)
    tbins_us     = load_tbins(tbins_path)

    debug_spectra(table)

    if record_index is None:
        record_index = find_first_nonzero_record(table)

    plot_single_spectrum(
        table,
        tbins_us,
        record_index=record_index,
        max_time_ms=max_time_ms,
    )


def list_active_dates(product_types=("DAN_RDR_AC",)):
    """
    Return a sorted list of active DAN dates as strings in the form "YYYY-MM-DD".
    Uses only INDEX.TAB (fast; no network calls except the cached download).
    """
    dates = set()

    for row in _iter_index_rows():
        pt = row["PRODUCT_TYPE"]
        if pt not in product_types:
            continue

        start_time = row["START_TIME"]
        if start_time is None:
            continue

        d = start_time.date()
        dates.add(d)

    return sorted(dates)


def load_dan_by_date(
    date_str: str,
    record_index: int | None = None,
    fallback: str | None = "nearest",
    max_time_ms: float | None = None,
    product_types=("DAN_RDR_AC",),
    enforce_derived_activ: bool = True,
):
    """
    High-level helper: fetch and decode DAN active data for a given Earth date.

    This wraps the lower-level helpers in this module:

    - looks up DAN active RDR on the given Earth date (UTC),
    - downloads label + data into the local pds_cache,
    - decodes using the local FMT (dan_rdr_derived_activ_mod.fmt),
    - returns a dict with:
        * "table"     : raw decoded table (columns -> arrays),
        * "tbins_us"  : time-bin upper edges [microseconds],
        * "t_ms"      : bin centers [ms] (if max_time_ms is not None),
        * "ctn_rate"  : thermal count rate [counts/ms],
        * "cetn_rate" : epithermal count rate [counts/ms],
        * "record_index" : index of the record used,
        * "lbl_path", "dat_path" : local cache paths.

    Parameters
    ----------
    date_str : str
        Target Earth date in "YYYY-MM-DD" format.
    record_index : int or None
        If None, the first record with non-zero CTN/CETN is auto-selected.
    fallback : {"nearest", "next", "previous", None}
        Behaviour if no exact product exists on the given date.
    max_time_ms : float or None
        If given, truncate the time axis to t <= max_time_ms. If None, full range.
    product_types, enforce_derived_activ :
        Passed through to load_from_pds_by_earth_date.
    """
    # 1) Pull matching product from PDS
    lbl_path, dat_path = load_from_pds_by_earth_date(
        date_str,
        product_types=product_types,
        enforce_derived_activ=enforce_derived_activ,
        fallback=fallback,
    )

    fmt_path   = BASE_DIR / FMT_NAME
    tbins_path = BASE_DIR / TBINS_NAME

    record_bytes = parse_lbl_for_record_bytes(lbl_path)
    columns      = read_fmt_columns(fmt_path)
    table        = decode_dan_table(dat_path, columns, record_bytes)
    tbins_us     = load_tbins(tbins_path)

    # Auto-pick a record with signal if not specified
    if record_index is None:
        record_index = find_first_nonzero_record(table)

    # Pull out CTN/CETN spectra and compute count rates,
    # reusing the same logic as plot_single_spectrum but without plotting.
    column_names = list(table.keys())
    ctn_name, cetn_name = find_ctn_cetn_names(column_names)
    ctn_spec = table[ctn_name][record_index]
    cetn_spec = table[cetn_name][record_index]

    tbins_ms = tbins_us / 1000.0
    t_edges_ms = tbins_ms

    # Bin centers & widths
    t_centers_ms = 0.5 * (t_edges_ms[:-1] + t_edges_ms[1:])
    dt_ms = np.diff(t_edges_ms)

    n_bins = min(len(t_centers_ms), len(dt_ms), len(ctn_spec), len(cetn_spec))
    t_centers_ms = t_centers_ms[:n_bins]
    dt_ms       = dt_ms[:n_bins]
    ctn_spec    = ctn_spec[:n_bins]
    cetn_spec   = cetn_spec[:n_bins]

    ctn_rate  = ctn_spec  / dt_ms
    cetn_rate = cetn_spec / dt_ms

    if max_time_ms is not None:
        mask = t_centers_ms <= max_time_ms
        t_centers_ms = t_centers_ms[mask]
        ctn_rate = ctn_rate[mask]
        cetn_rate = cetn_rate[mask]

    return {
        "table": table,
        "tbins_us": tbins_us,
        "t_ms": t_centers_ms,
        "ctn_rate": ctn_rate,
        "cetn_rate": cetn_rate,
        "record_index": record_index,
        "lbl_path": lbl_path,
        "dat_path": dat_path,
    }
