# nasagamma/data_fetcher.py
# -*- coding: utf-8 -*-
"""
Stdlib-only PDS data fetcher with mission-specific adapters.

- DAWN: XML + TAB pairs; filters by YYMMDD-YYMMDD window in filename
- Lunar Prospector: DAT + LBL pairs; filters by YYYY_DDD (day-of-year)

Public API:
    fetch_mission(mission, start_dt=None, end_dt=None,
                  progress_callback=None, cancel_flag=None, dest_dir=None)
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from urllib.request import urlopen, Request
from html.parser import HTMLParser


# Tiny HTML link parser
class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for (k, v) in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def _http_get_text(url: str, timeout: float = 30.0) -> str:
    # Some servers dislike default user-agent; set a simple UA.(fixed some issues)
    req = Request(url, headers={"User-Agent": "nasagamma-fetcher/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _list_directory(base_url: str) -> List[str]:
    """Return hrefs from an Apache-style index page (non-recursive)."""
    html = _http_get_text(base_url)
    p = _LinkParser()
    p.feed(html)
    # Filter out parent links and directories (keep files only - fixed)
    hrefs = [h for h in p.hrefs if h and h not in ("../", "./")]
    return hrefs


def _write_stream(
    url: str, dest: Path, chunk: int = 1 << 14, timeout: float = 60.0
) -> None:
    req = Request(url, headers={"User-Agent": "nasagamma-fetcher/0.1"})
    with urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)


# Helper functions
def _pkg_root() -> Path:
    return Path(__file__).resolve().parent


def _default_cache_dir(mission_code: str) -> Path:
    return _pkg_root() / "cache" / mission_code


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _within(d: date, start: Optional[date], end: Optional[date]) -> bool:
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def _emit(
    cb, n: Optional[int] = None, total: Optional[int] = None, note: Optional[str] = None
):
    if cb:
        try:
            cb(
                int(n) if n is not None else 0,
                int(total) if total is not None else None,
                str(note) if note else None,
            )
        except Exception:
            pass


def _canceled(cancel_flag) -> bool:
    try:
        return bool(cancel_flag and cancel_flag())
    except Exception:
        return False


# Mission specific information
@dataclass
class Record:
    key: str
    date_start: date
    date_end: date
    files: List[str]


@dataclass
class MissionSpec:
    code: str
    label: str
    base_url: str
    pair_exts: Tuple[str, str]
    list_records: Callable[[str], List[Record]]


# DAWN (XML+TAB)  YYMMDD-YYMMDD
_DAWN_DATE_RE = re.compile(r"(\d{6})-(\d{6})")


def _parse_dawn_dates(stem: str) -> Optional[Tuple[date, date]]:
    m = _DAWN_DATE_RE.search(stem)
    if not m:
        return None

    def to_date(yyMMdd: str) -> date:
        yy = int(yyMMdd[:2])
        year = 2000 + yy  # assume 20YY
        mm = int(yyMMdd[2:4])
        dd = int(yyMMdd[4:6])
        return date(year, mm, dd)

    d0, d1 = to_date(m.group(1)), to_date(m.group(2))
    if d1 < d0:
        d0, d1 = d1, d0
    return d0, d1


def _list_dawn_records(base_url: str) -> List[Record]:
    hrefs = _list_directory(base_url)
    xmls = [h for h in hrefs if h.lower().endswith(".xml")]
    tabs = [h for h in hrefs if h.lower().endswith(".tab")]

    def stem(p: str) -> str:
        return os.path.splitext(p)[0]

    tab_by_stem = {stem(t): t for t in tabs}
    recs: List[Record] = []
    for x in xmls:
        s = stem(x)
        if s not in tab_by_stem:
            continue
        dates = _parse_dawn_dates(s)
        if not dates:
            continue
        d0, d1 = dates
        recs.append(
            Record(key=s, date_start=d0, date_end=d1, files=[x, tab_by_stem[s]])
        )
    return recs


DAWN = MissionSpec(
    code="DAWN",
    label="DAWN GRAND CERES",
    base_url="https://sbnarchive.psi.edu/pds4/dawn/grand/dawn-grand-ceres_1.0/data_calibrated/",
    pair_exts=(".xml", ".tab"),
    list_records=_list_dawn_records,
)


# Lunar Prospector
_LP_RE = re.compile(
    r"(?P<year>\d{4})_(?P<doy>\d{3})_grs\.(?P<ext>dat|lbl)$", re.IGNORECASE
)


def _lp_name_to_date(name: str) -> Optional[date]:
    m = _LP_RE.search(name)
    if not m:
        return None
    year = int(m.group("year"))
    doy = int(m.group("doy"))
    return datetime.strptime(f"{year}-{doy:03d}", "%Y-%j").date()


def _lp_stem(p: str) -> Optional[str]:
    m = _LP_RE.search(p)
    if not m:
        return None
    return f"{m.group('year')}_{m.group('doy')}_grs"


def _list_lp_records(base_url: str) -> List[Record]:
    hrefs = _list_directory(base_url)
    dats = [h for h in hrefs if h.lower().endswith(".dat")]
    lbls = [h for h in hrefs if h.lower().endswith(".lbl")]
    lbl_by = {_lp_stem(l): l for l in lbls if _lp_stem(l)}

    recs: List[Record] = []
    for d in dats:
        s = _lp_stem(d)
        if not s or s not in lbl_by:
            continue
        d0 = _lp_name_to_date(d)
        if not d0:
            continue
        recs.append(Record(key=s, date_start=d0, date_end=d0, files=[d, lbl_by[s]]))
    return recs


LP = MissionSpec(
    code="LP",
    label="Lunar Prospector GRS",
    base_url="https://pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/",
    pair_exts=(".dat", ".lbl"),
    list_records=_list_lp_records,
)


# Registry and public API

MISSION_REGISTRY: Dict[str, MissionSpec] = {
    "DAWN": DAWN,
    "LP": LP,
    "Lunar Prospector": LP,
}


def _resolve_mission(mission: str) -> MissionSpec:
    for k, spec in MISSION_REGISTRY.items():
        if mission.lower() == k.lower() or mission.lower() == spec.label.lower():
            return spec
    raise ValueError(
        f"Unknown mission: {mission}. Options: {', '.join(MISSION_REGISTRY)}"
    )


def fetch_mission(
    mission: str,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    progress_callback: Optional[
        Callable[[int, Optional[int], Optional[str]], None]
    ] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    dest_dir: Optional[Path] = None,
) -> int:
    """
    Download selected mission files into a cache directory.

    Returns: number of files downloaded (counts every file, e.g., xml+tab = 2).
    """
    spec = _resolve_mission(mission)
    if dest_dir is None:
        dest_dir = _default_cache_dir(spec.code)
    _ensure_dir(dest_dir)

    _emit(progress_callback, note=f"Listing {spec.label} records…")
    records = spec.list_records(spec.base_url)

    sdate = start_dt.date() if start_dt else None
    edate = end_dt.date() if end_dt else None

    # Keep record if either its start or end date falls within the window
    filtered: List[Record] = []
    for r in records:
        if (
            (sdate is None and edate is None)
            or _within(r.date_start, sdate, edate)
            or _within(r.date_end, sdate, edate)
        ):
            filtered.append(r)

    total_files = sum(len(r.files) for r in filtered)
    done_files = 0

    if not filtered:
        _emit(progress_callback, note="No records matched the date filter.")
        return 0

    _emit(progress_callback, 0, total_files, f"Downloading {len(filtered)} record(s)…")

    for rec in filtered:
        if _canceled(cancel_flag):
            _emit(progress_callback, note="Canceled.")
            break

        for relname in rec.files:
            if _canceled(cancel_flag):
                break
            url = urljoin(spec.base_url, relname)
            dest = dest_dir / relname
            if dest.exists():
                done_files += 1
                _emit(progress_callback, done_files, total_files, f"Exists: {relname}")
                continue
            try:
                _emit(
                    progress_callback,
                    done_files,
                    total_files,
                    f"Downloading: {relname}",
                )
                _ensure_dir(dest.parent)
                _write_stream(url, dest)
                done_files += 1
                _emit(progress_callback, done_files, total_files, f"Saved: {relname}")
            except Exception as e:
                _emit(
                    progress_callback,
                    done_files,
                    total_files,
                    f"Failed: {relname} ({e})",
                )

        time.sleep(0.005)  # tiny pause to keep UI responsive if needed (fixed)

    return done_files


# CLI for testing (can be removed later)
def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Stdlib-only PDS fetch")
    ap.add_argument("--mission", required=True, help="DAWN | LP | Lunar Prospector")
    ap.add_argument("--start", help="YYYY-MM-DD (optional)")
    ap.add_argument("--end", help="YYYY-MM-DD (optional)")
    ap.add_argument("--dest", help="Destination directory (optional)")
    args = ap.parse_args()

    sdt = datetime.fromisoformat(args.start) if args.start else None
    edt = datetime.fromisoformat(args.end) if args.end else None
    dest = Path(args.dest) if args.dest else None

    def cb(n, total=None, note=None):
        if total and total > 0:
            pct = int(n * 100 / total)
            sys.stdout.write(f"\r{n}/{total} ({pct:3d}%)  {note or ''}   ")
        else:
            sys.stdout.write(f"\r      {note or ''}   ")
        sys.stdout.flush()

    print(f"\nFetching mission={args.mission} start={args.start} end={args.end}\n")
    count = fetch_mission(
        args.mission, sdt, edt, progress_callback=cb, cancel_flag=None, dest_dir=dest
    )
    print(f"\n\nDone. Downloaded {count} files.\n")


if __name__ == "__main__":
    _cli()
