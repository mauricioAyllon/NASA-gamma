"""
Test PiecewiseLinearCalibration with the test-nonlinear parquet spectrum.

The parquet file has columns: channels, counts.
We synthesise realistic calibration points that match a detector whose
keV/channel ratio shifts at 3000 keV, then apply the class and produce:
  1. A calibration plot (fit quality, both segments, breakpoint).
  2. The spectrum plotted in energy units.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nasagamma import energy_calibration as ecal

# ---------------------------------------------------------------------------
# Load spectrum
# ---------------------------------------------------------------------------
df = pd.read_parquet("../data/test-nonlinear.parquet")
channels = df["channels"].to_numpy(dtype=float)
counts = df["counts"].to_numpy(dtype=float)

print(f"Spectrum: {len(channels)} bins, "
      f"ch range [{channels.min():.0f}, {channels.max():.0f}]")

# ---------------------------------------------------------------------------
# Synthetic calibration points
#
# We pretend a detector with two linear regions:
#   Lower segment (E < 3000 keV):  E = 0.100 * ch + 5.0
#   Upper segment (E >= 3000 keV): E_break + 0.082 * (ch - ch_break)
#
# ch_break from lower eqn: (3000 - 5) / 0.100 = 29950
# So channels 0–29950 → 0–3000 keV (slope 0.100 keV/ch)
#      channels 29950–65528 → 3000–6000 keV (slope 0.082 keV/ch, slightly compressed)
#
# We add small Gaussian noise to simulate real peak fitting uncertainties.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

TRUE_SLOPE1    = 0.100   # keV / channel
TRUE_INTERCEPT = 5.0     # keV
E_BREAK        = 3000.0  # keV
TRUE_CH_BREAK  = (E_BREAK - TRUE_INTERCEPT) / TRUE_SLOPE1   # ≈ 29950
TRUE_SLOPE2    = 0.082   # keV / channel (slope above breakpoint)
NOISE_KEV      = 1.5     # keV standard deviation of calibration-peak uncertainty

# --- lower calibration peaks (8 peaks below 3000 keV) ---
cal_ch_low  = np.linspace(2000, 28000, 8)
cal_erg_low = TRUE_SLOPE1 * cal_ch_low + TRUE_INTERCEPT + rng.normal(0, NOISE_KEV, 8)

# --- upper calibration peaks (5 peaks above 3000 keV) ---
cal_ch_high  = np.linspace(32000, 62000, 5)
cal_erg_high = (E_BREAK
                + TRUE_SLOPE2 * (cal_ch_high - TRUE_CH_BREAK)
                + rng.normal(0, NOISE_KEV, 5))

mean_vals = np.concatenate([cal_ch_low, cal_ch_high])
erg       = np.concatenate([cal_erg_low, cal_erg_high])

print(f"\nCalibration points: {len(mean_vals)} total "
      f"({len(cal_ch_low)} low, {len(cal_ch_high)} high)")

# ---------------------------------------------------------------------------
# Fit PiecewiseLinearCalibration
# ---------------------------------------------------------------------------
pcal = ecal.PiecewiseLinearCalibration(
    mean_vals=mean_vals,
    erg=erg,
    channels=channels,
    e_break=E_BREAK,
    e_units="keV",
)

meta = pcal.metadata()
print("\n--- Calibration metadata ---")
for k, v in meta.items():
    if k not in ("mean_vals", "erg"):
        print(f"  {k}: {v}")

print(f"\nTrue slope1={TRUE_SLOPE1:.4f}  fitted slope1={meta['slope1']:.4f}")
print(f"True slope2={TRUE_SLOPE2:.4f}  fitted slope2={meta['slope2']:.4f}")
print(f"True ch_break={TRUE_CH_BREAK:.1f}  fitted ch_break={meta['ch_break']:.1f}")

# ---------------------------------------------------------------------------
# Figure: calibration plot + spectrum in energy units
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

# Panel 1 – calibration fit
ax_cal = fig.add_subplot(gs[0])
pcal.plot(ax=ax_cal)
ax_cal.set_title(
    f"Piecewise linear calibration  |  "
    rf"$R^2_{{low}}$ = {meta['r2_lower']:.5f},  "
    rf"$R^2_{{high}}$ = {meta['r2_upper']:.5f}",
    fontsize=13,
)

# Panel 2 – spectrum in energy space
ax_spec = fig.add_subplot(gs[1])
energies = pcal.predicted           # energy for every channel bin

ax_spec.stairs(
    counts,
    np.append(energies, energies[-1] + (energies[-1] - energies[-2])),
    color="royalblue",
    lw=1.0,
    fill=True,
    alpha=0.6,
    label="Spectrum",
)
ax_spec.axvline(
    E_BREAK,
    ls=":",
    lw=2,
    color="gray",
    label=f"Breakpoint = {E_BREAK:.0f} keV",
)
ax_spec.set_xlabel("Energy [keV]")
ax_spec.set_ylabel("Counts")
ax_spec.set_title("Spectrum after piecewise energy calibration", fontsize=13)
ax_spec.legend(fontsize=11)
ax_spec.set_yscale("log")
ax_spec.set_xlim([energies.min(), energies.max()])

