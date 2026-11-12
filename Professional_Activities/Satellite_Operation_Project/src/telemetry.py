"""
Shared data access and plot style for the SpaceOps take-home test.

Telemetry files: Support Files/<satellite>.csv, one per satellite
-----------------------------------------------------------------
timestamp      Unix epoch in milliseconds, UTC.
               Window: 2019-05-30 10:00 -> 2019-06-02 10:00 UTC (72 h).
gyro_x/y/z     Body angular rates [deg/s] around the three body axes (MEMS gyros, ~10 Hz).
               Raw counts of 1/262.4 deg/s (~0.0038 deg/s). While pointing at the Earth
               the body only turns at about the orbital rate (360 deg / ~95 min ~ 0.06 deg/s);
               slews are short and fast; rates above 2 deg/s that last mean the
               satellite is tumbling.
wheel_s/x/y/z  Reaction wheel speeds [rpm] (~every 10 s, 100 rpm steps, reported as
               magnitudes >= 0). The x, y and z wheels sit on the body axes, s is the
               skewed redundant wheel. The body is turned by speeding wheels up or down
               (exchange of angular momentum). At 8000 rpm a wheel is saturated: it
               cannot store more momentum, so control on that axis is lost.
low_voltage    Flag (0/1) raised on board when the bus voltage drops below a threshold.
               The satellite then switches off loads such as the camera.
bus_voltage    Main power bus voltage [V]. 6.7-8.3 V here, i.e. a 2-cell Li-ion battery
               (8.4 V full, ~6 V empty). It charges in sunlight and drains in eclipse,
               so it follows the ~95 min orbit.

Beta file: Support Files/beta_sun_deg.csv
-----------------------------------------
Daily beta angle [deg] from 2018-06-02 to 2019-06-02: angle between the Sun direction
and the orbital plane. It sets the eclipse length (longest at beta = 0, none at high
|beta|). Satellites sharing an orbital plane share the same beta curve.

Format pitfalls handled here
----------------------------
* Asynchronous packets. Each row only carries the channels sampled at that instant
  (gyros ~10 Hz, wheels ~10 s, voltage ~1 s to ~90 s). An empty cell means "not
  sampled", so each channel must be read on its own (`channel()`), never row-wise,
  and plotted lines must be built from the non-empty values only.
* Fill values. delta reports bus_voltage = 0 V and low_voltage = 0 on every row.
  A powered satellite cannot read 0 V, and a real 0 V would raise the flag, so the
  power channels of a satellite that never reports a positive voltage are set to
  NaN (= not telemetered).
* Glitches. Bus voltages outside the physical range of the battery
  (BUS_VOLTAGE_RANGE) are single-sample errors (foxtrot reads 2.07 V once, between
  8.30 V samples, with the flag still at 0) and are dropped.
* Outages. Some satellites go up to 18 h without any packet (echo). `break_gaps()`
  inserts NaN so plotted lines do not bridge an outage.
* Beta file. Dates are m/d/yyyy in an unnamed first column and the satellites come
  in a different order; `load_beta()` returns them in the standard order.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "inputs"
PLOTS_DIR = ROOT / "plots"

SATELLITES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
GYROS = ["gyro_x", "gyro_y", "gyro_z"]
WHEELS = ["wheel_s", "wheel_x", "wheel_y", "wheel_z"]
POWER = ["low_voltage", "bus_voltage"]
CHANNELS = GYROS + WHEELS + POWER

LABELS = {
    "gyro_x": "Gyro rate X", "gyro_y": "Gyro rate Y", "gyro_z": "Gyro rate Z",
    "wheel_s": "Wheel S (skew)", "wheel_x": "Wheel X", "wheel_y": "Wheel Y", "wheel_z": "Wheel Z",
    "low_voltage": "Low-voltage flag", "bus_voltage": "Bus voltage",
}
UNITS = {**{c: "deg/s" for c in GYROS}, **{c: "rpm" for c in WHEELS},
         "low_voltage": "0/1", "bus_voltage": "V"}

# Limits from the brief, plus the physical range of a 2-cell Li-ion bus.
WHEEL_SATURATION_RPM = 8000.0
GYRO_RATE_LIMIT_DEG_S = 2.0
BUS_VOLTAGE_RANGE = (5.5, 9.0)
MAX_GAP = pd.Timedelta("10min")  # longer silences are drawn as breaks in the lines

# Colors: one fixed hue per satellite and per axis, status red for limits.
SAT_COLORS = dict(zip(SATELLITES, [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
]))
AXIS_COLORS = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a", "s": "#6250d6"}
LIMIT_COLOR = "#d03b3b"   # thresholds (saturation, 2 deg/s)
FLAG_COLOR = "#ec835a"    # low-voltage periods
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def axis_color(col: str) -> str:
    """Color of a gyro or wheel channel, from its axis letter (gyro_x -> x)."""
    return AXIS_COLORS[col.rsplit("_", 1)[1]]


def load_satellite(name: str, verbose: bool = True) -> pd.DataFrame:
    """Read one satellite CSV and return it cleaned, indexed by UTC time (naive datetime)."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", dtype={"timestamp": "int64"})
    df["time"] = pd.to_datetime(df.pop("timestamp"), unit="ms")
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df = df[CHANNELS].astype(float)

    print(df[CHANNELS])

    notes = []
    if not (df["bus_voltage"] > 0).any(): #No value in the bus_voltage column >0
        df[POWER] = np.nan
        notes.append("power channels are a constant 0 fill value -> not telemetered")
    glitch = df["bus_voltage"].notna() & ~df["bus_voltage"].between(*BUS_VOLTAGE_RANGE)
    if glitch.any():
        values = ", ".join(f"{v:.2f} V" for v in df.loc[glitch, "bus_voltage"]) #to be printed
        df.loc[glitch, "bus_voltage"] = np.nan
        notes.append(f"dropped {glitch.sum()} bus voltage glitch(es) outside "
                     f"{BUS_VOLTAGE_RANGE} V: {values}")
    df = df.dropna(how="all")  # rows that only carried fill values


    df.attrs["name"] = name
    df.attrs["notes"] = notes
    if verbose:
        for note in notes:
            print(f"  [{name}] {note}")
    return df


def load_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Load every satellite, in the standard order."""
    return {name: load_satellite(name, verbose) for name in SATELLITES}


def load_beta() -> pd.DataFrame:
    """Daily beta angle [deg], indexed by date, one column per satellite (standard order)."""
    beta = pd.read_csv(DATA_DIR / "beta_sun_deg.csv", index_col=0)
    beta.index = pd.to_datetime(beta.index, format="%m/%d/%Y")
    beta.index.name = "date"
    return beta[SATELLITES]


def channel(df: pd.DataFrame, col: str) -> pd.Series:
    """One channel with its non-sampled epochs removed."""
    return df[col].dropna()


def break_gaps(series: pd.Series, max_gap: pd.Timedelta = MAX_GAP) -> pd.Series:
    """Insert a NaN inside every gap longer than max_gap so a plotted line breaks there."""
    if len(series) < 2:
        return series
    gap_ends = series.index[1:][np.diff(series.index.values) > max_gap.to_timedelta64()]
    if len(gap_ends) == 0:
        return series
    holes = pd.Series(np.nan, index=gap_ends - pd.Timedelta(milliseconds=1))
    return pd.concat([series, holes]).sort_index()


def break_step_gaps(series: pd.Series, max_gap: pd.Timedelta = MAX_GAP) -> pd.Series:
    """Same as break_gaps, for a step line (drawstyle="steps-post").

    A step line holds its level until the next point, so the NaN must sit right
    after the last sample before the gap, otherwise the level is drawn across it.
    """
    if len(series) < 2:
        return series
    gap_starts = series.index[:-1][np.diff(series.index.values) > max_gap.to_timedelta64()]
    if len(gap_starts) == 0:
        return series
    holes = pd.Series(np.nan, index=gap_starts + pd.Timedelta(milliseconds=1))
    return pd.concat([series, holes]).sort_index()


def telemetry_window(data: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last epoch over all satellites."""
    return (min(df.index[0] for df in data.values()),
            max(df.index[-1] for df in data.values()))


def setup_style() -> None:
    """Common matplotlib look: light surface, recessive grid, neutral text."""
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": MUTED_COLOR, "axes.labelcolor": TEXT_COLOR,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.6,
        "xtick.color": MUTED_COLOR, "ytick.color": MUTED_COLOR,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "text.color": TEXT_COLOR, "font.size": 9,
        "lines.linewidth": 0.6,
    })


def format_time_axis(ax: plt.Axes) -> None:
    """UTC date axis, readable over a 3-day window."""
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))


def save(fig: plt.Figure, filename: str) -> Path:
    """Save a figure into plots/ and close it."""
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
