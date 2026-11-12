"""
Question 2: for each satellite, share of the available telemetry points where
  a. one or more wheels are saturated: speed >= 8000 rpm
  b. the low-voltage flag is active: flag == 1
  c. one or more gyro rates are above 2 deg/s in absolute value (-3 deg/s counts too)

"Available points" are counted per criterion, over the epochs where its own channels
were sampled: wheel samples for (a), flag samples for (b), gyro samples for (c).
The CSV rows are asynchronous, so dividing by the total number of rows would also
count epochs where the channel was not measured at all.

Sampling is not uniform. On echo and golf the flag is sampled about every second
while it is 0, but only every ~90 s while it is 1, so the share of points understates
the time spent in low voltage. The share of time is given as well: each flag sample
holds until the next one, and silences longer than MAX_GAP (outages) are left out.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from telemetry import (GYRO_RATE_LIMIT_DEG_S, GYROS, MAX_GAP, WHEEL_SATURATION_RPM, WHEELS,
                       channel, load_all)


def share_of_points(mask: pd.Series) -> float:
    """Percentage of True values, NaN when the channel is not telemetered."""
    return 100 * mask.mean() if len(mask) else np.nan


def share_of_time(mask: pd.Series, max_gap: pd.Timedelta = MAX_GAP) -> float:
    """Percentage of time where mask is True, each sample holding until the next one.

    Intervals longer than max_gap are outages: the state is unknown, so they are left out.
    """
    if len(mask) < 2:
        return np.nan
    #print(mask)
    held = np.diff(mask.index.values)
    known = held <= max_gap.to_timedelta64()
    held = held[known].astype("int64")
    return 100 * (held * mask.to_numpy()[:-1][known]).sum() / held.sum()


def percentages(df: pd.DataFrame) -> dict:
    """The three criteria of question 2 for one satellite, with their number of points."""
    wheels = df[WHEELS].dropna(how="all") #All rows containing three NaN values for the wheel are excluded
    gyros = df[GYROS].dropna(how="all")
    flag = channel(df, "low_voltage") #select "low_voltage" and drop all the missing value
    low = flag == 1 #table with 3 columns (timestamp, low = flag ==1 (boolean))
    return {
        "a. wheel saturated [% pts]": share_of_points((wheels >= WHEEL_SATURATION_RPM).any(axis=1)),
        "b. low voltage [% pts]": share_of_points(low),
        "b. low voltage [% time]": share_of_time(low),
        "c. gyro > 2 deg/s [% pts]": share_of_points((gyros.abs() > GYRO_RATE_LIMIT_DEG_S).any(axis=1)),
        "wheel pts": len(wheels),
        "flag pts": len(flag),
        "gyro pts": len(gyros),
    }


def run(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute the table for every satellite, print it and return it."""
    table = pd.DataFrame({name: percentages(df) for name, df in data.items()}).T
    table.index.name = "satellite"
    counts = [c for c in table.columns if c.endswith(" pts")]
    shown = table.copy()
    for col in table.columns:
        fmt = "{:,.0f}" if col in counts else "{:.2f}"
        shown[col] = [fmt.format(v) if pd.notna(v) else "n/a" for v in table[col]]
    print(shown.to_string())
    return table


if __name__ == "__main__":
    t0 = time.perf_counter()
    run(load_all())
    print(f"q2 done in {time.perf_counter() - t0:.1f} s")
