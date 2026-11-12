"""
Question 4: one detection function per anomaly found in question 3.

Each function takes ONE telemetry channel, a pandas Series indexed by UTC time such as
`telemetry.channel(df, "wheel_y")` (NaN epochs are allowed and ignored), and returns
True when the issue is present, False otherwise. A channel with no data returns False:
missing telemetry is not evidence of the issue.

No false positives: each threshold has a physical meaning and sits several times above
the worst case seen on the satellites without the issue over the 72 h window.

  function                       channel      issue (satellites)            worst case without   threshold   with the issue
  is_wheel_stopped               wheel_*      dead wheel (delta)            0 rpm for 8 min      1 h         9 to 72 h
  is_wheel_saturated_too_long    wheel_*      no momentum margin (delta)    2.2 min              10 min      35 min
  is_tumbling                    gyro_*       attitude lost (delta)         1.2 min              30 min      4 h
  is_battery_not_recharging      bus_voltage  power-negative (echo, bravo)  3 orbits (golf)      6 orbits    11 to 24
  is_low_voltage_latched         low_voltage  power-negative (echo, bravo)  42 min (golf)        95 min      4 to 7 h
  is_telemetry_lost              any          silent satellite (echo)       1 h (bravo)          6 h         18 h
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from telemetry import (CHANNELS, GYRO_RATE_LIMIT_DEG_S, GYROS, MAX_GAP, WHEEL_SATURATION_RPM,
                       WHEELS, channel, load_all)

ORBIT = pd.Timedelta("95min")
WHEEL_STOPPED_AFTER = pd.Timedelta("1h")
WHEEL_SATURATED_AFTER = pd.Timedelta("10min")
TUMBLING_AFTER = pd.Timedelta("30min")
RATE_SMOOTHING = "5min"
FULL_CHARGE_MIN_V = 8.1        # a healthy battery peaks at ~8.3 V every orbit
NOT_RECHARGING_ORBITS = 6      # ~9.5 h
TELEMETRY_LOST_AFTER = pd.Timedelta("6h")


def longest_run(mask: pd.Series) -> pd.Timedelta:
    """Longest time span covered by consecutive True samples.

    A silence longer than MAX_GAP ends a run, since the state during an outage is unknown.
    """
    if not mask.any():
        return pd.Timedelta(0)
    times = mask.index.to_series()
    new_run = (mask != mask.shift()) | (times.diff() > MAX_GAP)
    run_id = new_run.cumsum()
    bounds = times[mask].groupby(run_id[mask]).agg(["first", "last"])
    return (bounds["last"] - bounds["first"]).max()


def is_wheel_stopped(wheel_speed: pd.Series) -> bool:
    """Dead wheel: it reads 0 rpm without a break for an hour or more.

    Speeds are reported as magnitudes, so a working wheel only touches 0 rpm while it
    reverses, for a few minutes at most.
    """
    speed = wheel_speed.dropna()
    return longest_run(speed == 0) >= WHEEL_STOPPED_AFTER


def is_wheel_saturated_too_long(wheel_speed: pd.Series) -> bool:
    """Wheel held at saturation (>= 8000 rpm) for 10 min or more.

    A slew saturates a wheel for a couple of minutes. Staying there means the momentum
    is not dumped, so attitude control has no margin left on that axis.
    """
    speed = wheel_speed.dropna()
    return longest_run(speed >= WHEEL_SATURATION_RPM) >= WHEEL_SATURATED_AFTER


def is_tumbling(gyro_rate: pd.Series) -> bool:
    """Body rate above 2 deg/s on this axis, sustained for 30 min or more.

    A 5-min rolling median of |rate| ignores slews and spikes, which never keep it above
    2 deg/s for more than about a minute on a controlled satellite.
    """
    rate = gyro_rate.dropna().abs().rolling(RATE_SMOOTHING).median()
    return longest_run(rate > GYRO_RATE_LIMIT_DEG_S) >= TUMBLING_AFTER


def is_battery_not_recharging(bus_voltage: pd.Series) -> bool:
    """The battery has not come back near full charge for 6 orbits in a row.

    The peak voltage of each ~95 min window (one orbit) is compared with 8.1 V. A window
    without data breaks the streak.
    """
    volts = bus_voltage.dropna()
    if volts.empty:
        return False
    peaks = volts.resample(ORBIT, origin="start").max()
    streak = longest = 0
    for low in peaks < FULL_CHARGE_MIN_V:  # NaN (no data) compares as False
        streak = streak + 1 if low else 0
        longest = max(longest, streak)
    return longest >= NOT_RECHARGING_ORBITS


def is_low_voltage_latched(low_voltage: pd.Series) -> bool:
    """Low-voltage flag set without a break for more than one orbit.

    Even a full sunlit phase did not bring the bus back above the onboard threshold.
    """
    flag = low_voltage.dropna()
    return longest_run(flag == 1) >= ORBIT


def is_telemetry_lost(any_channel: pd.Series) -> bool:
    """No sample at all for 6 h or more in the middle of the data."""
    times = any_channel.dropna().index.values
    return len(times) > 1 and np.diff(times).max() >= TELEMETRY_LOST_AFTER.to_timedelta64()


# Each detector and the channels it applies to.
DETECTORS = {
    "wheel stopped": (is_wheel_stopped, WHEELS),
    "wheel saturated too long": (is_wheel_saturated_too_long, WHEELS),
    "tumbling": (is_tumbling, GYROS),
    "battery not recharging": (is_battery_not_recharging, ["bus_voltage"]),
    "low voltage latched": (is_low_voltage_latched, ["low_voltage"]),
    "telemetry lost": (is_telemetry_lost, CHANNELS),
}


def run(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply every detector to every satellite, print the result and the triggering channels."""
    table, details = {}, []
    for sat, df in data.items():
        table[sat] = {}
        for name, (detect, cols) in DETECTORS.items():
            hits = [c for c in cols if detect(channel(df, c))]
            table[sat][name] = bool(hits)
            if hits:
                details.append(f"  {sat}: {name} on {', '.join(hits)}")
    table = pd.DataFrame(table)
    table.index.name = "issue"
    print(table.replace({True: "TRUE", False: "-"}).to_string())
    print("\n".join(details))
    return table


if __name__ == "__main__":
    t0 = time.perf_counter()
    run(load_all())
    print(f"q4 done in {time.perf_counter() - t0:.1f} s")
