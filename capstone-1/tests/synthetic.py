"""Synthetic stand-in for SeoulBikeData.csv, used only by the tests.

It reproduces the *format* of the real file (header spelling, latin-1 encoding with the degree
sign as byte 0xB0, dd/mm/yyyy dates, category spellings, hours with the system switched off) so
that the loading, cleaning, training and serving code can be exercised without a download.
The numbers are made up: never report results obtained on this data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_HEADER = [
    "Date",
    "Rented Bike Count",
    "Hour",
    "Temperature(\u00b0C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(\u00b0C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons",
    "Holiday",
    "Functioning Day",
]

_HOLIDAYS = {
    "2017-12-25", "2018-01-01", "2018-02-15", "2018-02-16", "2018-03-01", "2018-05-07",
    "2018-05-22", "2018-06-06", "2018-08-15", "2018-09-24", "2018-09-25", "2018-10-03", "2018-10-09",
}
_SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                    6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def make_raw_frame(seed: int = 0, start: str = "2017-12-01", days: int = 365) -> pd.DataFrame:
    """Return a frame with the real file's header names and value formats (all columns raw)."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start, periods=days * 24, freq="h")
    n = len(stamps)
    hour = stamps.hour.to_numpy()
    doy = stamps.dayofyear.to_numpy()
    month = stamps.month.to_numpy()
    weekday = stamps.dayofweek.to_numpy()
    date_str = stamps.strftime("%Y-%m-%d")
    holiday = np.array([d in _HOLIDAYS for d in date_str])
    day_off = (weekday >= 5) | holiday

    seasonal = 13.0 - 15.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    daily = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    temp = np.round(seasonal + daily + rng.normal(0, 2.0, n), 1)
    humidity = np.clip(np.round(60 + rng.normal(0, 18, n) - 0.4 * daily * 3), 12, 98).astype(int)
    humidity[rng.choice(n, 6, replace=False)] = 0        # a few physically implausible readings
    wind = np.clip(np.round(rng.gamma(2.0, 0.85, n), 1), 0, 7.4)
    visibility = np.clip(np.round(rng.normal(1500, 600, n), -1), 27, 2000).astype(int)
    dew = np.round(temp - (100 - np.where(humidity == 0, 60, humidity)) / 5.0, 1)
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0, None)
    solar = np.round(sun * (1.6 + 0.9 * np.sin(2 * np.pi * (doy - 80) / 365.0)) * rng.uniform(0.5, 1.0, n), 2)
    wet = rng.random(n) < 0.05
    rain = np.where(wet, np.round(rng.gamma(1.5, 3.0, n), 1), 0.0)
    snow = np.where((month <= 2) & (rng.random(n) < 0.03), np.round(rng.gamma(1.2, 1.0, n), 1), 0.0)

    commute = np.exp(-((hour - 8) ** 2) / 4.0) + 1.3 * np.exp(-((hour - 18) ** 2) / 5.0)
    leisure = np.exp(-((hour - 15) ** 2) / 18.0)
    profile = np.where(day_off, 0.9 * leisure, 0.25 + commute)
    warmth = np.clip((temp + 8) / 30.0, 0.05, 1.3)
    demand = 1100 * profile * warmth * np.where(wet, 0.35, 1.0) * np.where(snow > 0, 0.4, 1.0)
    count = np.clip(np.round(demand * rng.lognormal(0, 0.12, n)), 0, None).astype(int)

    functioning = np.ones(n, dtype=bool)
    n_closed_days = min(12, days // 8)          # 12 closed days + 7 closed hours = 295 hours for a full year
    closed_days = rng.choice(days, n_closed_days, replace=False)
    day_index = np.repeat(np.arange(days), 24)
    functioning[np.isin(day_index, closed_days)] = False
    functioning[rng.choice(n, 7, replace=False)] = False
    count[~functioning] = 0

    frame = pd.DataFrame(
        {
            "Date": stamps.strftime("%d/%m/%Y"),
            "Rented Bike Count": count,
            "Hour": hour,
            "Temperature(\u00b0C)": temp,
            "Humidity(%)": humidity,
            "Wind speed (m/s)": wind,
            "Visibility (10m)": visibility,
            "Dew point temperature(\u00b0C)": dew,
            "Solar Radiation (MJ/m2)": solar,
            "Rainfall(mm)": rain,
            "Snowfall (cm)": snow,
            "Seasons": [_SEASON_BY_MONTH[m] for m in month],
            "Holiday": np.where(holiday, "Holiday", "No Holiday"),
            "Functioning Day": np.where(functioning, "Yes", "No"),
        }
    )
    return frame[RAW_HEADER]


def write_raw_csv(path, **kwargs) -> None:
    """Write the synthetic frame the way the real file is stored (latin-1, comma separated)."""
    make_raw_frame(**kwargs).to_csv(path, index=False, encoding="latin-1")
