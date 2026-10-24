#!/usr/bin/env python3
"""
checker.py - data-quality checker for a weather-station data pack.

Run:
    python checker.py <pack_folder>

Writes flags.csv into the folder you run it from (the current working directory),
one row per problem the checker believes is real. Every row follows this schema:

    station_id  e.g. S03
    channel     a column name from the data (e.g. temp_c), or "all", or "metadata"
    start_utc   when the problem starts, true UTC (YYYY-MM-DD HH:MM)
    end_utc     when the problem ends,   true UTC (YYYY-MM-DD HH:MM)
    issue       a few words
    action      exclude, correct or keep (keep = looks odd but is real)
    note        one sentence of evidence; if correct, say how

How it works (in the order the checks run)
------------------------------------------
1. Timestamps are repaired first, because every later check compares stations
   with each other at the same instant and a wrong clock makes a healthy station
   look broken:
     - rows that are exact copies of earlier rows (a replayed upload) are dropped;
     - the logger clock is compared with GPS time from the .nmea file, and any
       offset larger than a few minutes is corrected;
     - missing intervals (gaps) are reported.
2. Single-station checks: impossible values / spikes, stuck (flat-lined) sensors,
   humidity stuck at saturation.
3. Network checks, each station against the median of the OTHER stations
   (a median of seven is not fooled by one or two broken neighbours):
     - sudden lasting steps (pressure offset, wind-speed scale change, ...);
     - a daytime warm bias that grows with sunshine (radiation-shield problem);
     - a rain gauge that stays dry while the neighbours get rain, and the burst
       of water released when it is cleaned;
     - solar timing: the irradiance peak must sit at solar noon, otherwise the
       clock is wrong (a back-up for stations without usable GPS).
4. Metadata: install-sheet position / elevation against the GPS, and an
   explanation ("keep") for a station that is an outlier only because of altitude.
5. Each flag is matched with the volunteers' maintenance log, if there is a
   relevant entry, so the coordinator can see the cause next to the evidence.
6. The table is validated against the schema above before it is written.

Nothing about one particular pack is hard-coded (no station ids, dates or offsets),
so the same command works on an unseen pack in the same format. All thresholds are
in one block below so they are easy to audit.

Only pandas and numpy are needed.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------
# Constants and thresholds (all in one place so they can be audited / tuned)
# ----------------------------------------------------------------------------------

# The measurement columns we expect in every logger CSV, in file order.
CHANNELS = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh",
            "wind_dir_deg", "rain_mm", "ghi_wm2"]

# Output file (written into the folder the command is run from) and its schema.
OUTPUT_FILE = Path("flags.csv")
OUTPUT_COLUMNS = ["station_id", "channel", "start_utc", "end_utc",
                  "issue", "action", "note"]
ALLOWED_ACTIONS = {"exclude", "correct", "keep"}
ALLOWED_CHANNELS = set(CHANNELS) | {"all", "metadata"}
TIME_FORMAT = "%Y-%m-%d %H:%M"

# Logger clock vs GPS: offsets smaller than this are normal jitter (a few seconds).
CLOCK_TOLERANCE_MIN = 5

# Hard physical / instrument limits. Anything outside is impossible, not weather.
PHYSICAL_LIMITS = {
    "temp_c": (-45.0, 50.0),
    "rh_pct": (0.0, 102.0),        # capacitive RH sensors may overshoot 100 % by ~2 %
    "pressure_hpa": (500.0, 1085.0),
    "wind_kmh": (0.0, 250.0),
    "wind_dir_deg": (0.0, 360.0),
    "rain_mm": (0.0, 25.0),        # per 10-minute interval
    "ghi_wm2": (0.0, 1500.0),
}

# A single reading this far from BOTH of its neighbours (and from the local
# median) is a spike, not weather.
SPIKE_LIMIT = {"temp_c": 10.0, "rh_pct": 40.0, "pressure_hpa": 5.0}

# The same value repeated for this long means the sensor (or its channel) is stuck.
FLATLINE_HOURS = 3.0

# Humidity stuck at saturation: RH >= RH_SATURATION for at least
# RH_SATURATION_HOURS while the neighbours' median RH stays below
# RH_NEIGHBOUR_DRY, i.e. the air around it was not actually saturated.
RH_SATURATION = 99.0
RH_SATURATION_HOURS = 6.0
RH_NEIGHBOUR_DRY = 85.0
RH_NEIGHBOUR_FOG = 90.0            # neighbours this humid too -> real fog, "keep"
RH_FOG_MIN_HOURS = 2.0

# Step changes against the network.
#   kind "offset": station minus neighbours jumps by >= threshold (channel units)
#   kind "scale" : station / neighbours ratio jumps by a factor >= exp(threshold)
#   window_h     : how many hours either side are compared (median) to find a step
STEP_RULES = {
    "pressure_hpa": ("offset", 2.5, 24),     # ~1.5 hPa site differences are normal
    "temp_c": ("offset", 2.5, 48),
    "rh_pct": ("offset", 15.0, 48),
    "wind_kmh": ("scale", math.log(1.8), 48),
    "ghi_wm2": ("scale", math.log(1.5), 48),
}
# A fault (sensor swap, new instrument) changes the level from one reading to the
# next; weather differences between sites (a front arriving an hour later, cold
# air pooling) build up over hours. So the jump measured with 1-hour medians at
# the step must be at least this fraction of the 1-2 day jump ...
STEP_SHARPNESS = 0.7
STEP_SHORT_ROWS = 6
# ... and the shifted level must last at least the comparison window, otherwise
# it is a passing weather difference (single bad readings are caught as spikes).

# Radiation-shield warm bias: the slope of (station - neighbours) temperature
# against irradiance rises by at least this many degC per 1000 W/m2.
WARM_BIAS_SLOPE = 2.0
WARM_BIAS_TSTAT = 4.0

# Rain gauge checks.
RAIN_REF_WET_MM = 3.0              # neighbours' median daily rain for a "wet day"
RAIN_WET_FRACTION = 0.6            # fraction of neighbours with >= 1 mm that day
RAIN_MISSED_MM = 10.0              # a dry spell must miss at least this much rain
                                   # (less is normal shower-to-shower variation)
RAIN_BURST_MM = 4.0                # a 10-minute total this big ...
RAIN_BURST_REF_MM = 0.5            # ... while neighbours' median is below this
RAIN_DUMP_DAYS = 7                 # a burst this soon after a dry spell = gauge emptied

# Solar timing: irradiance-weighted centre of the day vs solar noon. Clouds on
# one half of a day can shift it by ~40 min, so we need a bigger, longer error.
SOLAR_TIMING_MIN = 60
SOLAR_TIMING_DAYS = 3

# Words that tie a maintenance-log entry to a flag's channel, so a note quotes
# the relevant visit and not just any visit on a nearby date.
LOG_KEYWORDS = {
    "temp_c": ["temp", "shield", "paint", "white", "sensor"],
    "rh_pct": ["humid", "rh", "fog", "sensor"],
    "pressure_hpa": ["pressure", "barom"],
    "wind_kmh": ["wind", "anemometer", "cups"],
    "wind_dir_deg": ["wind", "vane", "direction", "anemometer"],
    "rain_mm": ["rain", "gauge", "funnel", "bucket"],
    "ghi_wm2": ["sun", "solar", "pyranometer", "radiation", "snow"],
    "all": ["data", "website", "battery", "clock", "reboot", "sim", "time", "logger", "twice"],
}

# Metadata vs GPS.
POSITION_TOLERANCE_KM = 0.5
ELEVATION_TOLERANCE_M = 50

# A station that is warmer/colder than the network by at least this much, but
# by about the amount its altitude predicts, gets a "keep" explanation.
ELEVATION_KEEP_MIN_C = 3.0
LAPSE_RATE_C_PER_M = -0.0065       # standard atmosphere

# Network checks need at least this many neighbours reporting at the same time.
MIN_NEIGHBOURS = 3


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------

# Turn a time into text like 2025-03-21 17:20.
def fmt_time(ts):
    return pd.Timestamp(ts).strftime(TIME_FORMAT)


# Turn minutes into text like '6 h' or '1 h 35 min'.
def fmt_duration(minutes):
    minutes = int(round(abs(minutes)))
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours} h {mins} min"
    return f"{hours} h" if hours else f"{mins} min"


# Find each block of back-to-back True values; return (first, last) positions.
def true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    # Pad with False on both sides so every run has a rising and a falling edge.
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    yield from zip(starts, ends)


# Convert a GPS coordinate like 3953.6752,N into decimal degrees.
def nmea_coordinate_to_decimal(value, hemisphere):
    if not value or not hemisphere:
        return np.nan
    raw_value = float(value)
    degrees = int(raw_value // 100)
    minutes = raw_value - degrees * 100
    decimal_degrees = degrees + minutes / 60
    return -decimal_degrees if hemisphere in ("S", "W") else decimal_degrees


# Check a GPS sentence's checksum so corrupted lines can be ignored.
def nmea_checksum_ok(sentence):
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, _, given = sentence[1:].partition("*")
    calculated = 0
    for char in body:
        calculated ^= ord(char)
    try:
        return calculated == int(given.strip()[:2], 16)
    except ValueError:
        return False


# Distance in km between two lat/lon points.
def distance_in_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    haversine = (math.sin((lat2 - lat1) / 2) ** 2
                 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(haversine))


# How many minutes the sun runs ahead of or behind clock time on a given day.
def equation_of_time_min(day_of_year):
    b = 2 * np.pi * (np.asarray(day_of_year, dtype=float) - 81) / 364
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)


# How high the sun is (degrees) at each UTC time at a given place.
def solar_elevation_deg(times_utc, lat, lon):
    times_utc = pd.DatetimeIndex(times_utc)
    doy = times_utc.day_of_year.to_numpy()
    hours = (times_utc.hour + times_utc.minute / 60).to_numpy()
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    solar_time = hours + lon / 15 + equation_of_time_min(doy) / 60
    hour_angle = np.radians(15 * (solar_time - 12))
    lat_r = np.radians(lat)
    sin_elev = (np.sin(lat_r) * np.sin(declination)
                + np.cos(lat_r) * np.cos(declination) * np.cos(hour_angle))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


# Subtract each station's own median, so stations at different sites can be compared.
def anomalies(panel):
    return panel - panel.median()


# Median of all OTHER stations at each time (NaN if too few report).
def neighbour_median(panel, station, min_neighbours=MIN_NEIGHBOURS):
    others = panel.drop(columns=[station])
    needed = min(min_neighbours, max(1, others.shape[1]))
    reference = others.median(axis=1, skipna=True)
    reference[others.notna().sum(axis=1) < needed] = np.nan
    return reference


# ----------------------------------------------------------------------------------
# Flag collection and output rules
# ----------------------------------------------------------------------------------

# Holds all flag rows and stops the same problem being written twice.
class Flags:

    # Start with an empty list of flags.
    def __init__(self):
        self.rows = []

    # True if an existing exclude/correct flag already covers this time window.
    def covered(self, station, channel, start, end):
        for row in self.rows:
            if (row["station_id"] == station
                    and row["channel"] in (channel, "all")
                    and row["action"] in ("exclude", "correct")
                    and row["_start"] <= start and end <= row["_end"]):
                return True
        return False

    # Add one flag row (skipped if an earlier flag already covers it).
    def add(self, station, channel, start, end, issue, action, note):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if action != "keep" and self.covered(station, channel, start, end):
            return False
        self.rows.append({
            "station_id": station, "channel": channel,
            "start_utc": fmt_time(start), "end_utc": fmt_time(end),
            "issue": issue, "action": action, "note": note,
            "_start": start, "_end": end,   # kept as Timestamps for sorting/overlap
        })
        return True

    # Return the flags as a sorted table with exactly the 7 output columns.
    def to_frame(self):
        if not self.rows:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        frame = pd.DataFrame(self.rows).sort_values(["station_id", "_start", "channel"])
        return frame[OUTPUT_COLUMNS].reset_index(drop=True)


# Tidy a note into a single sentence ending with one full stop.
def one_sentence(text):
    text = re.sub(r"\s+", " ", text).strip().rstrip(".;, ")
    return text + "."


# Check every row obeys the flags.csv rules; stop with an error if not.
def validate_output(frame):
    assert list(frame.columns) == OUTPUT_COLUMNS, "wrong output columns"
    for i, row in frame.iterrows():
        where = f"flags.csv row {i + 2} ({row['station_id']} {row['channel']})"
        assert row["channel"] in ALLOWED_CHANNELS, f"{where}: bad channel"
        assert row["action"] in ALLOWED_ACTIONS, f"{where}: bad action"
        start = pd.to_datetime(row["start_utc"], format=TIME_FORMAT)
        end = pd.to_datetime(row["end_utc"], format=TIME_FORMAT)
        assert start <= end, f"{where}: start after end"
        assert row["issue"] and len(row["issue"].split()) <= 8, f"{where}: issue too long"
        note = row["note"]
        assert note.endswith("."), f"{where}: note must end with a full stop"
        # No ". X" sentence break inside the note (decimals like 6.5 are fine).
        assert not re.search(r"[.!?]\s+[A-Z]", note[:-1]), f"{where}: note is more than one sentence"


# ----------------------------------------------------------------------------------
# Reading the pack
# ----------------------------------------------------------------------------------

# Find stations.csv, logger CSVs, GPS files and the maintenance log anywhere in the pack folder.
def find_pack_files(pack_dir):
    station_tables = sorted(pack_dir.rglob("stations.csv"))
    if not station_tables:
        sys.exit(f"error: no stations.csv found under {pack_dir}")
    stations = pd.read_csv(station_tables[0], dtype={"station_id": str})
    stations["station_id"] = stations["station_id"].str.strip()
    stations = stations.set_index("station_id")

    data_files = {}
    for path in sorted(pack_dir.rglob("*.csv")):
        if path.name == "stations.csv":
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns.str.strip()
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
        if "timestamp_utc" in header:
            data_files[path.stem] = path

    gps_files = {path.stem: path for path in sorted(pack_dir.rglob("*.nmea"))}
    log_files = sorted(pack_dir.rglob("maintenance_log*"))
    return stations, data_files, gps_files, (log_files[0] if log_files else None)


# Read one logger CSV; bad numbers become blanks and missing columns are added.
def load_logger_csv(path):
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()
    text = raw["timestamp_utc"].str.strip()
    timestamps = pd.to_datetime(text, format=TIME_FORMAT, errors="coerce")
    # Fall back to a lenient parser for any other format (e.g. with seconds).
    if timestamps.isna().any():
        timestamps = timestamps.fillna(pd.to_datetime(text, errors="coerce"))
    frame = pd.DataFrame({"timestamp": timestamps})
    for channel in CHANNELS:
        frame[channel] = (pd.to_numeric(raw[channel], errors="coerce")
                          if channel in raw.columns else np.nan)
    return frame


# Read a GPS file and keep only good sentences with a valid fix (time, position, altitude).
def parse_nmea(path):
    records = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if "," not in line:
                continue
            stamp, sentence = line.split(",", 1)
            if not nmea_checksum_ok(sentence):
                continue
            fields = sentence[1:].split("*")[0].split(",")
            kind = fields[0][-3:]
            record = {"logger_ts": stamp.strip(), "kind": kind, "gps_ts": None,
                      "lat": np.nan, "lon": np.nan, "alt": np.nan}
            try:
                if kind == "RMC" and len(fields) >= 10 and fields[2] == "A":
                    record["gps_ts"] = fields[9] + fields[1][:6]          # ddmmyyHHMMSS
                    record["lat"] = nmea_coordinate_to_decimal(fields[3], fields[4])
                    record["lon"] = nmea_coordinate_to_decimal(fields[5], fields[6])
                elif kind == "GGA" and len(fields) >= 10 and fields[6] not in ("", "0"):
                    record["lat"] = nmea_coordinate_to_decimal(fields[2], fields[3])
                    record["lon"] = nmea_coordinate_to_decimal(fields[4], fields[5])
                    record["alt"] = float(fields[9]) if fields[9] else np.nan
                else:
                    continue   # 'V' = no fix, or a sentence type we don't use
            except ValueError:
                continue
            records.append(record)

    gps = pd.DataFrame(records, columns=["logger_ts", "kind", "gps_ts", "lat", "lon", "alt"])
    # Convert all timestamps in one go (much faster than per line).
    gps["logger_ts"] = pd.to_datetime(gps["logger_ts"], errors="coerce")
    gps["gps_ts"] = pd.to_datetime(gps["gps_ts"], format="%d%m%y%H%M%S", errors="coerce")
    return gps[gps["logger_ts"].notna()].reset_index(drop=True)


# Read the volunteers' notes as (date, station, text).
def parse_maintenance_log(path):
    if path is None:
        return []
    entries = []
    pattern = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+(\S+)\s+(.*\S)")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            entries.append((pd.Timestamp(match.group(1)), match.group(2), match.group(3)))
    return entries


# ----------------------------------------------------------------------------------
# Stage 1: timestamps (replays, logger clock, gaps)
# ----------------------------------------------------------------------------------

# For each logger row, how far the logger clock is from GPS time (minutes).
def gps_clock_offsets(frame, gps):
    offsets = pd.Series(np.nan, index=frame.index)
    rmc = gps[(gps["kind"] == "RMC") & gps["gps_ts"].notna()].copy()
    if rmc.empty:
        return offsets

    rmc["offset_s"] = (rmc["logger_ts"] - rmc["gps_ts"]).dt.total_seconds()
    rmc["segment"] = (rmc["logger_ts"].diff().dt.total_seconds() < -30).cumsum()
    # A running median of 5 sentences ignores the odd glitched GPS time.
    rmc["offset_s"] = rmc.groupby("segment")["offset_s"].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).median())

    data_segment = (frame["timestamp"].diff().dt.total_seconds() <= 0).cumsum()
    paired = data_segment.max() == rmc["segment"].max()
    segment_ids = range(int(data_segment.max()) + 1) if paired else [None]

    for seg in segment_ids:
        rows = frame if seg is None else frame[data_segment == seg]
        sentences = rmc if seg is None else rmc[rmc["segment"] == seg]
        rows = rows[rows["timestamp"].notna()]
        if rows.empty or sentences.empty:
            continue
        matched = pd.merge_asof(
            rows[["timestamp"]].reset_index().sort_values("timestamp"),
            sentences[["logger_ts", "offset_s"]].sort_values("logger_ts"),
            left_on="timestamp", right_on="logger_ts",
            direction="nearest", tolerance=pd.Timedelta(hours=3))
        offsets.loc[matched["index"].to_numpy()] = (matched["offset_s"] / 60).to_numpy()
    return offsets


# How far the daily sunlight peak is from solar noon (minutes); used when there is no GPS.
def solar_noon_error_min(timestamps, ghi, lon):
    frame = pd.DataFrame({"t": pd.DatetimeIndex(timestamps), "g": np.asarray(ghi, dtype=float)})
    frame = frame[frame["g"] > 0].dropna()
    if frame.empty:
        return np.nan
    local_solar = frame["t"] + pd.to_timedelta(lon / 15, unit="h")
    day = local_solar.dt.floor("D")
    hour = (local_solar - day) / pd.Timedelta(hours=1)
    weighted = pd.DataFrame({"g": frame["g"].to_numpy(), "gh": (frame["g"] * hour).to_numpy(),
                             "day": day.to_numpy()}).groupby("day").agg(
        g=("g", "sum"), gh=("gh", "sum"), n=("g", "size"))
    weighted = weighted[weighted["n"] >= 0.8 * weighted["n"].max()]     # whole days only
    if len(weighted) < 2:
        return np.nan
    centre = weighted["gh"] / weighted["g"]
    expected = 12 - equation_of_time_min(pd.DatetimeIndex(weighted.index).day_of_year) / 60
    return float(((centre - expected) * 60).median())


# Without GPS: find whole-hour clock errors by checking where the sun peaks.
def solar_clock_offsets(frame, lon):
    offsets = pd.Series(np.nan, index=frame.index)
    step = frame["timestamp"].diff()
    piece = (step != step.mode().iloc[0]).cumsum()
    for _, rows in frame.groupby(piece):
        error = solar_noon_error_min(rows["timestamp"], rows["ghi_wm2"], lon)
        if (np.isfinite(error) and abs(error) > SOLAR_TIMING_MIN
                and abs(error / 60 - round(error / 60)) < 0.25):
            offsets[rows.index] = round(error / 60) * 60
    return offsets


# Repair the timestamps: drop unreadable rows and repeated uploads, and fix the clock using GPS or the sun.
def fix_timestamps(station, raw, gps, lon, interval_min, flags):
    frame = raw.copy()

    # a) Unreadable timestamps: nothing can be done with these rows. One flag per
    # run of bad rows, dated by the readable rows on either side of it.
    unreadable = frame["timestamp"].isna().to_numpy()
    step = pd.Timedelta(minutes=interval_min)
    for first, last in true_runs(unreadable):
        before = frame["timestamp"].iloc[:first].dropna()
        after = frame["timestamp"].iloc[last + 1:].dropna()
        start = before.iloc[-1] + step if len(before) else after.iloc[0] - step * (last - first + 1)
        end = after.iloc[0] - step if len(after) else start + step * (last - first)
        flags.add(station, "all", start, max(start, end), "unreadable timestamp", "exclude",
                  f"{last - first + 1} row(s) at file line {first + 2} have a timestamp that "
                  f"cannot be parsed, so their readings were dropped.")
    frame = frame[~unreadable]

    # b) Exact replays: same timestamp AND same values as an earlier row.
    # This is what a re-sent upload looks like (the website "showing the same day twice").
    replay = frame.duplicated(subset=["timestamp"] + CHANNELS, keep="first").to_numpy()
    for first, last in true_runs(replay):
        block = frame.iloc[first:last + 1]
        flags.add(station, "all", block["timestamp"].min(), block["timestamp"].max(),
                  "records sent twice", "correct",
                  f"{len(block)} rows (file rows {first + 2}-{last + 2}) repeat these timestamps "
                  f"value-for-value; drop the second copy and keep the first.")
    frame = frame[~replay].reset_index(drop=True)

    # c) Logger clock vs GPS (preferred), else vs the sun.
    has_gps = gps is not None and (gps["kind"] == "RMC").any()
    if has_gps:
        offsets_min = gps_clock_offsets(frame, gps)
        evidence = "GPS time in the .nmea file shows"
    elif np.isfinite(lon):
        offsets_min = solar_clock_offsets(frame, lon)
        evidence = "With no usable GPS, the daily sunlight peak (which must fall at solar noon) shows"
    else:
        offsets_min = pd.Series(np.nan, index=frame.index)
        evidence = ""
    offsets_min = offsets_min.round().fillna(0)
    offsets_min[offsets_min.abs() < CLOCK_TOLERANCE_MIN] = 0
    jumped_back = (frame["timestamp"].diff().dt.total_seconds() <= 0).to_numpy()

    corrected = (frame["timestamp"]
                 - pd.to_timedelta(offsets_min, unit="min")).dt.round(f"{interval_min}min")

    # One flag per run of rows sharing the same non-zero clock offset.
    run_id = (offsets_min != offsets_min.shift()).cumsum()
    nonzero = offsets_min != 0
    for _, run in offsets_min[nonzero].groupby(run_id[nonzero]):
        offset = run.iloc[0]
        logger_stamps = frame.loc[run.index, "timestamp"]
        direction = "behind" if offset < 0 else "ahead of"
        fix = "add" if offset < 0 else "subtract"
        reset = " after the clock jumped backwards" if jumped_back[run.index[0]] else ""
        flags.add(station, "all", corrected[run.index].min(), corrected[run.index].max(),
                  f"logger clock {fmt_duration(offset)} {direction.split()[0]}", "correct",
                  f"{evidence} the logger clock {fmt_duration(offset)} "
                  f"{direction} UTC{reset} (logger stamps {fmt_time(logger_stamps.iloc[0])} "
                  f"to {fmt_time(logger_stamps.iloc[-1])}); {fix} {fmt_duration(offset)} "
                  f"to these timestamps.")

    frame["timestamp"] = corrected

    # d) Collisions left over (e.g. a backwards jump with no GPS to fix it):
    # we cannot tell which copy is right, so keep the first and flag the window.
    later_copy = frame["timestamp"].duplicated(keep="first").to_numpy()
    for first, last in true_runs(later_copy):
        block = frame.iloc[first:last + 1]
        flags.add(station, "all", block["timestamp"].min(), block["timestamp"].max(),
                  "conflicting duplicate timestamps", "exclude",
                  f"{len(block)} rows reuse timestamps already in the file with different values "
                  f"(likely a clock reset with no GPS to correct it); only the first copy was kept.")
    frame = frame.drop_duplicates("timestamp", keep="first")

    return frame.set_index("timestamp").sort_index()


# Flag every stretch of missing records.
def check_gaps(station, clean, grid, interval_min, flags):
    missing = ~grid.isin(clean.index)
    for first, last in true_runs(missing):
        count = last - first + 1
        flags.add(station, "all", grid[first], grid[last], "no data (gap)", "exclude",
                  f"{count} consecutive {interval_min}-minute records are missing "
                  f"({fmt_duration(count * interval_min)}), so totals such as rain are "
                  f"incomplete for this window.")


# ----------------------------------------------------------------------------------
# Stage 2: single-station checks
# ----------------------------------------------------------------------------------

# Flag humidity stuck at 100 % while the other stations are dry.
def check_rh_saturation(station, panels, masks, interval_min, flags):
    rh = panels["rh_pct"][station]
    reference = neighbour_median(panels["rh_pct"], station)
    saturated = (rh >= RH_SATURATION).to_numpy()
    runs = list(true_runs(saturated))
    had_fault = any((last - first + 1) * interval_min / 60 >= RH_SATURATION_HOURS
                    and reference.iloc[first:last + 1].median() < RH_NEIGHBOUR_DRY
                    for first, last in runs)
    for first, last in runs:
        window = slice(first, last + 1)
        hours = (last - first + 1) * interval_min / 60
        ref_values = reference.iloc[window]
        ref_median = float(ref_values.median()) if ref_values.notna().any() else np.nan
        start, end = rh.index[first], rh.index[last]
        if hours >= RH_SATURATION_HOURS and ref_median < RH_NEIGHBOUR_DRY:
            peak = rh.iloc[window].max()
            flags.add(station, "rh_pct", start, end, "humidity stuck at saturation", "exclude",
                      f"RH stayed at {RH_SATURATION:.0f}-{peak:.0f} % for {hours:.0f} h while the "
                      f"other stations' median was {ref_median:.0f} %, so the sensor, not the air, "
                      f"was saturated.")
            masks["rh_pct"].iloc[window, masks["rh_pct"].columns.get_loc(station)] = True
        elif had_fault and hours >= RH_FOG_MIN_HOURS and ref_median >= RH_NEIGHBOUR_FOG:
            flags.add(station, "rh_pct", start, end, "saturated humidity, real fog", "keep",
                      f"RH at 100 % for {hours:.1f} h, but this time the other stations' median "
                      f"was {ref_median:.0f} % as well, so it is real fog or rain and not the "
                      f"earlier sensor fault.")


# Flag impossible values and single-reading spikes.
def check_impossible_and_spikes(station, panels, masks, flags):
    for channel in CHANNELS:
        series = panels[channel][station]
        low, high = PHYSICAL_LIMITS[channel]
        impossible = (series < low) | (series > high)

        spike = pd.Series(False, index=series.index)
        if channel in SPIKE_LIMIT:
            limit = SPIKE_LIMIT[channel]
            local_median = series.rolling(7, center=True, min_periods=3).median()
            spike = (((series - local_median).abs() > limit)
                     & ((series - series.shift(1)).abs() > limit)
                     & ((series - series.shift(-1)).abs() > limit))

        bad = (impossible | spike).fillna(False).to_numpy()
        column = masks[channel].columns.get_loc(station)
        for first, last in true_runs(bad):
            values = series.iloc[first:last + 1]
            start, end = values.index[0], values.index[-1]
            if flags.covered(station, channel, start, end):
                masks[channel].iloc[first:last + 1, column] = True
                continue
            context = series.iloc[max(0, first - 3):last + 4].drop(values.index).median()
            shown = ", ".join(f"{v:g}" for v in values.unique()[:3])
            kind = "impossible value" if impossible.iloc[first:last + 1].any() else "spike"
            hint = (" (a typical sensor error code)"
                    if channel == "temp_c" and set(values.unique()) <= {-40.0, 85.0} else "")
            flags.add(station, channel, start, end, f"{kind} {shown}", "exclude",
                      f"{channel} read {shown}{hint} against about {context:.1f} in the "
                      f"surrounding half hour.")
            masks[channel].iloc[first:last + 1, column] = True


# Flag a sensor that repeats exactly the same value for hours.
def check_flatlines(station, panels, masks, positions, interval_min, flags):
    min_rows = int(FLATLINE_HOURS * 60 / interval_min)
    lat, lon = positions[station]
    elevation = solar_elevation_deg(panels["ghi_wm2"].index, lat, lon)
    wind = panels["wind_kmh"][station]

    for channel in ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "wind_dir_deg", "ghi_wm2"]:
        series = panels[channel][station].where(~masks[channel][station])
        if channel == "ghi_wm2":
            series[elevation < 10] = np.nan            # only look at daylight
        if channel == "rh_pct":
            series[series >= RH_SATURATION] = np.nan
        if channel == "wind_kmh":
            series[series == 0] = np.nan
        if channel == "wind_dir_deg":
            series[wind.rolling(6, min_periods=1).max() < 2] = np.nan

        # Label runs of identical consecutive values (NaN breaks a run).
        run_id = (series != series.shift()).cumsum()
        run_len = series.groupby(run_id).transform("size")
        stuck = (series.notna() & (run_len >= min_rows)).to_numpy()
        column = masks[channel].columns.get_loc(station)
        for first, last in true_runs(stuck):
            value = series.iloc[first]
            hours = (last - first + 1) * interval_min / 60
            flags.add(station, channel, series.index[first], series.index[last],
                      f"sensor stuck at {value:g}", "exclude",
                      f"{channel} repeated exactly {value:g} for {hours:.0f} h "
                      f"({last - first + 1} readings) while the other stations kept changing.")
            masks[channel].iloc[first:last + 1, column] = True


# ----------------------------------------------------------------------------------
# Stage 3: network checks
# ----------------------------------------------------------------------------------

# Find sudden, lasting jumps in a station-minus-neighbours series.
def find_steps(residual, window, threshold, max_steps=3):
    values = np.asarray(residual, dtype=float).copy()
    n = len(values)
    short = STEP_SHORT_ROWS                     # 1 hour of 10-minute samples
    steps, ignored = [], []
    for _ in range(10):                         # a few attempts, max_steps accepted
        if len(steps) >= max_steps:
            break
        series = pd.Series(values)
        before = series.rolling(window, min_periods=window // 2).median().shift(1)
        after = series[::-1].rolling(window, min_periods=window // 2).median()[::-1]
        jump = (after - before).to_numpy().copy()   # writable copy
        for pos in steps + ignored:             # don't look at the same place twice
            jump[max(0, pos - window):pos + window] = np.nan
        if np.all(np.isnan(jump)):
            break
        pos = int(np.nanargmax(np.abs(jump)))
        size = jump[pos]
        if abs(size) < threshold:
            break

        # Refine to single-sample resolution with 1-hour medians near the candidate.
        mean_before = series.rolling(short, min_periods=short // 2).median().shift(1)
        mean_after = series[::-1].rolling(short, min_periods=short // 2).median()[::-1]
        local_jump = (np.sign(size) * (mean_after - mean_before)).to_numpy()
        lo, hi = max(short, pos - window // 2), min(n - short, pos + window // 2)
        if hi <= lo or not np.any(np.isfinite(local_jump[lo:hi])):
            ignored.append(pos)
            continue
        refined = lo + int(np.nanargmax(local_jump[lo:hi]))

        if local_jump[refined] < STEP_SHARPNESS * abs(size):
            ignored.append(pos)                 # gradual change = weather, not a fault
            continue

        # The 1-hour medians leave a plateau of equally good positions; snap to the
        # biggest reading-to-reading change (in the step's direction) within it.
        a, b = max(1, refined - short), min(n, refined + short + 1)
        single = np.sign(size) * (values[a:b] - values[a - 1:b - 1])
        if np.any(np.isfinite(single)):
            refined = a + int(np.nanargmax(single))
        steps.append(refined)
        values[refined:] -= size                # remove this step, look for others
    return sorted(steps)


# Split the series at the steps and give each piece its typical level.
def segment_levels(residual, steps, threshold, min_length):
    values = np.asarray(residual, dtype=float)

    # Median level and number of readings for one piece.
    def level_of(a, b):
        chunk = values[a:b + 1]
        count = int(np.isfinite(chunk).sum())
        return (float(np.nanmedian(chunk)) if count else np.nan), count

    bounds = [0] + list(steps) + [len(values)]
    segments = [(a, b - 1) for a, b in zip(bounds[:-1], bounds[1:])]
    merged = True
    while merged and len(segments) > 1:
        merged = False
        # 1) fold away segments that are too short to count as a level
        for i, (a, b) in enumerate(segments):
            if b - a + 1 >= min_length:
                continue
            level = level_of(a, b)[0]
            gaps = {}
            for j in (i - 1, i + 1):
                if 0 <= j < len(segments):
                    other = level_of(*segments[j])[0]
                    gaps[j] = abs(other - level) if np.isfinite(other - level) else 0.0
            j = min(gaps, key=gaps.get)
            lo, hi = sorted((i, j))
            segments[lo:hi + 1] = [(segments[lo][0], segments[hi][1])]
            merged = True
            break
        if merged:
            continue
        # 2) merge neighbours whose levels are within the threshold
        for i in range(len(segments) - 1):
            (a1, b1), (a2, b2) = segments[i], segments[i + 1]
            l1, _ = level_of(a1, b1)
            l2, _ = level_of(a2, b2)
            if not (abs(l1 - l2) >= threshold):     # also merges when a level is NaN
                segments[i:i + 2] = [(a1, b2)]
                merged = True
                break
    return [(a, b, *level_of(a, b)) for a, b in segments]


# Convert station pressure to sea level so stations at different heights compare.
def sea_level_pressure(panels, elevations):
    pressure, temperature = panels["pressure_hpa"], panels["temp_c"]
    reduced = pd.DataFrame(index=pressure.index)
    for station in pressure.columns:
        z = elevations.get(station, np.nan)
        t_kelvin = temperature[station].fillna(10) + 273.15 - LAPSE_RATE_C_PER_M * z / 2
        reduced[station] = pressure[station] * np.exp(9.80665 * z / (287.05 * t_kelvin))
    return reduced


# Flag sudden lasting shifts against the other stations (e.g. a new sensor) and say how to correct them.
def check_steps(station, panels, masks, elevations, interval_min, flags):
    rows_per_hour = 60 // interval_min
    for channel, (kind, threshold, window_h) in STEP_RULES.items():
        panel = panels[channel].where(~masks[channel])
        series = panel[station]
        if series.notna().sum() < 4 * window_h * rows_per_hour:
            continue

        if channel == "pressure_hpa":
            relative = panel / panel.median() - 1
            residual = (relative[station] - neighbour_median(relative, station)) * series.median()
        elif kind == "offset":
            anomaly = anomalies(panel)
            residual = anomaly[station] - neighbour_median(anomaly, station)
        else:
            normalised = panel / panel.mean()
            reference = neighbour_median(normalised, station) * series.mean()
            residual = np.log((series + 0.5) / (reference + 0.5))
            if channel == "ghi_wm2":
                residual[(reference < 50) | (series < 5)] = np.nan
            else:
                residual[reference < 2] = np.nan       # ratios in near-calm are noise

        window = window_h * rows_per_hour
        steps = find_steps(residual.to_numpy(), window, threshold)
        if not steps:
            continue
        segments = segment_levels(residual.to_numpy(), steps, threshold, window)
        if len(segments) < 2:
            continue

        # Pick the segment we trust.
        if channel == "pressure_hpa":
            slp = sea_level_pressure(panels, elevations).where(~masks["pressure_hpa"])
            absolute = (slp[station] - neighbour_median(slp, station)).to_numpy()

            # How far this piece's sea-level pressure is from the other stations.
            def distance_from_network(segment):
                part = absolute[segment[0]:segment[1] + 1]
                return abs(np.nanmedian(part)) if np.isfinite(part).any() else np.inf
            base = min(segments, key=distance_from_network)
        else:
            base = max(segments, key=lambda segment: segment[3])
        base_level = base[2]

        column = panels[channel].columns.get_loc(station)
        for first, last, level, count in segments:
            if not count or abs(level - base_level) < threshold:
                continue
            start, end = series.index[first], series.index[last]
            # Describe where the sudden step is: at the start of this segment,
            # or (for the first segment) at its end.
            when = (f"from the sudden step at {fmt_time(start)}" if first > 0
                    else f"until the sudden step at {fmt_time(series.index[last + 1])}")
            if kind == "offset":
                shift = level - base_level
                issue = f"{channel.split('_')[0]} offset {shift:+.1f}"
                fix = (f"subtract {shift:.1f} from" if shift > 0 else f"add {-shift:.1f} to")
                note = (f"{channel} reads {shift:+.1f} against the other stations {when}, "
                        f"while the rest of the record agrees with them; {fix} these readings.")
                panels[channel].iloc[first:last + 1, column] -= shift
            else:
                factor = math.exp(base_level - level)
                if channel == "wind_kmh" and abs(factor / 3.6 - 1) < 0.1:
                    # The measured change is within 10 % of 3.6 = km/h per m/s: a unit
                    # mix-up is far more likely than a coincidental calibration error,
                    # so correct with the exact unit factor.
                    measured, factor = factor, 3.6
                    how = (f"multiply these readings by 3.6, because the measured change "
                           f"({measured:.2f}) matches m/s being logged as km/h")
                else:
                    how = f"multiply these readings by {factor:.2f}"
                issue = f"{channel.split('_')[0]} scale change x{1 / factor:.2f}"
                note = (f"{channel} is {1 / factor:.2f}x its usual ratio to the other stations "
                        f"{when}; {how}.")
                panels[channel].iloc[first:last + 1, column] *= factor
            flags.add(station, channel, start, end, issue, "correct", note)


# Flag temperature that reads too warm only in sunshine (radiation shield problem).
def check_daytime_warm_bias(station, panels, masks, positions, flags):
    temp = panels["temp_c"].where(~masks["temp_c"])
    ghi = panels["ghi_wm2"].where(~masks["ghi_wm2"])
    anomaly = anomalies(temp)
    residual = anomaly[station] - neighbour_median(anomaly, station)
    lat, lon = positions[station]
    solar_day = (temp.index + pd.to_timedelta(lon / 15, unit="h")).floor("D")

    data = pd.DataFrame({"r": residual.to_numpy(), "g": ghi[station].to_numpy() / 1000,
                         "day": solar_day}).dropna()
    data = data[data["g"] > 0.1]
    slopes = {}
    for day, group in data.groupby("day"):
        if len(group) >= 12 and group["g"].std() > 0.05:
            slopes[day] = np.polyfit(group["g"], group["r"], 1)[0]
    slopes = pd.Series(slopes, dtype=float).sort_index()
    if len(slopes) < 12:
        return

    # Best single split: the day where the increase of the mean daily slope is
    # statistically clearest (largest t-statistic).
    best = None
    values = slopes.to_numpy()
    for k in range(5, len(values) - 4):
        before, after = values[:k], values[k:]
        diff = after.mean() - before.mean()
        se = math.sqrt(before.var(ddof=1) / len(before) + after.var(ddof=1) / len(after))
        tstat = diff / se if se > 0 else 0
        if best is None or tstat > best[2]:
            best = (k, diff, tstat)
    if best is None:
        return
    k, diff, tstat = best
    if diff < WARM_BIAS_SLOPE or tstat < WARM_BIAS_TSTAT:
        return

    start = slopes.index[k] - pd.to_timedelta(lon / 15, unit="h")   # back to UTC
    start = temp.index[temp.index >= start][0]
    end = temp[station].last_valid_index()
    period_before = residual[:start][ghi[station][:start] > 600]
    period_after = residual[start:end][ghi[station][start:end] > 600]
    flags.add(station, "temp_c", start, end, "daytime warm bias (shield)", "exclude",
              f"In sunshine (>600 W/m2) temperature reads a median {period_after.median():+.1f} degC "
              f"(up to {residual[start:end].max():+.1f}) against the other stations, versus "
              f"{period_before.median():+.1f} degC before, and the excess grows with sunlight "
              f"rather than time of day.")
    masks["temp_c"].loc[start:end, station] = True


# Flag a blocked rain gauge (dry while neighbours get rain) and rain bursts the neighbours don't see.
def check_rain(station, panels, masks, interval_min, flags):
    rain = panels["rain_mm"].where(~masks["rain_mm"])
    series = rain[station]
    others = rain.drop(columns=[station])
    if others.shape[1] < 2 or series.notna().sum() == 0:
        return
    step = pd.Timedelta(minutes=interval_min)

    reference_10min = neighbour_median(rain, station)
    burst = ((series >= RAIN_BURST_MM) & (reference_10min < RAIN_BURST_REF_MM)).fillna(False)

    daily_station = series.resample("D").sum(min_count=1)
    daily_others = others.resample("D").sum(min_count=1)
    daily_reference = daily_others.median(axis=1)
    wet_fraction = (daily_others >= 1.0).sum(axis=1) / daily_others.notna().sum(axis=1)
    suspicious = ((daily_station == 0) & (daily_reference >= RAIN_REF_WET_MM)
                  & (wet_fraction >= RAIN_WET_FRACTION))

    tips = series[series > 0].index
    handled = set()
    for day in suspicious[suspicious].index:
        if day in handled:
            continue
        previous = tips[tips < day]
        following = tips[tips >= day]
        start = previous[-1] + step if len(previous) else series.index[0]
        first_after = following[0] if len(following) else None
        stop = first_after if first_after is not None else series.index[-1] + step
        days = suspicious[(suspicious.index >= start.floor("D")) & (suspicious.index < stop)]
        days = days[days].index
        handled.update(days)
        missed = daily_reference.loc[days].sum()

        end = (first_after - step) if first_after is not None else series.index[-1]
        dump_text = ""
        dumped_after = (first_after is not None and burst.get(first_after, False)
                        and first_after - days[-1] <= pd.Timedelta(days=RAIN_DUMP_DAYS))
        if missed < RAIN_MISSED_MM and not dumped_after:
            continue                            # ordinary patchy showers
        if dumped_after:
            # Extend through the consecutive burst intervals (the dump itself).
            end = first_after
            while burst.get(end + step, False):
                end += step
            dumped = series[first_after:end].sum()
            minutes = int((end - first_after) / step + 1) * interval_min
            burst[first_after:end] = False
            dump_text = (f", then {dumped:.1f} mm in {fmt_duration(minutes)} from "
                         f"{fmt_time(first_after)} while the neighbours were dry (water released "
                         f"when the funnel was cleared)")
        flags.add(station, "rain_mm", start, end, "rain gauge blocked", "exclude",
                  f"Recorded 0 mm on {len(days)} day(s) ({days[0]:%d %b} to {days[-1]:%d %b}) when the "
                  f"other stations' median total was {missed:.1f} mm{dump_text}.")
        masks["rain_mm"].loc[start:end, station] = True

    column = masks["rain_mm"].columns.get_loc(station)
    for first, last in true_runs(burst.to_numpy()):
        chunk = series.iloc[first:last + 1]
        flags.add(station, "rain_mm", chunk.index[0], chunk.index[-1], "rain burst, neighbours dry",
                  "exclude", f"{chunk.sum():.1f} mm in {fmt_duration(len(chunk) * interval_min)} while "
                  f"the other stations' median was below {RAIN_BURST_REF_MM} mm.")
        masks["rain_mm"].iloc[first:last + 1, column] = True


# Back-up check: flag days when the sunlight peak is far from solar noon (clock wrong).
def check_solar_timing(station, panels, masks, positions, flags):
    ghi = panels["ghi_wm2"].where(~masks["ghi_wm2"])[station]
    lat, lon = positions[station]
    local_solar = ghi.index + pd.to_timedelta(lon / 15, unit="h")
    day = local_solar.floor("D")
    hour = (local_solar - day) / pd.Timedelta(hours=1)
    frame = pd.DataFrame({"g": ghi.to_numpy(), "h": hour, "day": day}).dropna()
    frame = frame[frame["g"] > 0]
    if frame.empty:
        return
    sums = frame.assign(gh=frame["g"] * frame["h"]).groupby("day")[["g", "gh"]].sum()
    counts = frame.groupby("day").size()
    centre = sums["gh"] / sums["g"]
    expected = 12 - equation_of_time_min(pd.DatetimeIndex(centre.index).day_of_year) / 60
    error_min = ((centre - expected) * 60)[counts >= counts.median() * 0.8]
    smoothed = error_min.rolling(3, center=True, min_periods=2).median()
    bad = (smoothed.abs() > SOLAR_TIMING_MIN).to_numpy()
    for first, last in true_runs(bad):
        if last - first + 1 < SOLAR_TIMING_DAYS:
            continue
        start = smoothed.index[first] - pd.to_timedelta(lon / 15, unit="h")
        end = smoothed.index[last] + pd.Timedelta(days=1) - pd.to_timedelta(lon / 15, unit="h")
        shift = float(smoothed.iloc[first:last + 1].median())
        whole_hours = abs(shift / 60 - round(shift / 60)) < 0.25
        action = "correct" if whole_hours else "exclude"
        how = (f"; {'subtract' if shift > 0 else 'add'} {fmt_duration(round(shift / 60) * 60)}"
               if whole_hours else "")
        flags.add(station, "all", max(start, ghi.index[0]), min(end, ghi.index[-1]),
                  "timestamps off solar time", action,
                  f"The daily sunlight peak is centred {fmt_duration(shift)} "
                  f"{'after' if shift > 0 else 'before'} solar noon on {last - first + 1} days, "
                  f"so the clock is wrong{how}.")


# ----------------------------------------------------------------------------------
# Stage 4: metadata
# ----------------------------------------------------------------------------------

# Compare stations.csv (position, height, interval, install date) with what the station reports.
def check_metadata(station, stations, gps, clean, flags):
    start, end = clean.index.min(), clean.index.max()
    if station not in stations.index:
        flags.add(station, "metadata", start, end, "station missing from stations.csv", "correct",
                  "There is a data file for this station but no row in stations.csv; add its "
                  "position, elevation and install date.")
        return
    info = stations.loc[station]

    if gps is not None and gps["lat"].notna().sum() >= 10:
        gps_lat, gps_lon = gps["lat"].median(), gps["lon"].median()
        km = distance_in_km(info["lat"], info["lon"], gps_lat, gps_lon)
        if km > POSITION_TOLERANCE_KM:
            flags.add(station, "metadata", start, end, "install-sheet position wrong", "correct",
                      f"{int(gps['lat'].notna().sum())} valid GPS fixes put the station at "
                      f"{gps_lat:.4f}, {gps_lon:.4f}, {km:.1f} km from the stations.csv position "
                      f"{info['lat']}, {info['lon']}; use the GPS position.")
        altitude = gps["alt"].median()
        if pd.notna(altitude) and abs(altitude - info["elevation_m"]) > ELEVATION_TOLERANCE_M:
            flags.add(station, "metadata", start, end, "install-sheet elevation wrong", "correct",
                      f"GPS altitude is {altitude:.0f} m but stations.csv says "
                      f"{info['elevation_m']:.0f} m; use {altitude:.0f} m.")

    spacing = clean.index.to_series().diff().dt.total_seconds().div(60).mode()
    sheet_interval = info.get("logger_interval_min", np.nan)
    if len(spacing) and pd.notna(sheet_interval) and spacing.iloc[0] != sheet_interval:
        flags.add(station, "metadata", start, end, "logging interval differs", "correct",
                  f"Records are {spacing.iloc[0]:.0f} minutes apart but stations.csv says "
                  f"{sheet_interval:.0f}; update the table.")

    installed = pd.to_datetime(info.get("installed", None), errors="coerce")
    if pd.notna(installed) and installed > start:
        flags.add(station, "metadata", start, end, "data before install date", "correct",
                  f"First record is {fmt_time(start)} but stations.csv says installed "
                  f"{installed:%Y-%m-%d}; correct whichever of the two is wrong.")


# Mark as 'keep' a station that is colder or warmer only because of its altitude.
def check_elevation_outlier(panels, masks, elevations, flags):
    temp = panels["temp_c"].where(~masks["temp_c"])
    means = temp.mean()
    for station in temp.columns:
        if station not in elevations or temp[station].notna().sum() == 0:
            continue
        others = [s for s in temp.columns if s != station and s in elevations]
        if len(others) < 2:
            continue
        actual = means[station] - means[others].median()
        expected = LAPSE_RATE_C_PER_M * (elevations[station]
                                         - np.median([elevations[s] for s in others]))
        if (abs(actual) >= ELEVATION_KEEP_MIN_C
                and abs(actual - expected) <= max(1.5, 0.3 * abs(expected))):
            flags.add(station, "temp_c", temp[station].first_valid_index(),
                      temp[station].last_valid_index(),
                      f"much {'colder' if actual < 0 else 'warmer'} than network", "keep",
                      f"Averages {actual:+.1f} degC against the other stations, close to the "
                      f"{expected:+.1f} degC the standard lapse rate predicts for its "
                      f"{elevations[station]:.0f} m elevation, so this is real.")


# ----------------------------------------------------------------------------------
# Stage 5: link flags to the maintenance log
# ----------------------------------------------------------------------------------

# Add a short quote from a relevant maintenance-log entry to each flag's note.
def attach_maintenance_notes(flags, maintenance):
    for row in flags.rows:
        # Metadata and single-reading glitches are not caused by a site visit.
        if row["channel"] == "metadata" or row["_start"] == row["_end"]:
            continue
        keywords = LOG_KEYWORDS.get(row["channel"], [])
        start_day, end_day = row["_start"].normalize(), row["_end"].normalize()
        candidates = [(abs((date - start_day).days), date, text)
                      for date, station, text in maintenance
                      if station == row["station_id"]
                      and start_day - pd.Timedelta(days=1) <= date <= end_day + pd.Timedelta(days=1)
                      and any(re.search(rf"\b{word}", text, re.IGNORECASE) for word in keywords)]
        if candidates:
            _, date, text = min(candidates)
            # Keep only the first sentence of the log entry, without its full stop,
            # so the note remains one sentence.
            quote = re.split(r"(?<=[.!?])\s+", text)[0].rstrip(".!?")
            if len(quote) > 70:
                quote = quote[:67].rsplit(" ", 1)[0] + "..."
            row["note"] = row["note"].rstrip(".") + f"; maintenance log {date:%d %b}: \"{quote}\"."


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

# Run every check in order and write flags.csv.
def main():
    parser = argparse.ArgumentParser(description="Check a weather-station data pack and write flags.csv.")
    parser.add_argument("pack", type=Path, help="folder containing the data pack")
    args = parser.parse_args()
    if not args.pack.is_dir():
        sys.exit(f"error: {args.pack} is not a folder")

    stations, data_files, gps_files, log_file = find_pack_files(args.pack)
    if not data_files:
        sys.exit(f"error: no logger CSV files (with a timestamp_utc column) under {args.pack}")
    maintenance = parse_maintenance_log(log_file)
    flags = Flags()

    # Logging interval: from the station table if present, else 10 minutes.
    interval_min = (int(stations["logger_interval_min"].mode().iloc[0])
                    if "logger_interval_min" in stations.columns else 10)

    # Best known position per station: GPS if available, else the install sheet.
    gps_data, positions, elevations = {}, {}, {}
    for station in data_files:
        gps = parse_nmea(gps_files[station]) if station in gps_files else None
        gps_data[station] = gps
        if gps is not None and gps["lat"].notna().sum() >= 10:
            positions[station] = (gps["lat"].median(), gps["lon"].median())
        elif station in stations.index:
            positions[station] = (stations.loc[station, "lat"], stations.loc[station, "lon"])
        else:
            positions[station] = (np.nan, np.nan)
        if station in stations.index:
            elevations[station] = float(stations.loc[station, "elevation_m"])

    # ---- Stage 1: load each station and repair its timestamps ----
    clean = {}
    for station, path in data_files.items():
        raw = load_logger_csv(path)
        clean[station] = fix_timestamps(station, raw, gps_data[station], positions[station][1],
                                        interval_min, flags)
    station_ids = sorted(clean)
    has_position = {s: not np.isnan(positions[s][0]) for s in station_ids}

    # One shared regular time grid covering every station's record.
    first = min(frame.index.min() for frame in clean.values())
    last = max(frame.index.max() for frame in clean.values())
    grid = pd.date_range(first, last, freq=f"{interval_min}min")
    for station in station_ids:
        check_gaps(station, clean[station], grid, interval_min, flags)

    # panels[channel] is a time x station table; masks[channel] marks excluded samples
    # so later checks (and neighbour medians) ignore data already known to be bad.
    panels = {channel: pd.DataFrame({s: clean[s][channel].reindex(grid) for s in station_ids})
              for channel in CHANNELS}
    masks = {channel: pd.DataFrame(False, index=grid, columns=station_ids) for channel in CHANNELS}

    # ---- Stage 2: single-station checks ----
    for station in station_ids:
        check_rh_saturation(station, panels, masks, interval_min, flags)
        check_impossible_and_spikes(station, panels, masks, flags)
        if has_position[station]:
            check_flatlines(station, panels, masks, positions, interval_min, flags)

    # ---- Stage 3: network checks (need neighbours) ----
    if len(station_ids) >= 3:
        for station in station_ids:
            check_steps(station, panels, masks, elevations, interval_min, flags)
        for station in station_ids:
            if has_position[station]:
                check_daytime_warm_bias(station, panels, masks, positions, flags)
            check_rain(station, panels, masks, interval_min, flags)
    for station in station_ids:
        if has_position[station]:
            check_solar_timing(station, panels, masks, positions, flags)

    # ---- Stage 4: metadata ----
    for station in station_ids:
        check_metadata(station, stations, gps_data.get(station), clean[station], flags)
    for station in stations.index.difference(station_ids):
        flags.add(station, "all", grid[0], grid[-1], "no data file", "exclude",
                  "stations.csv lists this station but the pack has no logger file for it.")
    check_elevation_outlier(panels, masks, elevations, flags)

    # ---- Stage 5: maintenance log, tidy, validate, write ----
    attach_maintenance_notes(flags, maintenance)
    for row in flags.rows:
        row["note"] = one_sentence(row["note"])
    result = flags.to_frame()
    validate_output(result)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(result)} flags to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
