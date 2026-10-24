Neil Cardoz
neilcardoz1@gmail.com
Video: [View the project video on Google Drive](https://drive.google.com/file/d/1Zrbl-UhqqhiWh1-BdoPJYmjIFR07V_BF/view?usp=sharing)
Run: `python checker.py <pack_folder>`

# Ridgeline weather network: data checker

`checker.py` reads a data pack (stations.csv, the logger CSVs, the `.nmea` GPS files and the
maintenance log, found anywhere under the folder you give it) and writes `flags.csv` into the
folder you run it from. Every row is one problem I believe is real, with `exclude`, `correct`
or `keep` and one sentence of evidence.

```
pip install -r requirements.txt
python checker.py path/to/data          # writes ./flags.csv (about 2 s)
python plotting.py path/to/data         # optional: writes s04_evidence.png
```

Only pandas and numpy are needed for the checker. Nothing about this particular pack (station
ids, dates, offsets) is hard-coded, and all thresholds are in one block at the top of the file.
I also ran it on a flat copy of the pack with no maintenance log and with S06's GPS file
removed: it still found every problem, including S06's clock error (from the sun instead of GPS).

## Files

| File | What it is |
|---|---|
| `checker.py` | the checker |
| `flags.csv` | its output for the pack I was given (16 flags) |
| `report.html` / `report.pdf` | one-page report for the network coordinator |
| `DECISIONS.md` | the technician visit, my least certain call, AI mistakes, next year, the by-hand sum |
| `plotting.py`, `s04_evidence.png` | the evidence behind the S04 warm-bias flag |
| `ai_logs/` | AI chat logs are saved here; the available Copilot session export is partial |

The data pack is not in this repository (see `.gitignore`).

## What each check does (in the order they run)

Timestamps are fixed first, because every later check compares a station with the others at
the same moment, and a wrong clock makes a healthy station look broken.

1. **Unreadable timestamps**: rows whose time cannot be parsed are dropped and flagged.
2. **Records sent twice**: rows that repeat an earlier row value-for-value (a re-sent upload)
   are dropped, keeping the first copy.
3. **Logger clock vs GPS**: the logger stamp on each `$GPRMC` sentence is compared with the GPS
   time in it (checksums checked, running median of 5 to ignore glitches). Offsets over 5 min are
   corrected. With no usable GPS it falls back to the sun: the daily sunlight peak must sit at
   solar noon, so a whole-hour error shows up there.
4. **Gaps**: runs of missing 10-minute records.
5. **Humidity stuck at saturation**: RH at 99 % or more for 6 h or longer while the other
   stations' median is below 85 %. A later 100 % spell when the neighbours are also at 90 % or
   more is kept as real fog.
6. **Impossible values and spikes**: outside physical limits, or one reading far from both of
   its neighbours (`-40` and `85` °C are named as sensor error codes).
7. **Stuck sensors**: the exact same value for 3 h or more (sunlight checked in daytime only;
   calm wind ignored).
8. **Sudden lasting steps against the network**: station minus the median of the other
   stations, for pressure, temperature and RH (offsets) and wind and sunlight (ratios). A step
   must be sharp (a sensor swap) and must last, not build up over hours (weather). The segment
   closest to the network is taken as the true one. A wind ratio within 10 % of 3.6 is corrected
   as m/s logged as km/h.
9. **Daytime warm bias**: each day, how much warmer than the neighbours the station gets per
   unit of sunshine. A clear lasting rise points to a radiation-shield problem.
10. **Blocked rain gauge**: 0 mm on days when most neighbours got rain, followed by a burst
    the neighbours did not see (water released when the funnel is cleaned).
11. **Solar timing**: backup clock check for every station, from the sunlight peak.
12. **Metadata**: install-sheet position and elevation against the GPS fixes, logging interval
    and install date against the data.
13. **Altitude explanation**: a station much colder or warmer than the rest by about what its
    elevation predicts is marked `keep`.
14. **Maintenance log**: a flag gets a short quote from a volunteer note for the same station
    within a day of the problem, if the note mentions the right thing (e.g. "anemometer" for wind).
15. **Output check**: allowed channels and actions, times in order, one-sentence notes.

Corrections and exclusions are applied as the checks run, so a broken neighbour does not
distort the comparisons for the others.

## What I did not get to

- **The S04 warm-bias start date** is a statistical split (3 April). The bias seems to build up
  from about 19 March, and that build-up period is not flagged.
- **Corrections are simple**: a constant pressure offset, a constant wind factor, a whole clock
  shift. A slowly drifting sensor would only be caught when it is large.
- **Thresholds were chosen by looking at this one pack.** On a new pack I would expect some
  borderline false alarms or misses, especially for rain (showers are patchy) and small steps.
- **No checks** for slow drift, wind direction offsets (a vane turned on its mast), or dirt
  on the sunlight sensor.
- **The 35 h gap at S08** cannot be filled; it is only flagged so rain totals are not trusted.
- **No tests** beyond running it on altered copies of the pack.

