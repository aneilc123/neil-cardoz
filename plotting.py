# Evidence plots for the S04 warm-bias flag.
# Usage: python plotting.py <pack_folder>   -> writes s04_evidence.png
import sys, glob, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

pack = sys.argv[1]
files = sorted(glob.glob(os.path.join(pack, "**", "S0*.csv"), recursive=True))
T, G = {}, {}
for p in files:
    s = os.path.basename(p)[:3]
    d = pd.read_csv(p, parse_dates=["timestamp_utc"]).drop_duplicates("timestamp_utc", keep="first")
    d = d.set_index("timestamp_utc")
    if s == "S06":                       # S06 clock: add 6 h after the reboot (see flags.csv)
        continue
    T[s], G[s] = d.temp_c, d.ghi_wm2
T, G = pd.DataFrame(T), pd.DataFrame(G)
T = T.where((T > -39) & (T < 80))        # drop S03 error codes
T.loc["2025-03-18 06:00":"2025-03-21 14:00", "S03"] = np.nan   # S03 stuck
others = [c for c in T if c not in ("S04", "S01")]   # S01 is 1,300 m higher
anom = T - T.median()
resid = anom["S04"] - anom[others].median(axis=1)    # S04 minus typical neighbour

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# 1. One sunny day early vs late: S04 against the other stations
for day, style in [("2025-03-12", ":"), ("2025-04-11", "-")]:
    d = T.loc[day]
    h = d.index.hour + d.index.minute / 60
    ax[0].plot(h, d[others].median(axis=1), "k" + style, lw=1.5, label=f"others {day[5:]}")
    ax[0].plot(h, d["S04"], style, color="#c8102e", lw=2, label=f"S04 {day[5:]}")
ax[0].set(title="Sunny days: S04 vs the other stations", xlabel="hour (UTC)", ylabel="°C")
ax[0].legend(fontsize=8)

# 2. Excess temperature vs sunshine: before and after mid-March
day_ = resid.index.normalize()
for (lo, hi), c, lab in [(("2025-03-01", "2025-03-14"), "0.5", "1–14 Mar"),
                         (("2025-04-01", "2025-04-11"), "#c8102e", "1–11 Apr")]:
    m = (resid.index >= lo) & (resid.index < pd.Timestamp(hi) + pd.Timedelta(days=1))
    ax[1].scatter(G["S04"][m], resid[m], s=4, c=c, label=lab, alpha=.5)
ax[1].axhline(0, color="k", lw=.8)
ax[1].set(title="S04 excess warmth grows with sunshine", xlabel="S04 sunlight (W/m²)",
          ylabel="S04 minus others (°C)")
ax[1].legend(fontsize=8, markerscale=3)

# 3. Daily slope (°C per 1000 W/m²) through time, with the paint date
slopes = {}
df = pd.DataFrame({"r": resid, "g": G["S04"] / 1000}).dropna()
df = df[df.g > 0.1]
for d, g in df.groupby(df.index.normalize()):
    if len(g) > 12:
        slopes[d] = np.polyfit(g.g, g.r, 1)[0]
sl = pd.Series(slopes)
ax[2].bar(sl.index, sl.values, color=["#c8102e" if v > 2 else "0.6" for v in sl.values], width=.8)
ax[2].axvline(pd.Timestamp("2025-03-15"), color="k", ls="--")
ax[2].text(pd.Timestamp("2025-03-15"), ax[2].get_ylim()[1] * .9, " post painted", fontsize=8)
ax[2].set(title="How much warmer per unit of sunshine, each day", ylabel="°C per 1000 W/m²")
ax[2].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig("s04_evidence.png", dpi=110)

# Numbers to quote
sunny = G["S04"] > 600
for lo, hi in [("2025-03-01", "2025-03-14"), ("2025-03-19", "2025-03-27"), ("2025-03-28", "2025-04-11")]:
    r = resid[lo:hi]; print(f"{lo}..{hi}: median excess in sun {r[sunny[lo:hi]].median():+.1f} °C, "
                            f"at night {r[G['S04'][lo:hi] == 0].median():+.1f} °C, max {r.max():+.1f}")
print(sl.round(1).to_string())
