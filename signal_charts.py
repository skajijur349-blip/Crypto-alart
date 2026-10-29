"""
Day high / day low alert bot (GitHub Actions only)

Checked right after every 1-minute candle closes:
  - closed 1m candle makes a NEW DAY HIGH (IST day) -> photo + text
  - closed 1m candle makes a NEW DAY LOW  (IST day) -> photo + text
Photo  : one plain 1H candle chart (about 5 days) with horizontal support / resistance
         lines at price levels that were rejected several times. No arrows, no labels.
Text   : New High / Target previous high / Stop loss current low
         (New Low / Target previous low / Stop loss current high)
"""
import json
import os
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests

# =====================================================================
# SETTINGS
# =====================================================================
TOKEN = os.environ.get("TG_BOT_TOKEN", "")
CHAT = os.environ.get("TG_CHAT_ID", "")
BASE = os.environ.get("DELTA_BASE", "https://api.india.delta.exchange")
TEST_MODE = os.environ.get("TEST_MODE", "").strip().lower() == "true"
LOOP_MINUTES = float(os.environ.get("LOOP_MINUTES", "0") or 0)
CHECK_EVERY = 60                        # check once per 1m candle close

SYMBOLS = ["ETHUSD", "ZECUSD", "NEARUSD"]
SIGNAL_TF = "1m"
CHART_TF = "1h"

MIN_BARS = 30           # ignore the first N minutes after midnight IST (no real day high/low yet)
COOLDOWN_MIN = 0        # minutes to wait before another signal of the same coin + same side (0 = off)
HEARTBEAT_MIN = 60      # "bot alive" message every N minutes (0 = off)

TF_SEC = {"1m": 60, "1h": 3600}
FETCH_N = {"1m": 1500, "1h": 300}
SHOW_N = 120            # 1H candles drawn on the chart (about 5 days)
LOOKBACK = 240          # 1H candles used to find support / resistance
PIV = 3                 # a swing high / low = highest / lowest of 3 candles on each side
LEVEL_TOL = 0.25        # % - swing points this close count as the same level
MIN_TOUCH = 3           # touches needed for a strong level (falls back to 2 if none)
LEVELS_PER_SIDE = 2     # lines drawn above and below the current price

BG = "#fbfbc9"


# =====================================================================
# DATA
# =====================================================================
_cache = {}


def fetch(sym, tf):
    key = (sym, tf)
    if key in _cache:
        return _cache[key]
    end = int(time.time())
    start = end - FETCH_N[tf] * TF_SEC[tf]
    last_err = None
    for _ in range(3):
        try:
            r = requests.get(
                f"{BASE}/v2/history/candles",
                params={"symbol": sym, "resolution": tf, "start": start, "end": end},
                timeout=25,
            )
            r.raise_for_status()
            rows = r.json().get("result") or []
            if not rows:
                raise ValueError(f"no candles returned for {sym} {tf}")
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["time"], unit="s")
            df = df.sort_index().rename(columns=str.capitalize)
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            df = df[~df.index.duplicated()]
            _cache[key] = df
            return df
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def closed_part(df, tf):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return df[df.index + pd.Timedelta(seconds=TF_SEC[tf]) <= now]


def fmt(p):
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.3f}"
    return f"{p:.5f}"


# =====================================================================
# SIGNAL: closed 1m candle makes a new day high / day low (IST day)
# =====================================================================
def day_keys(index):
    return np.asarray((index + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d"))


def find_signals(sym):
    df = closed_part(fetch(sym, SIGNAL_TF), SIGNAL_TF)
    if len(df) < 60:
        return [], None
    keys = day_keys(df.index)

    hi_cm = df["High"].groupby(keys).cummax()
    lo_cm = df["Low"].groupby(keys).cummin()
    prev_hi = hi_cm.groupby(keys).shift(1)          # day high BEFORE this candle
    prev_lo = lo_cm.groupby(keys).shift(1)          # day low BEFORE this candle
    bars = df["Close"].groupby(keys).cumcount()     # candle number inside the day

    i = len(df) - 1                                 # only the last closed candle matters
    row = df.iloc[i]
    ok_bars = bool(bars.iloc[i] >= MIN_BARS)
    new_hi = ok_bars and bool(row["High"] > prev_hi.iloc[i])
    new_lo = ok_bars and bool(row["Low"] < prev_lo.iloc[i])
    last_ts = df.index[i]
    if new_hi == new_lo:                            # nothing, or one candle took both ends
        return [], last_ts

    if new_hi:
        s = dict(sym=sym, ts=last_ts, kind="HIGH", level=float(row["High"]),
                 prev=float(prev_hi.iloc[i]), stop=float(row["Low"]))
    else:
        s = dict(sym=sym, ts=last_ts, kind="LOW", level=float(row["Low"]),
                 prev=float(prev_lo.iloc[i]), stop=float(row["High"]))
    return [s], last_ts


def signal_text(s):
    if s["kind"] == "HIGH":
        return (f"{s['sym']}\n"
                f"New High: {fmt(s['level'])}\n"
                f"Target: previous high {fmt(s['prev'])}\n"
                f"Stop loss: current low {fmt(s['stop'])}")
    return (f"{s['sym']}\n"
            f"New Low: {fmt(s['level'])}\n"
            f"Target: previous low {fmt(s['prev'])}\n"
            f"Stop loss: current high {fmt(s['stop'])}")


# =====================================================================
# CHART: plain 1H candles + horizontal support / resistance lines
# =====================================================================
def swing_levels(d):
    """Price levels where several swing highs / lows sit at almost the same price.
    Returns a list of (price, first_position, touches)."""
    h = d["High"].values
    l = d["Low"].values
    n = len(d)
    piv = []
    for i in range(PIV, n - PIV):
        if h[i] >= h[i - PIV:i + PIV + 1].max():
            piv.append((float(h[i]), i))
        if l[i] <= l[i - PIV:i + PIV + 1].min():
            piv.append((float(l[i]), i))
    piv.sort()
    groups = []
    for p, i in piv:
        if groups and p <= np.mean([x[0] for x in groups[-1]]) * (1 + LEVEL_TOL / 100):
            groups[-1].append((p, i))
        else:
            groups.append([(p, i)])
    return [(float(np.mean([x[0] for x in g])), min(x[1] for x in g), len(g)) for g in groups]


def pick_levels(d, last_close):
    """Nearest strong levels above (resistance) and below (support) the current price."""
    lv = swing_levels(d)
    for need in (MIN_TOUCH, 2):
        good = [x for x in lv if x[2] >= need]
        above = sorted([x for x in good if x[0] > last_close])[:LEVELS_PER_SIDE]
        below = sorted([x for x in good if x[0] <= last_close], reverse=True)[:LEVELS_PER_SIDE]
        if above or below:
            return above + below
    return [(float(d["High"].max()), 0, 1), (float(d["Low"].min()), 0, 1)]


def make_chart(sym, path):
    df = fetch(sym, CHART_TF)
    d = df.iloc[-LOOKBACK:]
    dfp = d.iloc[-SHOW_N:]
    off = len(d) - len(dfp)
    n = len(dfp)
    levels = pick_levels(d, float(dfp["Close"].iloc[-1]))

    mc = mpf.make_marketcolors(
        up="white", down="#ff5722",
        edge={"up": "#009688", "down": "#ff5722"},
        wick={"up": "#009688", "down": "#ff5722"},
    )
    style = mpf.make_mpf_style(marketcolors=mc, facecolor=BG, figcolor=BG,
                               gridstyle="", y_on_right=True, edgecolor="#555555")
    fig, axes = mpf.plot(
        dfp, type="candle", style=style, volume=False,
        figsize=(9, 7), title=f"{sym}  1H  Delta Futures",
        returnfig=True, tight_layout=True, datetime_format="%d %b", xrotation=0,
    )
    ax = axes[0]
    for price, first, touches in levels:
        x0 = max(first - off, 0)                       # line starts at the first touch, runs to the right
        ax.hlines(price, x0, n + 7, colors="black", linewidth=1.3, zorder=3)
        ax.annotate(fmt(price), xy=(n - 1, price), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7, color="white",
                    bbox=dict(boxstyle="square,pad=0.25", fc="black", ec="none"), zorder=12)
    lo = min([float(dfp["Low"].min())] + [x[0] for x in levels])
    hi = max([float(dfp["High"].max())] + [x[0] for x in levels])
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-1, n + 7)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# =====================================================================
# TELEGRAM
# =====================================================================
def tg(method, **kw):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/{method}", timeout=90, **kw)
    if not r.ok:
        print("Telegram error:", r.status_code, r.text, flush=True)
    return r


def send_photo(sym, caption):
    """One chart photo with the text under it. Falls back to text only if the chart fails."""
    path = f"chart_{sym}.png"
    try:
        make_chart(sym, path)
        with open(path, "rb") as f:
            return tg("sendPhoto", data={"chat_id": CHAT, "caption": caption[:1000]},
                      files={"photo": f}).ok
    except Exception as e:
        print("chart failed for", sym, e, flush=True)
        return tg("sendMessage", data={"chat_id": CHAT, "text": caption}).ok


# =====================================================================
# STATE (so the same signal is never sent twice)
# =====================================================================
def load_state():
    try:
        with open("state.json") as f:
            st = json.load(f)
            st.setdefault("sent", [])
            st.setdefault("last_sent", {})
            return st
    except Exception:
        return {"sent": [], "last_sent": {}}


def save_state(state):
    state["sent"] = state["sent"][-300:]
    with open("state.json", "w") as f:
        json.dump(state, f, indent=1)


def sig_key(s):
    return f"{s['sym']}|{SIGNAL_TF}|{s['ts'].isoformat()}|{s['kind']}"


def commit_state():
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(["git", "config", "user.name", "signal-bot"], check=True)
        subprocess.run(["git", "config", "user.email", "signal-bot@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", "state.json"], check=True)
        subprocess.run(["git", "commit", "-m", "update state"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print("could not commit state:", e, flush=True)


# =====================================================================
# MAIN
# =====================================================================
def run_once():
    _cache.clear()
    state = load_state()
    sent = list(state["sent"])
    last_sent = state["last_sent"]
    changed = False

    for sym in SYMBOLS:
        if TEST_MODE:
            send_photo(sym, f"🧪 TEST {sym}\n1H chart + support / resistance lines")
            continue
        try:
            sigs, last_ts = find_signals(sym)
        except Exception as e:
            print(f"{sym}: could not analyse ({e})", flush=True)
            continue
        print(f"{time.strftime('%H:%M:%S')} {sym}: last closed 1m {last_ts} -> "
              f"{sigs[0]['kind'] if sigs else 'no signal'}", flush=True)

        for s in sigs:
            if sig_key(s) in sent:
                continue
            ck = f"{sym}|{s['kind']}"
            if COOLDOWN_MIN and time.time() - last_sent.get(ck, 0) < COOLDOWN_MIN * 60:
                continue
            if send_photo(sym, signal_text(s)):
                sent.append(sig_key(s))
                last_sent[ck] = time.time()
                changed = True

    if changed:
        state["sent"] = sent
        state["last_sent"] = last_sent
        save_state(state)
        commit_state()


def main():
    print(f"START test={TEST_MODE} loop={LOOP_MINUTES} "
          f"token={'set' if TOKEN else 'MISSING'} chat={'set' if CHAT else 'MISSING'}", flush=True)
    if TEST_MODE:
        run_once()
        return
    tg("sendMessage", data={"chat_id": CHAT, "text": "✅ Live 1m checker started"})
    deadline = time.time() + max(LOOP_MINUTES, 1) * 60
    last_hb = time.time()
    while True:
        now = time.time()
        nxt = (now // CHECK_EVERY + 1) * CHECK_EVERY + 5      # next 1m close + 5s
        if nxt >= deadline:
            break
        time.sleep(max(0, nxt - now))
        try:
            run_once()
        except Exception as e:
            print("cycle failed:", e, flush=True)
        if HEARTBEAT_MIN and time.time() - last_hb >= HEARTBEAT_MIN * 60:
            tg("sendMessage", data={"chat_id": CHAT, "text": "💓 Bot alive"})
            last_hb = time.time()


if __name__ == "__main__":
    main()
