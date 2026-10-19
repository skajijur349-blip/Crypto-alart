"""
Angel Ruh signal bot (GitHub Actions only - no TradingView needed)

Every 3 minutes (right after a 3m candle closes) it:
  1. downloads Delta Exchange futures candles (ETHUSD, ZECUSD, NEARUSD)
  2. looks for your rejection entry on the last CLOSED 3m, 4H and 1D candle:
        long wick + volume condition + level tested before (+ higher-timeframe check for 4H/1D)
  3. if there is a NEW BUY / SELL it sends a Telegram message plus charts for
     3m, 15m, 30m, 1H, 4H, 1D with BUY / SELL labels (period high/low with hit count,
     and entry markers), like the Angel Ruh indicator on TradingView.
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
# SETTINGS (change these if you want)
# =====================================================================
TOKEN = os.environ.get("TG_BOT_TOKEN", "")
CHAT = os.environ.get("TG_CHAT_ID", "")
BASE = os.environ.get("DELTA_BASE", "https://api.india.delta.exchange")
TEST_MODE = os.environ.get("TEST_MODE", "").strip().lower() == "true"
LOOP_MINUTES = float(os.environ.get("LOOP_MINUTES", "0") or 0)   # >0 = keep checking inside one run
CHECK_EVERY = 180                                               # seconds between checks (3 minutes)

SYMBOLS = ["ETHUSD", "ZECUSD", "NEARUSD"]   # only these Delta futures are used
SIGNAL_TFS = ["3m", "4h", "1d"]         # timeframes where the entry rule is checked
HTF_OF = {"4h": "1d", "1d": "1w"}       # higher timeframe used for confirmation (3m has none)
CHART_TFS = ["3m", "15m", "30m", "1h", "4h", "1d"]

TF_SEC = {"3m": 180, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
FETCH_N = {"3m": 960, "15m": 480, "30m": 360, "1h": 300, "4h": 300, "1d": 420}
SHOW_N = 100                            # candles drawn on each chart

# rejection rules (same meaning as the Pine script)
LB = 50                 # level lookback (candles)
LVL_TOL = 0.3           # % - what counts as "same price"
MIN_TESTS = 1           # earlier touches of the level
WICK_MIN = 0.55         # wick must be >= 55% of candle range
MIN_RNG_ATR = 0.6       # ignore tiny candles (x ATR)
VOL_MODE = "up"         # "up" (above average), "down" (below average) or "any"
VOL_LEN = 20
VOL_MULT = 1.2
ATR_LEN = 14
SL_BUF = 0.3            # stop buffer beyond the wick (x ATR)
RR = 2.0                # take profit = risk x RR
HTF_MODE = "rejection"  # "rejection" = higher TF also printed a rejection, "off" = ignore
HTF_RECENT = 2          # ... within the last N higher-TF candles
HTF_LB = 20
HTF_TOL = 0.5
PERIOD_TOL = 0.1        # % tolerance for the period high/low hit count

BG = "#fbfbc9"          # chart background (like your TradingView theme)


# =====================================================================
# DATA
# =====================================================================
_cache = {}


def fetch(sym, tf):
    key = (sym, tf)
    if key in _cache:
        return _cache[key]
    if tf == "1w":
        d = fetch(sym, "1d")
        w = d.resample("W-MON", closed="left", label="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Open"])
        _cache[key] = w
        return w
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
        except Exception as e:  # retry
            last_err = e
            time.sleep(2)
    raise last_err


def closed_part(df, tf):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return df[df.index + pd.Timedelta(seconds=TF_SEC[tf]) <= now]


# =====================================================================
# INDICATORS
# =====================================================================
def atr_series(df, n):
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi_series(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def analyze(df, lb, tol, min_tests):
    """Long-wick rejection at a tested level. Returns bull/bear flags per candle."""
    o, h, l, c, v = (df[x] for x in ["Open", "High", "Low", "Close", "Volume"])
    rng = h - l
    safe = rng.replace(0, np.nan)
    lo_pct = ((np.minimum(o, c) - l) / safe).fillna(0)
    up_pct = ((h - np.maximum(o, c)) / safe).fillna(0)
    atr = atr_series(df, ATR_LEN)
    big = rng >= MIN_RNG_ATR * atr

    vavg = v.rolling(VOL_LEN).mean()
    if VOL_MODE == "any":
        vol_ok = pd.Series(True, index=df.index)
    elif VOL_MODE == "up":
        vol_ok = v > vavg * VOL_MULT
    else:
        vol_ok = v < vavg
    vol_ok = vol_ok | (v <= 0) | v.isna()          # no volume data -> ignore

    bt = pd.Series(0, index=df.index)
    st = pd.Series(0, index=df.index)
    for k in range(1, lb + 1):
        bt += ((l.shift(k) - l).abs() <= l * tol / 100).astype(int)
        st += ((h.shift(k) - h).abs() <= h * tol / 100).astype(int)

    low_prev = l.rolling(lb).min().shift(1)
    high_prev = h.rolling(lb).max().shift(1)
    at_sup = l <= low_prev * (1 + tol / 100)
    at_res = h >= high_prev * (1 - tol / 100)

    bull = big & (lo_pct >= WICK_MIN) & vol_ok & at_sup & (bt >= min_tests)
    bear = big & (up_pct >= WICK_MIN) & vol_ok & at_res & (st >= min_tests)
    return pd.DataFrame(
        {"bull": bull.fillna(False).astype(bool), "bear": bear.fillna(False).astype(bool),
         "bt": bt, "st": st, "atr": atr},
        index=df.index,
    )


def find_signals(sym, tf):
    """All entry signals on CLOSED candles of tf (history + latest)."""
    df = closed_part(fetch(sym, tf), tf)
    if len(df) < LB + 5:
        return []
    f = analyze(df, LB, LVL_TOL, MIN_TESTS)
    ok_long = f["bull"].copy()
    ok_short = f["bear"].copy()

    if HTF_MODE == "rejection" and tf in HTF_OF:
        htf = HTF_OF[tf]
        hdf = closed_part(fetch(sym, htf), htf)
        if len(hdf) >= HTF_LB + 5:
            hf = analyze(hdf, HTF_LB, HTF_TOL, MIN_TESTS)
            hb = hf["bull"].astype(int).rolling(HTF_RECENT, min_periods=1).max().astype(bool).values
            hs = hf["bear"].astype(int).rolling(HTF_RECENT, min_periods=1).max().astype(bool).values
            h_close = (hdf.index + pd.Timedelta(seconds=TF_SEC[htf])).values
            s_close = (df.index + pd.Timedelta(seconds=TF_SEC[tf])).values
            pos = np.searchsorted(h_close, s_close, side="right") - 1
            valid = pos >= 0
            pc = np.clip(pos, 0, None)
            ok_long = ok_long & pd.Series(np.where(valid, hb[pc], False), index=df.index)
            ok_short = ok_short & pd.Series(np.where(valid, hs[pc], False), index=df.index)
        else:
            ok_long = ok_long & False
            ok_short = ok_short & False

    sigs = []
    mask = (ok_long | ok_short).values
    for i in np.where(mask)[0]:
        ts = df.index[i]
        side = "BUY" if ok_long.iloc[i] else "SELL"
        e = float(df["Close"].iloc[i])
        atr = float(f["atr"].iloc[i])
        if side == "BUY":
            sl = float(df["Low"].iloc[i]) - atr * SL_BUF
            risk = e - sl
            tp = e + risk * RR
            tested = int(f["bt"].iloc[i])
        else:
            sl = float(df["High"].iloc[i]) + atr * SL_BUF
            risk = sl - e
            tp = e - risk * RR
            tested = int(f["st"].iloc[i])
        if risk <= 0:
            continue
        sigs.append(dict(
            sym=sym, tf=tf, side=side, ts=ts,
            close_ts=ts + pd.Timedelta(seconds=TF_SEC[tf]),
            entry=e, sl=sl, tp=tp, tested=tested,
            last=(i == len(df) - 1), current=False,
        ))
    return sigs


# =====================================================================
# ANGEL RUH PERIOD HIGH / LOW MARKERS (same idea as the Pine indicator)
# =====================================================================
def period_markers(df, group):
    shifted = df.index + pd.Timedelta(hours=5, minutes=30)          # IST periods
    if group == "W":
        keys = np.asarray(shifted.to_period("W-SUN").astype(str))
    elif group == "M":
        keys = np.asarray(shifted.to_period("M").astype(str))
    else:
        keys = np.asarray(shifted.strftime("%Y-%m-%d"))
    order = list(dict.fromkeys(keys))
    out = []
    for name in order[1:-1]:                  # skip first (partial) and last (still forming)
        pos = np.where(keys == name)[0]
        highs = df["High"].values[pos]
        lows = df["Low"].values[pos]
        p_high, p_low = highs.max(), lows.min()
        h_hits = int((highs >= p_high * (1 - PERIOD_TOL / 100)).sum())
        l_hits = int((lows <= p_low * (1 + PERIOD_TOL / 100)).sum())
        out.append((int(pos[np.argmax(highs == p_high)]), float(p_high), h_hits, "SELL"))
        out.append((int(pos[np.argmax(lows == p_low)]), float(p_low), l_hits, "BUY"))
    return out


# =====================================================================
# CHART
# =====================================================================
GROUP_OF = {"3m": "D", "15m": "D", "30m": "D", "1h": "D", "4h": "W", "1d": "M"}


def fmt(p):
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.3f}"
    return f"{p:.5f}"


def make_chart(df, tf, sym, sigs, path):
    total = len(df)
    show = min(SHOW_N, total)
    off = total - show
    dfp = df.iloc[off:]

    marks = [(p - off, price, hits, side)
             for (p, price, hits, side) in period_markers(df, GROUP_OF[tf]) if p >= off]

    rsi = rsi_series(df["Close"], 14)
    ema9 = rsi.ewm(span=9, adjust=False).mean()
    wma45 = wma(rsi, 45)
    ap = [
        mpf.make_addplot(rsi.iloc[off:], panel=2, color="#8e24aa", width=1.3),
        mpf.make_addplot(ema9.iloc[off:], panel=2, color="#26c6da", width=1.0),
        mpf.make_addplot(wma45.iloc[off:], panel=2, color="#ffa726", width=1.0),
        mpf.make_addplot(pd.Series(50.0, index=dfp.index), panel=2, color="#9e9e9e", width=0.8),
    ]

    last_close = float(dfp["Close"].iloc[-1])
    hl_vals, hl_cols = [last_close], ["#2962ff"]
    # latest closed period high / low lines (black)
    for side in ("SELL", "BUY"):
        ms = [m for m in marks if m[3] == side]
        if ms:
            hl_vals.append(ms[-1][1])
            hl_cols.append("#222222")

    # entry markers that fall inside this chart
    entries = []
    cur = None
    for s in sigs:
        t = s["close_ts"]
        if t < dfp.index[0] or t >= dfp.index[-1] + pd.Timedelta(seconds=TF_SEC[tf]):
            continue
        x = int(dfp.index.searchsorted(t, side="right") - 1)
        entries.append((x, s))
        if s.get("current"):
            cur = s
    if cur is not None:
        hl_vals += [cur["entry"], cur["sl"], cur["tp"]]
        hl_cols += ["#757575", "#d32f2f", "#2e7d32"]

    mc = mpf.make_marketcolors(
        up="white", down="#ff5722",
        edge={"up": "#009688", "down": "#ff5722"},
        wick={"up": "#009688", "down": "#ff5722"},
        volume="#2962ff",
    )
    style = mpf.make_mpf_style(marketcolors=mc, facecolor=BG, figcolor=BG,
                               gridstyle="", y_on_right=True, edgecolor="#555555")
    fmt_dt = "%d %b" if tf == "1d" else "%d %H:%M"
    fig, axes = mpf.plot(
        dfp, type="candle", style=style, volume=True, addplot=ap,
        panel_ratios=(6, 2, 2.5), hlines=dict(hlines=hl_vals, colors=hl_cols, linestyle="-", linewidths=1),
        figsize=(9, 9.5), title=f"{sym}  {tf.upper()}  Delta Futures",
        returnfig=True, tight_layout=True, datetime_format=fmt_dt, xrotation=0,
    )
    ax = axes[0]

    # period high/low labels (like the Angel Ruh indicator)
    for (x, price, hits, side) in marks:
        col = "#ef5350" if side == "SELL" else "#43a047"
        up = side == "SELL"
        ax.annotate(
            f"{side} {fmt(price)}\n(hit {hits}x)", xy=(x, price),
            xytext=(0, 24 if up else -24), textcoords="offset points",
            ha="center", va="bottom" if up else "top", fontsize=6.5, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8), zorder=10,
        )
        ax.plot([x], [price], marker="v" if up else "^", color=col, markersize=5, zorder=10)

    # entry signals (BUY / SELL) on this chart
    for (x, s) in entries:
        buy = s["side"] == "BUY"
        col = "#00c853" if buy else "#d500f9"
        y = float(dfp["Low"].iloc[x]) if buy else float(dfp["High"].iloc[x])
        is_cur = bool(s.get("current"))
        if is_cur:
            label = (f"NEW ENTRY {s['side']} ({s['tf'].upper()})\n"
                     f"E {fmt(s['entry'])}  SL {fmt(s['sl'])}\nTP {fmt(s['tp'])}")
        else:
            label = f"{s['side']} {s['tf'].upper()}"
        dist = 52 if is_cur else 30
        ax.annotate(
            label, xy=(x, y), xytext=(0, -dist if buy else dist),
            textcoords="offset points", ha="center", va="top" if buy else "bottom",
            fontsize=7 if is_cur else 5.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3" if is_cur else "round,pad=0.15", fc=col, ec="black", lw=0.8),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5 if is_cur else 1.0), zorder=11 if is_cur else 9,
        )

    # current price tag
    ax.annotate(fmt(last_close), xy=(show - 1, last_close), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=7, color="white",
                bbox=dict(boxstyle="square,pad=0.25", fc="#2962ff", ec="none"), zorder=12)

    lo = float(dfp["Low"].min())
    hi = float(dfp["High"].max())
    if cur is not None:
        lo = min(lo, cur["sl"], cur["tp"])
        hi = max(hi, cur["sl"], cur["tp"])
    pad = (hi - lo) * 0.14
    ax.set_ylim(lo - pad * 1.5, hi + pad * 1.5)
    ax.set_xlim(-1, show + 7)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# =====================================================================
# TELEGRAM
# =====================================================================
def tg(method, **kw):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/{method}", timeout=90, **kw)
    if not r.ok:
        print("Telegram error:", r.status_code, r.text)
    return r


def send_with_charts(sym, text, sigs):
    ok = tg("sendMessage", data={"chat_id": CHAT, "text": text}).ok
    paths = []
    for tf in CHART_TFS:
        try:
            df = fetch(sym, tf)
            if len(df) < 20:
                print("not enough data for", sym, tf)
                continue
            path = f"chart_{sym}_{tf}.png"
            make_chart(df, tf, sym, sigs, path)
            paths.append(path)
        except Exception as e:
            print("chart failed for", sym, tf, e)
    if len(paths) >= 2:
        media = [{"type": "photo", "media": f"attach://f{i}"} for i in range(len(paths))]
        media[0]["caption"] = text[:1000]
        files = {f"f{i}": open(p, "rb") for i, p in enumerate(paths)}
        tg("sendMediaGroup", data={"chat_id": CHAT, "media": json.dumps(media)}, files=files)
    elif len(paths) == 1:
        tg("sendPhoto", data={"chat_id": CHAT, "caption": text[:1000]}, files={"photo": open(paths[0], "rb")})
    return ok


def signal_text(s):
    emoji = "🟢" if s["side"] == "BUY" else "🔴"
    lines = [
        f"{emoji} {s['side']} SIGNAL",
        f"{s['sym']} (Delta Futures) | Signal TF: {s['tf'].upper()}",
        f"Entry ~ {fmt(s['entry'])}",
        f"SL: {fmt(s['sl'])}",
        f"TP: {fmt(s['tp'])}  (RR 1:{RR:g})",
        f"Level tested: {s['tested']}x",
    ]
    if HTF_MODE == "rejection" and s["tf"] in HTF_OF:
        lines.append(f"Higher TF ({HTF_OF[s['tf']].upper()}) rejection confirmed")
    return "\n".join(lines)


# =====================================================================
# STATE (so the same signal is never sent twice)
# =====================================================================
def load_state():
    try:
        with open("state.json") as f:
            return json.load(f)
    except Exception:
        return {"sent": []}


def save_state(state):
    state["sent"] = state["sent"][-300:]
    with open("state.json", "w") as f:
        json.dump(state, f, indent=1)


def sig_key(s):
    return f"{s['sym']}|{s['tf']}|{s['ts'].isoformat()}|{s['side']}"


# =====================================================================
# MAIN
# =====================================================================
def commit_state():
    """Save state.json to the repository right away (so a restart never repeats a signal)."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(["git", "config", "user.name", "signal-bot"], check=True)
        subprocess.run(["git", "config", "user.email", "signal-bot@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", "state.json"], check=True)
        subprocess.run(["git", "commit", "-m", "update state"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print("could not commit state:", e)


def run_once():
    _cache.clear()                                  # always use fresh candles
    state = load_state()
    sent = list(state.get("sent", []))
    changed = False

    for sym in SYMBOLS:
        try:
            all_sigs = []
            for tf in SIGNAL_TFS:
                all_sigs += find_signals(sym, tf)
        except Exception as e:
            print(f"{sym}: could not analyse ({e})")
            continue

        new = [s for s in all_sigs if s["last"] and sig_key(s) not in sent]
        print(f"{time.strftime('%H:%M:%S')} {sym}: {len(all_sigs)} historical signals, {len(new)} new")

        if TEST_MODE:
            text = (f"🧪 TEST - {sym} (Delta Futures)\nCharts: 3m 15m 30m 1H 4H 1D\n"
                    "Labels: period High/Low (hit count) + ENTRY BUY/SELL markers")
            send_with_charts(sym, text, all_sigs)

        for s in new:
            s["current"] = True
            if send_with_charts(sym, signal_text(s), all_sigs):
                sent.append(sig_key(s))
                changed = True

    if changed:
        state["sent"] = sent
        save_state(state)
        commit_state()


def main():
    if TEST_MODE or LOOP_MINUTES <= 0:
        run_once()
        return
    deadline = time.time() + LOOP_MINUTES * 60
    while True:
        now = time.time()
        nxt = (now // CHECK_EVERY + 1) * CHECK_EVERY + 5   # next 3m close + 5s for the exchange to publish it
        if nxt >= deadline:
            break
        time.sleep(max(0, nxt - now))
        try:
            run_once()
        except Exception as e:
            print("cycle failed:", e)


if __name__ == "__main__":
    main()
