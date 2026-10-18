import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import pandas as pd
import requests

TOKEN = os.environ["TG_BOT_TOKEN"]
CHAT = os.environ["TG_CHAT_ID"]
BASE = os.environ.get("DELTA_BASE", "https://api.india.delta.exchange")

# Only these Delta Exchange futures symbols are allowed
ALLOWED = {"ETHUSD", "ZECUSD"}

# timeframe -> seconds per candle
TFS = {"3m": 180, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}

payload = json.loads(os.environ.get("PAYLOAD") or "null") or {}

# manual "Run workflow" test (no payload)
if not payload.get("symbol"):
    payload = {"side": "BUY", "symbol": "DELTAIN:ETHUSD.P", "tf": "1D", "tested": "2"}


def norm_symbol(s):
    sym = str(s).split(":", 1)[-1].upper()
    sym = sym.replace(".P", "").replace("PERP", "")
    if sym.endswith("USDT"):
        sym = sym[:-4] + "USD"
    return sym


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def delta_df(sym, tf, n=120):
    end = int(time.time())
    start = end - n * TFS[tf]
    r = requests.get(
        f"{BASE}/v2/history/candles",
        params={"symbol": sym, "resolution": tf, "start": start, "end": end},
        timeout=20,
    )
    r.raise_for_status()
    rows = r.json().get("result", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["time"], unit="s")
    df = df.sort_index().rename(columns=str.capitalize)
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def make_chart(df, title, entry, sl, tp, path):
    df = df.tail(90)
    levels, colors = [], []
    for v, c in ((entry, "#bdbdbd"), (sl, "#ef5350"), (tp, "#26a69a")):
        if v:
            levels.append(v)
            colors.append(c)
    kw = {}
    if levels:
        kw["hlines"] = dict(hlines=levels, colors=colors, linestyle="--", linewidths=1)
    mpf.plot(
        df, type="candle", style="nightclouds", title=title, volume=False,
        figsize=(9, 5.5), tight_layout=True,
        savefig=dict(fname=path, dpi=110), **kw,
    )


def tg(method, **kw):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/{method}", timeout=60, **kw)
    if not r.ok:
        print("Telegram error:", r.status_code, r.text)
    return r


sym = norm_symbol(payload["symbol"])
if sym not in ALLOWED:
    print(f"Ignored symbol {payload['symbol']} (allowed: {sorted(ALLOWED)})")
    raise SystemExit(0)

side = str(payload.get("side", "SIGNAL")).upper()
sig_tf = payload.get("tf", "?")
entry, sl, tp = num(payload.get("entry")), num(payload.get("sl")), num(payload.get("tp"))
tested = payload.get("tested", "?")
rr = payload.get("rr", "?")
emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "⚪"

lines = [f"{emoji} {side} SIGNAL", f"{sym} (Delta Futures) | Signal TF: {sig_tf}"]
if entry:
    lines.append(f"Entry ~ {entry}")
if sl:
    lines.append(f"SL: {sl}")
if tp:
    lines.append(f"TP: {tp}  (RR 1:{rr})")
lines.append(f"Level tested: {tested}x")
text = "\n".join(lines)
tg("sendMessage", data={"chat_id": CHAT, "text": text})

paths = []
for tf in TFS:
    try:
        df = delta_df(sym, tf)
        if df is None or len(df) < 10:
            print("not enough data for", tf)
            continue
        path = f"chart_{tf}.png"
        make_chart(df, f"{sym}  {tf.upper()}  ({side})", entry, sl, tp, path)
        paths.append(path)
    except Exception as e:
        print("chart failed for", tf, e)

if len(paths) >= 2:
    media = [{"type": "photo", "media": f"attach://f{i}"} for i in range(len(paths))]
    media[0]["caption"] = text[:1000]
    files = {f"f{i}": open(p, "rb") for i, p in enumerate(paths)}
    tg("sendMediaGroup", data={"chat_id": CHAT, "media": json.dumps(media)}, files=files)
elif len(paths) == 1:
    tg("sendPhoto", data={"chat_id": CHAT, "caption": text[:1000]}, files={"photo": open(paths[0], "rb")})
else:
    print("No charts could be generated")
