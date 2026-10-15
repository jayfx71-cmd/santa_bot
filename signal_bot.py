"""
Free automated signal bot (Forex + Deriv synthetic indices) with a learning layer.
Live strategy: London Breakout (break of the overnight range at the London open, with a 200 EMA trend filter)
on EUR/USD, GBP/USD and gold. Research strategies (backtest/sweep only): EMA Cross, Price Action, Mean Reversion.
ATR-based stop loss, two targets (TP1 1:3, TP2 1:4). Runs on a schedule via GitHub Actions,
posts to a Telegram channel, and tracks TP/SL results per strategy in state/signals.json.

Learning layer: every strategy x instrument combination has its own rolling record.
A combination that is clearly losing is paused (no more posts to the channel) but keeps
logging silent "shadow" trades, so it can switch itself back on if it recovers.
Run `python signal_bot.py --backtest` to replay history and seed that learning.
Run `python signal_bot.py --sweep` to compare target sizes and trend filters.
Run `python signal_bot.py --news` to post only the news items (brief, alerts, headlines).
Run `python signal_bot.py --nfp` to test how price reacts after past Non-Farm Payrolls releases.
News filter: normal signals pause on the pairs involved around every high-impact release.
"""
import html
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG (edit these) ----------------
FOREX = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "GBP/JPY", "XAU/USD"]  # XAU/USD = gold

# Deriv synthetic indices are switched OFF (random-number markets: no strategy can have an edge there).
# To bring them back, move the entries from SYNTHETICS_OFF into SYNTHETICS.
SYNTHETICS = {}
SYNTHETICS_OFF = {
    "Volatility 75": "R_75",
    "Volatility 75 (1s)": "1HZ75V",
    "Volatility 25": "R_25",
    "Volatility 25 (1s)": "1HZ25V",
    "Volatility 100": "R_100",
    "Volatility 100 (1s)": "1HZ100V",
    "Volatility 90 (1s)": "1HZ90V",
}

STRATEGIES = ["Daily Break"]  # LIVE: these post to the channel (delete a name to switch it off)
RESEARCH_STRATEGIES = ["London Breakout", "EMA Cross", "Price Action", "Mean Reversion"]  # tested by --backtest/--sweep only; move a
                                                                       # name up into STRATEGIES once it proves itself
STRATEGY_PAIRS = {"Daily Break": ["XAU/USD"],   # the --widen test: only gold held up (t 3.7); the forex pairs were around zero or negative
                   "London Breakout": ["EUR/USD", "GBP/USD", "XAU/USD"]}
LIVE_INTERVAL, LIVE_MINUTES, LIVE_COUNT = "30min", 30, 1200   # live signals use 30-minute candles (about 25 days of history)
M30_ONLY = ["Daily Break"]   # built for 30-minute candles, so the hourly --backtest/--sweep skip it (see --lab)  # live pairs per strategy (tests use all pairs)
LONDON_FILTER = True   # London Breakout only trades with the 200 EMA trend (price on that side AND the EMA sloping that way)
TREND_EMA = 200
COSTS = True           # backtest/sweep subtract the spread from every trade (as a fraction of the risk)
SPREADS = {            # typical spread in price units (round trip is paid once); check your own broker
    "EUR/USD": 0.00010, "GBP/USD": 0.00015, "USD/JPY": 0.012, "AUD/USD": 0.00012,
    "USD/CAD": 0.00018, "GBP/JPY": 0.025, "XAU/USD": 0.35,
}
ALL_STRATEGIES = [x for x in STRATEGIES + RESEARCH_STRATEGIES if x not in M30_ONLY]
STRATEGY_KINDS = {"London Breakout": ("forex",), "Mean Reversion": ("forex",), "Daily Break": ("forex",)}  # others run on everything
STRATEGY_TARGETS = {"London Breakout": (1.0, 2.0), "Mean Reversion": (1.0, 1.5), "Daily Break": (1.0, 2.0), "Vol Break": (1.0, 2.0)}  # (TP1, TP2) in R; others use TP1_R/TP2_R
BB_PERIOD = 20
SESSION_START, SESSION_END = 7, 11   # London Breakout: signals only when the candle opens 07:00-11:00 UTC
ASIA_RANGE_HOURS = 7                 # overnight range = candles opening 00:00-06:00 UTC

CANDLE_SECONDS = 3600  # H1 candles
CANDLE_COUNT = 600     # enough history for the 200 EMA to settle
FAST_EMA, SLOW_EMA = 20, 50
RSI_PERIOD = 14
ATR_PERIOD = 14
SL_ATR_MULT = 1.5      # EMA Cross: stop loss = 1.5 x ATR
TP1_R = 3.0            # first target = 3 x risk (1:3): close half, move stop to break-even
TP2_R = 4.0            # second target = 4 x risk (1:4): the rest of the trade
BRAND = "PROFITS SANTA AI"     # name shown at the top of every signal
CHANNEL_LINK = ""      # e.g. "https://t.me/yourchannel" -- leave empty to hide the link line
EXPIRE_HOURS = 240     # close a trade as "expired" after this long (10 days)

# --- learning layer ---
LEARNING = True        # False = post every signal, never pause anything
WINDOW = 20            # judge each combination on its last 20 finished trades
MIN_TRADES = 8         # need at least this many finished trades before judging
PAUSE_BELOW_R = 0.0    # pause when the window total drops below this (in R)
RESUME_AT_R = 1.0      # switch back on when the window total reaches this (in R)
BACKTEST_MIN = 15      # a backtest needs this many trades before it can pause a combo
BACKTEST_BARS = 5000   # candles to replay (about 7 months of forex H1)
CUT_BELOW_R = -5.0     # REMOVE a combination if its backtest total is this bad (in R) ...
CUT_MIN_TRADES = 10    # ... and the backtest had at least this many trades

SWEEP_TARGETS = [(1.0, 1.5), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (3.0, 5.0)]   # (TP1, TP2) pairs the --sweep test compares
SWEEP_FILTERS = ["none", "ema200", "ema200+slope"]    # trend filters the --sweep test compares
SWEEP_FILE = Path("state/sweep.txt")

DERIV_URLS = [  # tried in order; the first one that answers is used
    "wss://api.derivws.com/trading/v1/options/ws/public",  # current public endpoint
    "wss://ws.derivws.com/websockets/v3?app_id=1089",      # older endpoint (fallback)
]

# --- news posts ---
NEWS = False                   # False = signals only
NEWS_FILE = Path("state/news.json")
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FEEDS = [
    ("FXStreet", "https://www.fxstreet.com/rss/news"),
    ("ForexLive", "https://www.forexlive.com/feed/news"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CNBC Economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
]   # a feed that is down is simply skipped
NEWS_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD"}
BRIEF_HOUR_UTC = 6             # morning brief goes out at the first run after 06:00 UTC (07:00 Nigeria)
ALERT_MINUTES = 90             # warn about high-impact releases up to this long before they happen
BLACKOUT_BEFORE_MIN = 30       # no new signals from this long before a high-impact release ...
BLACKOUT_AFTER_MIN = 90        # ... until this long after it, on the pairs that release affects
NFP_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]   # pairs used by the --nfp test
NFP_PAGES = 7                  # pages of 5000 H1 candles to fetch for the --nfp test (about 5 years)
NFP_FILE = Path("state/nfp.txt")
HEADLINES_PER_RUN = 1       # one headline at a time, never a burst
HEADLINE_GAP_MIN = 180        # and at least this many minutes between two headlines
HEADLINE_MAX_AGE_H = 18      # skip headlines older than this
HEADLINES_PER_DAY = 2          # only two news posts a day: one in the morning, one in the afternoon
HEADLINE_SLOTS = [(7, 12), (14, 19)]   # UTC hours in which the 1st and the 2nd headline may go out
# Signals: at most TWO a day for gold, one per slot (UTC minutes of the day). Nothing stays open overnight.
SIGNAL_SLOTS = {"morning": (420, 660), "evening": (1020, 1110)}   # 08:00-12:00 and 18:00-19:30 Nigeria time
LIVE_END_MIN = 1110
DAY_END_HOUR_UTC = 20          # any trade still open from this hour is closed at the market price (21:00 Nigeria) before the recap
STATE_FILE = Path("state/signals.json")
BACKTEST_FILE = Path("state/backtest.json")
# -----------------------------------------------------

TD_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

INSTRUMENTS = [{"name": p, "kind": "forex", "symbol": p} for p in FOREX] + [
    {"name": n, "kind": "synthetic", "symbol": s} for n, s in SYNTHETICS.items()
]


# ---------- Telegram ----------
def send(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
        timeout=30,
    )
    r.raise_for_status()


# ---------- State ----------
def load_state():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    else:
        state = {"open": [], "closed": [], "last_signal": {}}
    state.setdefault("paused", {})
    for t in state["open"] + state["closed"]:
        t.setdefault("strategy", "EMA Cross")  # trades saved by the first version
        t.setdefault("live", True)             # False = silent shadow trade
    return state


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_backtest():
    try:
        return json.loads(BACKTEST_FILE.read_text())
    except Exception:
        return {}


def combo_key(strategy, pair):
    return f"{strategy}|{pair}"


# ---------- Data ----------
def only_closed(df, seconds=None):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return df[df["datetime"] + timedelta(seconds=seconds or CANDLE_SECONDS) <= now].reset_index(drop=True)


def fetch_twelvedata(symbol, count=CANDLE_COUNT, interval="1h", seconds=None):
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": count,
            "timezone": "UTC",
            "apikey": TD_KEY,
        },
        timeout=60,
    )
    data = r.json()
    if "values" not in data:
        raise RuntimeError(data.get("message", str(data)))
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return only_closed(df, seconds)


def deriv_request(url, symbol, count):
    import websocket  # pip package: websocket-client

    ws = websocket.create_connection(url, timeout=30)
    try:
        ws.send(
            json.dumps(
                {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "granularity": CANDLE_SECONDS,
                    "style": "candles",
                }
            )
        )
        data = {}
        for _ in range(5):  # skip any unrelated messages
            data = json.loads(ws.recv())
            if "candles" in data or "error" in data:
                break
    finally:
        ws.close()
    if "error" in data:
        raise RuntimeError(str(data["error"].get("message", data["error"])))
    if "candles" not in data:
        raise RuntimeError("no candles returned")
    return data


def fetch_deriv(symbol, count=CANDLE_COUNT):
    data, errors = None, []
    for url in DERIV_URLS:
        try:
            data = deriv_request(url, symbol, count)
            break
        except Exception as e:
            host = url.split("/")[2]
            errors.append(f"{host}: {str(e)[:70]}")
    if data is None:
        raise RuntimeError(" | ".join(errors))
    df = pd.DataFrame(data["candles"])
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return only_closed(df)


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=FAST_EMA, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=SLOW_EMA, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    df["ema200"] = df["close"].ewm(span=TREND_EMA, adjust=False).mean()
    df["ema200_lag"] = df["ema200"].shift(10)

    mid = df["close"].rolling(BB_PERIOD).mean()
    sd = df["close"].rolling(BB_PERIOD).std(ddof=0)
    df["bb_up"], df["bb_low"] = mid + 2 * sd, mid - 2 * sd
    return df


# ---------- Helpers ----------
def decimals(inst):
    if inst["kind"] == "synthetic":
        return 4
    if "XAU" in inst["name"]:
        return 2
    return 3 if "JPY" in inst["name"] else 5


def unit_size(inst):
    if inst["kind"] == "synthetic":
        return 1.0
    if "XAU" in inst["name"]:
        return 0.1   # gold: 1 pip = 10 cents
    return 0.01 if "JPY" in inst["name"] else 0.0001


def unit_label(inst):
    return "pts" if inst["kind"] == "synthetic" else "pips"


def strategy_applies(strategy, inst):
    """Can this strategy be tested on this instrument?"""
    return inst["kind"] in STRATEGY_KINDS.get(strategy, ("forex", "synthetic"))


def live_ok(strategy, inst):
    """May this strategy post live signals on this instrument?"""
    return strategy_applies(strategy, inst) and inst["name"] in STRATEGY_PAIRS.get(strategy, [i["name"] for i in INSTRUMENTS])


def cost_r(inst, entry, sl):
    """Spread as a fraction of the amount risked (subtracted from backtest results)."""
    risk = abs(entry - sl)
    return min(2.0, SPREADS.get(inst["name"], 0.0) / risk) if COSTS and risk > 0 else 0.0


def build_signal(inst, strategy, side, entry, sl, candle_time):
    risk = abs(entry - sl)
    sign = 1 if side == "BUY" else -1
    r1, r2 = STRATEGY_TARGETS.get(strategy, (TP1_R, TP2_R))
    tp1 = entry + sign * r1 * risk
    tp2 = entry + sign * r2 * risk
    d = decimals(inst)
    return {
        "pair": inst["name"],
        "strategy": strategy,
        "side": side,
        "entry": round(entry, d),
        "sl": round(sl, d),
        "tp1": round(tp1, d),
        "tp2": round(tp2, d),
        "tp1_r": r1,
        "tp2_r": r2,
        "risk_pips": round(risk / unit_size(inst), 1),
        "unit": unit_label(inst),
        "candle": candle_time.isoformat(),
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "live": True,
    }


# ---------- Strategy 1: EMA cross + RSI ----------
def strategy_ema_cross(inst, df):
    if len(df) < SLOW_EMA + 5:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    crossed_up = prev.ema_fast <= prev.ema_slow and last.ema_fast > last.ema_slow
    crossed_down = prev.ema_fast >= prev.ema_slow and last.ema_fast < last.ema_slow

    entry = float(last.close)
    risk = SL_ATR_MULT * float(last.atr)
    if crossed_up and 50 < last.rsi < 70:
        return build_signal(inst, "EMA Cross", "BUY", entry, entry - risk, last.datetime)
    if crossed_down and 30 < last.rsi < 50:
        return build_signal(inst, "EMA Cross", "SELL", entry, entry + risk, last.datetime)
    return None


# ---------- Strategy 2: Price action (engulfing candle at the 50 EMA) ----------
def strategy_price_action(inst, df):
    if len(df) < SLOW_EMA + 5:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    atr = float(last.atr)
    if not atr > 0:
        return None
    if abs(last.close - last.open) < 0.5 * atr:  # engulfing candle must be strong
        return None

    uptrend = last.ema_fast > last.ema_slow
    downtrend = last.ema_fast < last.ema_slow
    near = 0.25 * atr

    bullish = (
        prev.close < prev.open
        and last.close > last.open
        and last.close >= prev.open
        and last.open <= prev.close
    )
    bearish = (
        prev.close > prev.open
        and last.close < last.open
        and last.close <= prev.open
        and last.open >= prev.close
    )

    entry = float(last.close)
    if uptrend and bullish and last.low <= last.ema_slow + near and last.close > last.ema_slow:
        side = "BUY"
        sl = min(float(prev.low), float(last.low)) - 0.2 * atr
    elif downtrend and bearish and last.high >= last.ema_slow - near and last.close < last.ema_slow:
        side = "SELL"
        sl = max(float(prev.high), float(last.high)) + 0.2 * atr
    else:
        return None

    risk = abs(entry - sl)
    if risk > 3 * atr:  # stop too far away, skip
        return None
    if risk < 0.5 * atr:  # stop too tight, widen it
        sl = entry - 0.5 * atr if side == "BUY" else entry + 0.5 * atr
    return build_signal(inst, "Price Action", side, entry, sl, last.datetime)


# ---------- Strategy 3 (research): London Breakout ----------
def strategy_london_breakout(inst, df):
    """First candle close beyond the overnight (Asian) range once London opens. Stop at mid-range."""
    if len(df) < 60:
        return None
    last = df.iloc[-1]
    t = last.datetime
    if not (SESSION_START <= t.hour <= SESSION_END):
        return None
    atr = float(last.atr)
    if not atr > 0:
        return None
    recent = df.iloc[-40:]
    today = recent[recent["datetime"].dt.date == t.date()]
    asia = today[today["datetime"].dt.hour < SESSION_START]
    if len(asia) < ASIA_RANGE_HOURS:   # incomplete night (data gap)
        return None
    hi, lo = float(asia["high"].max()), float(asia["low"].min())
    if not (0.5 * atr <= hi - lo <= 4 * atr):   # range must be a sensible size
        return None
    earlier = today[(today["datetime"].dt.hour >= SESSION_START) & (today["datetime"] < t)]
    if ((earlier["close"] > hi) | (earlier["close"] < lo)).any():   # already broke out today
        return None

    entry, mid = float(last.close), (hi + lo) / 2
    if entry > hi:
        side, sl = "BUY", mid - 0.1 * atr
    elif entry < lo:
        side, sl = "SELL", mid + 0.1 * atr
    else:
        return None
    if LONDON_FILTER:   # trade only with the long-term trend
        ema, ema_old = float(last.ema200), float(last.ema200_lag)
        if ema_old != ema_old:   # not enough history yet
            return None
        if side == "BUY" and not (entry > ema and ema > ema_old):
            return None
        if side == "SELL" and not (entry < ema and ema < ema_old):
            return None
    risk = abs(entry - sl)
    if risk > 3 * atr:
        return None
    if risk < 0.5 * atr:
        sl = entry - 0.5 * atr if side == "BUY" else entry + 0.5 * atr
    return build_signal(inst, "London Breakout", side, entry, sl, t)


# ---------- Strategy 4 (research): Mean reversion (Bollinger re-entry) ----------
def strategy_mean_reversion(inst, df):
    """Price closed outside the Bollinger band, then closes back inside: bet on a snap back."""
    if len(df) < BB_PERIOD + 5:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    atr = float(last.atr)
    if not atr > 0:
        return None
    entry = float(last.close)
    if prev.close < prev.bb_low and last.close > last.bb_low and last.rsi < 45:
        side, sl = "BUY", min(float(prev.low), float(last.low)) - 0.2 * atr
    elif prev.close > prev.bb_up and last.close < last.bb_up and last.rsi > 55:
        side, sl = "SELL", max(float(prev.high), float(last.high)) + 0.2 * atr
    else:
        return None
    risk = abs(entry - sl)
    if risk > 3 * atr:
        return None
    if risk < 0.5 * atr:
        sl = entry - 0.5 * atr if side == "BUY" else entry + 0.5 * atr
    return build_signal(inst, "Mean Reversion", side, entry, sl, last.datetime)


def signal_slot(candle_iso):
    """'morning' / 'evening' if this candle is inside a signal slot, else None."""
    t = pd.Timestamp(candle_iso)
    m = t.hour * 60 + t.minute
    for name, (a, b) in SIGNAL_SLOTS.items():
        if a <= m <= b:
            return name
    return None


def slot_used(state, slot, candle_iso):
    day = str(candle_iso)[:10]
    return any(t.get("live", True) and t.get("strategy") == "Daily Break" and str(t.get("candle", ""))[:10] == day
               and signal_slot(t["candle"]) == slot for t in state["closed"] + state["open"])


def strategy_daily_break(inst, df):
    """Live (tested as Break Multi): a 30-minute close beyond yesterday's high/low, then up to 2 more with-the-trend breaks of the day high/low (07:00-18:00 UTC)."""
    if len(df) < 450:
        return None
    minutes = max(1, int(round((df["datetime"].iloc[-1] - df["datetime"].iloc[-2]).total_seconds() / 60)))
    d = lab_prepare(df.reset_index(drop=True), minutes)
    cands = lab_daily_break_multi(d, live=True)
    if not cands or cands[-1][0] != d["n"] - 1:
        return None
    i, side, sl = cands[-1]
    return build_signal(inst, "Daily Break", side, float(d["c"][i]), sl, df["datetime"].iloc[-1])


STRATEGY_FUNCS = {
    "Daily Break": strategy_daily_break,
    "EMA Cross": strategy_ema_cross,
    "Price Action": strategy_price_action,
    "London Breakout": strategy_london_breakout,
    "Mean Reversion": strategy_mean_reversion,
}


def format_signal(s, winrate_line):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    arrow = "📈" if s["side"] == "BUY" else "📉"
    synthetic = s.get("unit") == "pts"
    kind = "SYNTHETIC" if synthetic else "FOREX"
    name = s["pair"] + (" Index" if synthetic else "")
    text = (
        f"{icon} <b>{BRAND} VIP SIGNAL</b>  ·  {kind}\n\n"
        f"<b>Pair:</b> {name}\n"
        f"<b>Direction:</b> {arrow} {s['side']}\n"
        f"<b>Strategy:</b> {s['strategy']}\n\n"
        f"<b>Entry:</b> <code>{s['entry']}</code>\n"
        f"<b>Stop Loss:</b> <code>{s['sl']}</code> ({s['risk_pips']} {s.get('unit', 'pips')})\n"
        f"<b>TP1 (1:{s.get('tp1_r', TP1_R):g}):</b> <code>{s['tp1']}</code>\n"
        f"<b>TP2 (1:{s.get('tp2_r', TP2_R):g}):</b> <code>{s['tp2']}</code>\n\n"
        f"<b>Risk:</b> 1% max per trade\n"
        f"{winrate_line}\n\n"
        f"💡 At TP1 close half and move your stop to entry.\n"
        f"⚠️ Trade at your own risk. Manage your position size. Past performance does not guarantee future results."
    )
    if CHANNEL_LINK:
        text += f"\n\n🔗 <a href=\"{CHANNEL_LINK}\">{BRAND.title()} VIP</a>"
    return text


# ---------- Result tracking ----------
SCORED = ("TP", "TP1", "TP2", "SL", "DAYEND")   # "TP" = trades saved by the old single-target version
WINS = ("TP", "TP1", "TP2")


def trade_r(result, tp1_r, tp2_r):
    """Half the trade closes at TP1, the stop then moves to entry, the other half runs to TP2."""
    if result == "SL":
        return -1.0
    if result == "TP1":   # TP1 hit, rest stopped at break-even
        return 0.5 * tp1_r
    if result == "TP2":
        return 0.5 * tp1_r + 0.5 * tp2_r
    return 0.0


def record(trades):
    scored = [t for t in trades if t["result"] in SCORED]
    wins = sum(1 for t in scored if t["result"] in WINS or (t["result"] == "DAYEND" and t.get("r", 0) > 0))
    total_r = sum(t["r"] for t in trades)
    n = len(scored)
    rate = (wins / n * 100) if n else 0
    return wins, n - wins, rate, total_r


def winrate_30d_line(state):
    """Real win rate of the signals posted to the channel in the last 30 days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    recent = [t for t in state["closed"]
              if t.get("live", True) and t["result"] in SCORED and t.get("closed_at", "") >= cutoff]
    w, l, rate, r = record(recent)
    if w + l < 10:
        return f"<b>Win Rate (30d):</b> building record ({w + l} closed so far)"
    return f"<b>Win Rate (30d):</b> {rate:.0f}% ({w}W / {l}L, {r:+.1f}R)"


def stats_line(state, strategy):
    # Public record: only signals that were actually posted to the channel.
    posted = [t for t in state["closed"] if t.get("live", True)]
    w, l, rate, r = record([t for t in posted if t.get("strategy") == strategy])
    aw, al, arate, ar = record(posted)
    return (
        f"📊 {strategy}: {w}W / {l}L ({rate:.0f}%) | {r:+.1f}R\n"
        f"📊 All strategies: {aw}W / {al}L ({arate:.0f}%) | {ar:+.1f}R"
    )


def t_score(values):
    """How many standard errors the average result is away from zero (about 2.5+ is worth noticing)."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean / ((var / n) ** 0.5) if var > 0 else 0.0


def walk_trade(side, entry, sl, tp1, tp2, highs, lows, buf=0.0):
    """Walk candles forward. Returns (result, index, tp1_hit).
    result: "SL", "TP1" (TP1 hit, rest stopped at break-even), "TP2", or None (still open).
    When one candle could have hit both the stop and a target, the stop is counted (conservative).
    buf = the broker spread in price units (live trades only). A broker fills a BUY target at the bid and a SELL target
    at the ask, so a target counts only once price is a full spread beyond it, and stops count half a spread early."""
    buy = side == "BUY"
    half = buf / 2
    stage = 0
    for j in range(len(highs)):
        h, l = highs[j], lows[j]
        if stage == 0:
            if (l <= sl + half) if buy else (h >= sl - half):
                return "SL", j, False
            if (h >= tp1 + buf) if buy else (l <= tp1 - buf):
                stage = 1
                if (h >= tp2 + buf) if buy else (l <= tp2 - buf):
                    return "TP2", j, True
        else:
            if (l <= entry + half) if buy else (h >= entry - half):
                return "TP1", j, True
            if (h >= tp2 + buf) if buy else (l <= tp2 - buf):
                return "TP2", j, True
    return None, None, stage == 1


# ---------- Live progress updates (pips made so far) ----------
PROGRESS_UPDATES = True
PIP_STEPS = [5, 10, 20, 25, 50, 100, 200, 500]


def pip_step(risk_pips):
    """Update every ~quarter of the stop distance, rounded to a tidy number (forex ~5-10 pips, gold ~50)."""
    target = risk_pips / 4
    return next((x for x in PIP_STEPS if x >= target), PIP_STEPS[-1])


def progress_note(t, after, prefix=""):
    """Message when a trade has reached a new profit step, or is halfway to its stop. Marks it in the trade so it is sent once."""
    if not PROGRESS_UPDATES or len(after) == 0 or not t.get("risk_pips"):
        return None
    unit_px = abs(t["entry"] - t["sl"]) / t["risk_pips"]
    if unit_px <= 0:
        return None
    buy = t["side"] == "BUY"
    last = float(after["close"].iloc[-1])
    best = (float(after["high"].max()) - t["entry"]) if buy else (t["entry"] - float(after["low"].min()))
    now_ = (last - t["entry"]) if buy else (t["entry"] - last)
    best_p, now_p = best / unit_px, now_ / unit_px
    step = pip_step(t["risk_pips"])
    unit = t.get("unit", "pips")
    mile = int(best_p // step) * step
    head = f"{prefix}<b>Trade update</b> · {t['side']} {t['pair']}"
    if mile > t.get("milestone", 0) and mile > 0:
        t["milestone"] = mile
        nxt = ("TP1", t["tp1_r"] * t["risk_pips"]) if not t.get("tp1_notified") else ("TP2", t["tp2_r"] * t["risk_pips"])
        left = max(0.0, nxt[1] - now_p)
        return (f"📈 {head}\n+{now_p:.0f} {unit} right now (best so far +{best_p:.0f})\n"
                f"{nxt[0]} is about {left:.0f} {unit} away.")
    if now_p <= -0.5 * t["risk_pips"] and not t.get("warned"):
        t["warned"] = True
        return (f"⚠️ {head}\nPrice is {abs(now_p):.0f} {unit} against us, about half way to the stop loss. "
                f"Keep your stop where it is.")
    return None


def resolve_open(state, candles):
    still_open = []
    now = datetime.now(timezone.utc)
    for t in state["open"]:
        df = candles.get(t["pair"])
        if df is None:
            still_open.append(t)
            continue

        tp1 = t.get("tp1", t.get("tp"))   # old trades only have a single "tp"
        tp2 = t.get("tp2", t.get("tp"))
        a, b = t.get("tp1_r", 4.0), t.get("tp2_r", 4.0)
        after = df[df["datetime"] > pd.Timestamp(t["candle"])]
        result, _, tp1_hit = walk_trade(t["side"], t["entry"], t["sl"], tp1, tp2,
                                        after["high"].values, after["low"].values, buf=SPREADS.get(t["pair"], 0.0))

        if result is None:
            opened = datetime.fromisoformat(t["opened_at"])
            if now - opened > timedelta(hours=EXPIRE_HOURS):
                result = "TP1" if tp1_hit else "EXPIRED"

        if result is None and now.hour >= DAY_END_HOUR_UTC and len(after):
            result = "DAYEND"   # nothing stays open overnight: close at the last price
            last = float(after["close"].iloc[-1])
            sign = 1 if t["side"] == "BUY" else -1
            cur = sign * (last - t["entry"]) / abs(t["entry"] - t["sl"])
            t["close_price"] = round(last, 2)
            t["dayend_r"] = round(0.5 * a + 0.5 * max(cur, 0.0), 2) if tp1_hit else round(max(cur, -1.0), 2)

        if result is None:
            if tp1_hit and not t.get("tp1_notified"):
                t["tp1_notified"] = True
                if t.get("live", True):
                    send(
                        f"🎯 <b>TP1 hit:</b> {t['side']} {t['pair']} ({t['strategy']})\n"
                        f"Close half and move your stop to entry. The rest runs to TP2."
                    )
            elif t.get("live", True):
                note = progress_note(t, after)
                if note:
                    send(note)
            still_open.append(t)
            continue

        t["result"] = result
        t["r"] = t["dayend_r"] if result == "DAYEND" else trade_r(result, a, b)
        t["closed_at"] = now.isoformat()
        state["closed"].append(t)

        if t.get("live", True):  # shadow trades never message the channel
            label = {
                "TP2": "✅ TP2 hit (full target)",
                "TP1": "🟡 TP1 hit, rest stopped at break-even",
                "SL": "❌ SL hit",
                "EXPIRED": "⏱ Expired",
                "DAYEND": "⏰ Closed for the day (nothing stays open overnight)",
            }[result]
            send(
                f"{label}: <b>{t['side']} {t['pair']}</b> ({t['strategy']})\n"
                f"Result: {t['r']:+.1f}R\n\n{stats_line(state, t['strategy'])}"
            )
    state["open"] = still_open


# ---------- Learning layer ----------
def window_trades(state, key):
    """Last WINDOW finished (TP/SL) trades of one combination, live and shadow together."""
    done = [
        t for t in state["closed"]
        if combo_key(t["strategy"], t["pair"]) == key and t["result"] in SCORED
    ]
    done.sort(key=lambda t: t.get("closed_at", ""))
    return done[-WINDOW:]


def cut_combos(backtest):
    """Combinations the backtest showed to be clearly losing: removed completely."""
    if not LEARNING:
        return set()
    return {
        key for key, bt in backtest.items()
        if bt["n"] >= CUT_MIN_TRADES and bt["r"] <= CUT_BELOW_R
    }


def combo_active(state, key, backtest):
    """Decide whether a strategy x instrument combination may post to the channel."""
    if not LEARNING:
        return True
    was_paused = state["paused"].get(key, False)
    recent = window_trades(state, key)
    if len(recent) >= MIN_TRADES:  # enough live/shadow evidence: it decides
        total = sum(t["r"] for t in recent)
        paused = (total < RESUME_AT_R) if was_paused else (total < PAUSE_BELOW_R)
        why = f"last {len(recent)} trades = {total:+.1f}R"
    else:  # not enough evidence yet: lean on the backtest, if there is one
        bt = backtest.get(key)
        paused = bool(bt and bt["n"] >= BACKTEST_MIN and bt["r"] < 0)
        why = f"backtest {bt['n']} trades = {bt['r']:+.1f}R" if bt else "no data yet"
    if paused != was_paused:
        print(f"LEARNING: {key} {'PAUSED' if paused else 'RESUMED'} ({why})")
    state["paused"][key] = paused
    return not paused


def print_leaderboard(state, backtest, cuts=frozenset()):
    keys = set(state["paused"]) | set(backtest)
    keys |= {combo_key(t["strategy"], t["pair"]) for t in state["open"] + state["closed"]}
    names = {i["name"] for i in INSTRUMENTS}
    keys = {k for k in keys if k.split("|")[0] in STRATEGIES and k.split("|", 1)[1] in names}  # live only
    if not keys:
        return
    print("---- LEARNING LEADERBOARD (strategy | pair) ----")
    rows = []
    for key in sorted(keys):
        recent = window_trades(state, key)
        r_recent = sum(t["r"] for t in recent)
        bt = backtest.get(key)
        bt_txt = f"bt {bt['n']}tr {bt['r']:+.1f}R" if bt else "bt -"
        status = "REMOVED" if key in cuts else ("PAUSED" if state["paused"].get(key) else "active")
        rows.append((key, status, f"{len(recent)}tr {r_recent:+.1f}R", bt_txt))
    for key, status, live_txt, bt_txt in rows:
        print(f"{status:8s} {key:40s} recent {live_txt:14s} {bt_txt}")
    print("-------------------------------------------------")


# ---------- Backtest ----------
def simulate(inst, df):
    """Replay history candle by candle: no peeking, one open trade per strategy at a time."""
    highs, lows = df["high"].values, df["low"].values
    expire = max(1, EXPIRE_HOURS * 3600 // CANDLE_SECONDS)
    trades = []
    for strategy in ALL_STRATEGIES:
        if not strategy_applies(strategy, inst):
            continue
        func = STRATEGY_FUNCS[strategy]
        busy_until = -1
        for i in range(SLOW_EMA + 6, len(df) - 1):
            if i <= busy_until:
                continue
            sig = func(inst, df.iloc[: i + 1])
            if not sig:
                continue
            end = min(len(df), i + 1 + expire)
            result, j, tp1_hit = walk_trade(sig["side"], sig["entry"], sig["sl"], sig["tp1"], sig["tp2"],
                                            highs[i + 1:end], lows[i + 1:end])
            if result is None:
                if end - (i + 1) < expire:  # history ended before the trade could finish
                    continue
                result, j = ("TP1" if tp1_hit else "EXPIRED"), expire - 1
            busy_until = i + 1 + j
            r = trade_r(result, sig["tp1_r"], sig["tp2_r"]) - cost_r(inst, sig["entry"], sig["sl"])
            trades.append({"strategy": strategy, "pair": inst["name"], "i": i, "result": result, "r": r})
    return trades


def summarize(trades, split_index):
    out = {}
    for key in sorted({combo_key(t["strategy"], t["pair"]) for t in trades}):
        mine = [t for t in trades if combo_key(t["strategy"], t["pair"]) == key]
        early = [t for t in mine if t["i"] < split_index[key.split("|")[1]]]
        late = [t for t in mine if t["i"] >= split_index[key.split("|")[1]]]
        w, l, rate, r = record(mine)
        out[key] = {
            "n": w + l, "wins": w, "win_rate": round(rate, 1), "r": round(r, 1),
            "first_70_r": round(sum(t["r"] for t in early), 1), "first_70_n": len(early),
            "last_30_r": round(sum(t["r"] for t in late), 1), "last_30_n": len(late),
        }
    return out


def run_backtest():
    all_trades, split_index = [], {}
    for k, inst in enumerate(INSTRUMENTS):
        try:
            if inst["kind"] == "forex":
                if k:
                    time.sleep(8)
                df = fetch_twelvedata(inst["symbol"], BACKTEST_BARS)
            else:
                df = fetch_deriv(inst["symbol"], BACKTEST_BARS)
            df = add_indicators(df)
        except Exception as e:
            print(f"Could not fetch {inst['name']}: {str(e)[:200]}")
            continue
        split_index[inst["name"]] = int(len(df) * 0.7)
        trades = simulate(inst, df)
        all_trades += trades
        print(f"{inst['name']}: {len(df)} candles, {len(trades)} trades simulated")
    if not all_trades:
        print("Backtest produced no trades. Check your API keys and connections.")
        sys.exit(1)

    result = summarize(all_trades, split_index)
    BACKTEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_FILE.write_text(json.dumps(result, indent=2))

    print("\n==== BACKTEST RESULTS (R = multiples of the amount risked) ====")
    print(f"{'combination':38s} {'trades':>6s} {'win%':>5s} {'total R':>8s} {'first70% R':>11s} {'last30% R':>10s}")
    for key, v in sorted(result.items(), key=lambda kv: -kv[1]["r"]):
        print(f"{key:38s} {v['n']:6d} {v['win_rate']:5.0f} {v['r']:+8.1f} {v['first_70_r']:+11.1f} {v['last_30_r']:+10.1f}")
    for strategy in ALL_STRATEGIES:
        mine = [t for t in all_trades if t["strategy"] == strategy]
        w, l, rate, r = record(mine)
        print(f"ALL {strategy}{'' if strategy in STRATEGIES else ' (research)'}: {w + l} trades, {rate:.0f}% win rate, {r:+.1f}R")
    print("Tip: a combination that wins in 'first70%' but loses in 'last30%' is probably luck.")


# ---------- Sweep: compare R:R values and trend filters ----------
def trend_ok(filt, side, close, ema, ema_old):
    """Only allow trades that go with the long-term (200 EMA) trend."""
    if filt == "none":
        return True
    up, down = close > ema, close < ema
    if filt == "ema200+slope":  # the 200 EMA itself must also be sloping the same way
        up, down = up and ema > ema_old, down and ema < ema_old
    return up if side == "BUY" else down


def run_sweep():
    data = []
    for k, inst in enumerate(INSTRUMENTS):
        try:
            if inst["kind"] == "forex":
                if k:
                    time.sleep(8)
                df = fetch_twelvedata(inst["symbol"], BACKTEST_BARS)
            else:
                df = fetch_deriv(inst["symbol"], BACKTEST_BARS)
            data.append((inst, add_indicators(df).reset_index(drop=True)))
        except Exception as e:
            print(f"Could not fetch {inst['name']}: {str(e)[:200]}")
    if not data:
        print("Sweep found no data. Check your API keys and connections.")
        sys.exit(1)

    global LONDON_FILTER
    LONDON_FILTER = False   # the sweep applies the trend filters itself, so compare them fairly
    expire = max(1, EXPIRE_HOURS * 3600 // CANDLE_SECONDS)
    rows = {}  # (kind, strategy, filter, (tp1, tp2)) -> list of (is_early, r, won)
    for inst, df in data:
        highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
        ema = df["close"].ewm(span=TREND_EMA, adjust=False).mean().values
        split = int(len(df) * 0.7)
        start = TREND_EMA + 10
        for strategy in ALL_STRATEGIES:
            if not strategy_applies(strategy, inst):
                continue
            func = STRATEGY_FUNCS[strategy]
            cands = []
            for i in range(start, len(df) - 1):
                sig = func(inst, df.iloc[: i + 1])
                if sig:
                    cands.append((i, sig))
            for filt in SWEEP_FILTERS:
                kept = [(i, sg) for i, sg in cands
                        if trend_ok(filt, sg["side"], closes[i], ema[i], ema[i - 10])]
                for tgt in SWEEP_TARGETS:
                    a, b = tgt
                    out = rows.setdefault((inst["kind"], strategy, filt, tgt), [])
                    busy = -1
                    for i, sg in kept:
                        if i <= busy:
                            continue
                        risk = abs(sg["entry"] - sg["sl"])
                        if risk <= 0:
                            continue
                        sign = 1 if sg["side"] == "BUY" else -1
                        tp1, tp2 = sg["entry"] + sign * a * risk, sg["entry"] + sign * b * risk
                        end = min(len(df), i + 1 + expire)
                        result, j, tp1_hit = walk_trade(sg["side"], sg["entry"], sg["sl"], tp1, tp2,
                                                        highs[i + 1:end], lows[i + 1:end])
                        if result is None:
                            if end - (i + 1) < expire:
                                continue
                            result, j = ("TP1" if tp1_hit else "EXPIRED"), expire - 1
                        busy = i + 1 + j
                        r = trade_r(result, a, b) - cost_r(inst, sg["entry"], sg["sl"])
                        out.append((i < split, r, r > 0))
        print(f"{inst['name']}: swept {len(df)} candles")

    lines = ["==== SWEEP (R = multiples of the amount risked" + (", AFTER typical spreads) ====" if COSTS else ") ====")]
    lines.append("Half closes at TP1, stop moves to entry, rest runs to TP2. Each row pools every instrument of that type.")
    lines.append("'t' says how real a result is (under 2 = could easily be luck). 'good' needs t of 3+, positive in BOTH the")
    lines.append("early 70% and the late 30% of history, and 30+ trades.")
    for kind in ("forex", "synthetic"):
        for strategy in ALL_STRATEGIES:
            if kind not in STRATEGY_KINDS.get(strategy, ("forex", "synthetic")):
                continue
            if not any(k[0] == kind and k[1] == strategy and v for k, v in rows.items()):
                continue
            lines.append("")
            lines.append(f"-- {kind.upper()} / {strategy}{'' if strategy in STRATEGIES else ' (research)'} --")
            lines.append(f"{'filter':13s} {'TP1/TP2':>8s} {'trades':>6s} {'win%':>5s} {'total R':>8s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}")
            for filt in SWEEP_FILTERS:
                for tgt in SWEEP_TARGETS:
                    t = rows.get((kind, strategy, filt, tgt), [])
                    if not t:
                        continue
                    n = len(t)
                    wins = sum(1 for x in t if x[2])
                    total = sum(x[1] for x in t)
                    early = sum(x[1] for x in t if x[0])
                    late = sum(x[1] for x in t if not x[0])
                    ts = t_score([x[1] for x in t])
                    good = "  good" if early > 0 and late > 0 and n >= 30 and ts >= 3 else ""
                    lines.append(
                        f"{filt:13s} {tgt[0]:g}/{tgt[1]:g}".ljust(22)
                        + f" {n:6d} {100 * wins / n:5.0f} {total:+8.1f} {total / n:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{good}"
                    )
    lines.append("")
    lines.append(("Typical spreads ARE subtracted; slippage is not. " if COSTS else "Spreads and slippage are NOT included. ")
                 + "Many variants were tried, so a lucky winner is likely.")
    lines.append("Trust only rows marked good, and only when neighbouring rows look similar.")
    text = "\n".join(lines)
    print(text)
    SWEEP_FILE.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_FILE.write_text(text + "\n")


# ---------- NFP research: how does price behave after Non-Farm Payrolls? ----------
def fetch_twelvedata_pages(symbol, pages):
    """Fetch several pages of 5000 H1 candles going further back in time (slow: obeys the rate limit)."""
    frames, end = [], None
    for page in range(pages):
        params = {"symbol": symbol, "interval": "1h", "outputsize": 5000, "timezone": "UTC", "apikey": TD_KEY}
        if end:
            params["end_date"] = end
        data = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=90).json()
        if "values" not in data:
            if page == 0:
                raise RuntimeError(data.get("message", str(data)))
            break
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        frames.append(df)
        if len(df) < 100:
            break
        end = df["datetime"].min().strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(8)
    df = pd.concat(frames).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return only_closed(df)


def nfp_release_times(first, last):
    """NFP comes out at 08:30 New York time on the 3rd Friday after the week containing the 12th,
    moved around holidays (Jan 1-3 -> a week later, July 4th weekend -> the Thursday before)."""
    ny, out = ZoneInfo("America/New_York"), []
    year, month = first.year, first.month - 1      # start one month early: release comes the month after
    if month == 0:
        year, month = year - 1, 12
    while (year, month) <= (last.year, last.month):
        twelfth = datetime(year, month, 12)
        ref_saturday = twelfth + timedelta(days=(5 - twelfth.weekday()) % 7)
        day = ref_saturday + timedelta(days=20)
        if day.month == 1 and day.day <= 3:
            day += timedelta(days=7)
        elif day.month == 7 and day.day in (3, 4):   # Jul 4 on a Friday, or the Friday off when Jul 4 is a Saturday
            day -= timedelta(days=1)
        release = datetime(day.year, day.month, day.day, 8, 30, tzinfo=ny).astimezone(timezone.utc)
        if first <= release.replace(tzinfo=None) <= last:
            out.append(release)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def run_nfp():
    NFP_TARGETS = [(1.0, 1.5), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    horizon = 24                       # follow each trade for up to 24 hours
    trades = {}                        # (rule, target) -> list of (event_order, r, win)
    used, per_pair = 0, {}
    for k, name in enumerate(NFP_PAIRS):
        try:
            if k:
                time.sleep(8)
            df = add_indicators(fetch_twelvedata_pages(name, NFP_PAGES)).reset_index(drop=True)
        except Exception as e:
            print(f"Could not fetch {name}: {str(e)[:200]}")
            continue
        highs, lows = df["high"].values, df["low"].values
        times = list(df["datetime"])
        events = nfp_release_times(df["datetime"].iloc[0].to_pydatetime(), df["datetime"].iloc[-1].to_pydatetime())
        count = 0
        for n, rel in enumerate(events):
            hour = pd.Timestamp(rel.replace(minute=0, second=0, tzinfo=None))
            idx = df.index[df["datetime"] == hour]
            if len(idx) == 0:
                continue
            i = int(idx[0])
            if i < 30 or i + 1 + horizon >= len(df):
                continue
            atr = float(df["atr"].iloc[i - 1])
            body = float(df["close"].iloc[i] - df["open"].iloc[i])
            if not atr > 0 or abs(body) < 0.5 * atr:      # the first hour must show a clear move
                continue
            move = 1 if body > 0 else -1
            entry = float(df["close"].iloc[i])
            risk = 1.5 * atr
            count += 1
            for rule, direction in (("follow the move", move), ("fade the move", -move)):
                side = "BUY" if direction > 0 else "SELL"
                for a, b in NFP_TARGETS:
                    sl = entry - direction * risk
                    tp1, tp2 = entry + direction * a * risk, entry + direction * b * risk
                    result, j, tp1_hit = walk_trade(side, entry, sl, tp1, tp2,
                                                    highs[i + 1:i + 1 + horizon], lows[i + 1:i + 1 + horizon])
                    if result is None:   # still open after 24h: close at the market
                        last_close = float(df["close"].iloc[i + horizon])
                        r = max(-1.0, min(b, direction * (last_close - entry) / risk))
                    else:
                        r = trade_r(result, a, b)
                    trades.setdefault((rule, (a, b)), []).append((rel, r, r > 0))
        per_pair[name] = (len(events), count)
        used += count
        print(f"{name}: {len(df)} candles, {count} usable NFP releases")
    if not trades:
        print("NFP test found no data.")
        sys.exit(1)

    lines = ["==== NFP TEST (R = multiples of the amount risked) ====",
             "After each NFP the bot looks at the first hour's move (must be at least half an ATR) and either",
             "FOLLOWS it or FADES it. Stop = 1.5 ATR. Half closes at TP1, stop to entry, rest runs to TP2.",
             f"Pairs: {', '.join(NFP_PAIRS)}. Usable releases per pair: "
             + ", ".join(f"{p} {c}" for p, (_, c) in per_pair.items()),
             "'t' says how real the result is (under 2 = could easily be luck). 'good' needs t of 2.5+, positive in BOTH the",
             "earlier 70% and later 30% of releases, and 30+ releases.", "",
             f"{'rule':16s} {'TP1/TP2':>8s} {'trades':>6s} {'win%':>5s} {'total R':>8s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
    for rule in ("follow the move", "fade the move"):
        for tgt in NFP_TARGETS:
            t = sorted(trades.get((rule, tgt), []), key=lambda x: x[0])
            if not t:
                continue
            cut = int(len(t) * 0.7)
            early, late = sum(x[1] for x in t[:cut]), sum(x[1] for x in t[cut:])
            total, n = sum(x[1] for x in t), len(t)
            wins = sum(1 for x in t if x[2])
            by_event = {}
            for x in t:                       # same-day trades on different pairs move together: count each release once
                by_event.setdefault(x[0], []).append(x[1])
            ts = t_score([sum(v) / len(v) for v in by_event.values()])
            good = "  good" if early > 0 and late > 0 and len(by_event) >= 30 and ts >= 2.5 else ""
            lines.append(f"{rule:16s} {tgt[0]:g}/{tgt[1]:g}".ljust(25)
                         + f" {n:6d} {100 * wins / n:5.0f} {total:+8.1f} {total / n:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{good}")
    lines += ["", "Spreads and slippage are NOT included, and they are large around NFP. Only a clear, consistent 'good'",
              "result (with neighbouring rows looking similar) would justify trading NFP."]
    text = "\n".join(lines)
    print(text)
    NFP_FILE.parent.mkdir(parents=True, exist_ok=True)
    NFP_FILE.write_text(text + "\n")


# ---------- News posts ----------
FLAGS = {"USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵", "AUD": "🇦🇺", "CAD": "🇨🇦"}
WAT = timezone(timedelta(hours=1))   # Nigeria time
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; pippulse-bot)"}


def load_news():
    try:
        news = json.loads(NEWS_FILE.read_text())
    except Exception:
        news = {}
    news.setdefault("brief_date", "")
    news.setdefault("alerted", [])
    news.setdefault("seen", [])
    news.setdefault("day", "")
    news.setdefault("day_count", 0)
    news.setdefault("lesson_date", "")
    news.setdefault("lesson_idx", 0)
    news.setdefault("levels_date", "")
    news.setdefault("recap_date", "")
    news.setdefault("last_headline", "")
    return news


def save_news(news):
    NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    news["alerted"] = news["alerted"][-200:]
    news["seen"] = news["seen"][-500:]
    NEWS_FILE.write_text(json.dumps(news, indent=2))


def fetch_calendar(impacts=("High",)):
    """Releases for the major currencies this week (high impact unless told otherwise), times in UTC."""
    r = requests.get(CALENDAR_URL, headers=HTTP_HEADERS, timeout=20)
    r.raise_for_status()
    events = []
    for e in r.json():
        if e.get("impact") not in impacts or e.get("country") not in NEWS_CURRENCIES:
            continue
        when = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        events.append({
            "id": f"{e['date']}|{e['country']}|{e['title']}",
            "time": when, "ccy": e["country"], "title": e["title"],
            "forecast": e.get("forecast") or "", "previous": e.get("previous") or "",
        })
    return sorted(events, key=lambda x: x["time"])


CURRENT_EVENTS = []   # high-impact releases this week; filled by run_news, used to pause signals


def pair_currencies(inst):
    return set(inst["name"].split("/"))


def blackout_event(inst, now):
    """The high-impact release (if any) that currently makes it unwise to signal this pair."""
    ccys = pair_currencies(inst)
    for e in CURRENT_EVENTS:
        if e["ccy"] in ccys and (
            e["time"] - timedelta(minutes=BLACKOUT_BEFORE_MIN) <= now <= e["time"] + timedelta(minutes=BLACKOUT_AFTER_MIN)
        ):
            return e
    return None


def is_nfp(e):
    return "non-farm" in e["title"].lower()


def affected_pairs(e):
    return [i["name"] for i in INSTRUMENTS if e["ccy"] in pair_currencies(i)]


def event_line(e):
    extra = []
    if e["forecast"]:
        extra.append(f"forecast {e['forecast']}")
    if e["previous"]:
        extra.append(f"previous {e['previous']}")
    tail = f" ({', '.join(extra)})" if extra else ""
    flag = FLAGS.get(e["ccy"], "")
    return f"• {e['time'].astimezone(WAT):%H:%M} {flag} {e['ccy']} {html.escape(e['title'])}{html.escape(tail)}"


# ---------- Plain-English explanations ----------
import re

EVENT_HELP = [  # (pattern in the event title, what it is, how price usually reacts) - first match wins
    (r"non-farm|nfp", "US jobs report: how many new jobs America added last month.",
     "More jobs than expected usually lifts the dollar. Fewer jobs does the opposite."),
    (r"adp", "A private jobs count that hints at what the official US jobs report will say.",
     "More jobs than expected usually helps the dollar."),
    (r"unemployment claims|jobless claims|claimant", "How many people applied for unemployment help. A HIGHER number is bad news for the economy.",
     "Higher than expected usually weakens the currency."),
    (r"unemployment rate", "The share of people without a job. LOWER is better.",
     "Lower than expected usually strengthens the currency."),
    (r"employment change|jobs", "How many jobs the economy added or lost.",
     "More jobs than expected usually strengthens the currency."),
    (r"average hourly earnings|wage", "How fast pay is rising. Faster pay can keep inflation high.",
     "Higher than expected can strengthen the currency because rates may stay high."),
    (r"core pce|pce", "The inflation number the US Federal Reserve watches most closely.",
     "Hotter inflation than expected usually lifts the dollar (rates stay high). Cooler inflation weakens it."),
    (r"cpi|inflation", "Inflation: how fast prices are rising. Central banks raise rates to fight it.",
     "Higher than expected usually lifts the currency (rates may stay high). Lower usually weakens it."),
    (r"ppi", "What producers pay for goods. It often shows up in shop prices later.",
     "Higher than expected can lift the currency, like inflation."),
    (r"fomc.*(statement|rate)|federal funds|fed.*rate", "The US central bank (the Fed) announces its interest rate.",
     "Higher rates or a firm tone usually lift the dollar. Cuts or a soft tone weaken it. The wording matters as much as the number."),
    (r"fomc.*minutes|minutes", "Notes from the last central bank meeting that show what they were thinking.",
     "Markets look for hints about future rate moves."),
    (r"press conference|speaks|speech|testif", "The central bank boss is talking. Traders hunt for hints about interest rates.",
     "A firm tone usually lifts the currency. A soft tone weakens it. Sudden swings are common."),
    (r"rate statement|rate decision|official bank rate|cash rate|main refinancing|policy rate|overnight rate|boj|boe|ecb|rba|boc", "A central bank announces its interest rate.",
     "Higher rates or a firm tone usually lift the currency. Cuts or a soft tone weaken it. A surprise moves price the most."),
    (r"gdp", "How fast the whole economy is growing.",
     "Stronger than expected usually lifts the currency."),
    (r"retail sales", "How much shoppers are spending. Spending drives most economies.",
     "Higher than expected usually strengthens the currency."),
    (r"pmi|ism", "A survey of business managers. Above 50 means the sector is growing.",
     "Higher than expected usually strengthens the currency."),
    (r"consumer (sentiment|confidence)|uom|zew|sentiment", "A survey of how confident people feel about the economy.",
     "Better than expected usually helps the currency."),
    (r"trade balance", "Exports minus imports. More exports than imports is good for a currency.",
     "Better than expected usually helps the currency."),
    (r"jolts|job openings", "How many jobs employers are trying to fill.",
     "Higher than expected usually helps the dollar."),
]

HEADLINE_HELP = [  # (words in a headline, one plain line about why it can matter)
    (r"\b(fed|powell|fomc)\b", "The Fed sets US interest rates. Higher-for-longer usually lifts the dollar. Cuts usually weaken it."),
    (r"\b(ecb|lagarde)\b", "The ECB sets euro rates. Higher rates usually lift the euro."),
    (r"\b(boe|bank of england|bailey)\b", "The Bank of England sets UK rates. Higher rates usually lift the pound."),
    (r"\b(boj|bank of japan|yen)\b", "Japan keeps rates very low, so the yen moves on any hint of change. Nervous markets often buy yen."),
    (r"\b(rba|reserve bank of australia)\b", "The RBA sets Australian rates. Higher rates usually lift the Australian dollar."),
    (r"\b(inflation|cpi|pce)\b", "Hot inflation can keep interest rates high, which usually lifts that currency."),
    (r"\b(payrolls|jobs report|unemployment|jobless)\b", "Strong jobs data usually lifts a currency. Weak jobs data usually weakens it."),
    (r"\b(gold|bullion)\b", "Gold often rises when the dollar falls or when investors feel nervous."),
    (r"\b(oil|crude|opec|brent)\b", "Oil moves inflation and currencies like the Canadian dollar, which sells a lot of oil."),
    (r"\b(tariff|trade war|sanction)\b", "Trade fights can rattle markets. Safe havens like the dollar, yen and gold often rise."),
    (r"\b(war|attack|missile|geopolit|conflict)\b", "Fear usually sends money into safe havens like the dollar, yen and gold."),
    (r"\b(dollar|dxy|greenback)\b", "Gold and the euro usually move the opposite way to the dollar."),
    (r"\b(rate cut|rate hike|interest rate|rates)\b", "Higher interest rates usually attract money into a currency. Lower rates usually push it out."),
]


def explain_event(e):
    title = e["title"].lower()
    for pattern, what, effect in EVENT_HELP:
        if re.search(pattern, title):
            gold = " Gold usually moves the opposite way to the dollar." if e["ccy"] == "USD" else ""
            return f"   <i>{html.escape(what)} {html.escape(effect)}{gold}</i>"
    return ""


def explain_headline(title):
    t = title.lower()
    for pattern, line in HEADLINE_HELP:
        if re.search(pattern, t):
            return f"\n💡 <i>In plain words: {html.escape(line)}</i>"
    return ""


def fetch_headlines():
    """Newest headlines from the free RSS feeds. Headline + link only."""
    per_feed = []
    for name, url in FEEDS:
        items = []
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            r.raise_for_status()
            for item in ET.fromstring(r.content).iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if title and link:
                    ts = None
                    try:
                        from email.utils import parsedate_to_datetime
                        ts = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
                    except Exception:
                        pass
                    items.append({"source": name, "title": title, "link": link, "ts": ts})
        except Exception as e:
            print(f"News feed {name} unavailable: {str(e)[:100]}")
        per_feed.append(items[:15])
    mixed = []  # interleave so one feed does not dominate
    for k in range(15):
        for items in per_feed:
            if k < len(items):
                mixed.append(items[k])
    return mixed


def run_news(force=False, repost=False):
    news = load_news()
    now = datetime.now(timezone.utc)
    local = now.astimezone(WAT)          # dates shown to subscribers use Nigeria time
    today = local.strftime("%Y-%m-%d")

    # 1) calendar: morning brief + "big news soon" warnings
    try:
        events = fetch_calendar()
    except Exception as e:
        events = None
        print(f"Calendar unavailable: {str(e)[:100]}")
    if events is not None:
        CURRENT_EVENTS[:] = events
        todays = [e for e in events if e["time"].astimezone(WAT).strftime("%Y-%m-%d") == today]
        brief_due = force or (now.weekday() < 5 and BRIEF_HOUR_UTC <= now.hour < 20)
        try:   # medium-impact releases: listed for information, they never pause signals
            todays_medium = [e for e in fetch_calendar(("Medium",)) if e["time"].astimezone(WAT).strftime("%Y-%m-%d") == today]
        except Exception:
            todays_medium = []
        if brief_due and (repost or news["brief_date"] != today):
            lines = [f"🗞 <b>{BRAND} MORNING BRIEF</b> · {local:%a %d %b}", ""]
            if todays:
                lines.append("High-impact releases today (Nigeria time):")
                for e in todays:
                    lines.append(event_line(e))
                    ex = explain_event(e)
                    if ex:
                        lines.append(ex)
                lines += ["", "📘 <b>How to read this:</b> 'forecast' is what experts expect. Price moves on the SURPRISE: "
                          "the further the real number is from the forecast, the bigger the move.",
                          "", f"Prices can jump around these. We pause new signals on the pairs involved "
                          f"from {BLACKOUT_BEFORE_MIN} min before to {BLACKOUT_AFTER_MIN} min after each release."]
            else:
                lines.append("No high-impact releases on the major currencies today. A quieter day.")
            if todays_medium:
                lines += ["", "Also today (smaller movers):"] + [event_line(e) for e in todays_medium[:6]]
            send("\n".join(lines))
            news["brief_date"] = today
        for e in events:
            mins = (e["time"] - now).total_seconds() / 60
            if 0 < mins <= ALERT_MINUTES and (repost or e["id"] not in news["alerted"]):
                until = (e["time"] + timedelta(minutes=BLACKOUT_AFTER_MIN)).astimezone(WAT)
                pairs = ", ".join(affected_pairs(e)) or e["ccy"]
                extra = ("NFP is the biggest mover of the month, so expect wild swings. "
                         if is_nfp(e) else "")
                send(f"⚠️ <b>High-impact news in about {int(mins)} min</b>\n{event_line(e)}\n{explain_event(e)}\n\n"
                     f"{extra}Spreads widen and price can jump, so we are NOT trading it. "
                     f"New signals on {pairs} are paused until {until:%H:%M} (Nigeria time).")
                news["alerted"].append(e["id"])

    # 1b) lesson of the day: a little fundamentals for newer traders
    if repost or ((force or now.weekday() < 5) and 11 <= now.hour < 20 and news["lesson_date"] != today):
        title, body = LESSONS[news["lesson_idx"] % len(LESSONS)]
        send(f"🎓 <b>Lesson of the day:</b> {html.escape(title)}\n\n{html.escape(body)}")
        news["lesson_date"] = today
        if not repost:
            news["lesson_idx"] += 1

    # 2) headlines (daytime only, capped so the channel is not spammed)
    if news["day"] != today:
        news["day"], news["day_count"] = today, 0
    slot_ok = force or (news["day_count"] < len(HEADLINE_SLOTS) and
                        HEADLINE_SLOTS[news["day_count"]][0] <= now.hour < HEADLINE_SLOTS[news["day_count"]][1])
    if slot_ok and news["day_count"] < HEADLINES_PER_DAY or force:
        items = fetch_headlines()
        first_run = not news["seen"]
        fresh = [h for h in items if repost or h["link"] not in news["seen"]]
        fresh = [h for h in fresh if not h.get("ts") or (now - h["ts"]).total_seconds() < HEADLINE_MAX_AGE_H * 3600]
        try:
            last = datetime.fromisoformat(news["last_headline"]) if news["last_headline"] else None
        except Exception:
            last = None
        waited_enough = repost or last is None or (now - last).total_seconds() >= HEADLINE_GAP_MIN * 60
        posted = 0
        for h in fresh:
            if not waited_enough or posted >= HEADLINES_PER_RUN or (news["day_count"] >= HEADLINES_PER_DAY and not repost):
                break
            send(f"📰 <b>Market Pulse</b> ({h['source']})\n{html.escape(h['title'])}{explain_headline(h['title'])}\n"
                 f"<a href=\"{html.escape(h['link'], quote=True)}\">Read more →</a>")
            posted += 1
            news["day_count"] += 1
            news["last_headline"] = now.isoformat()
            news["seen"].append(h["link"])
        if first_run:   # a brand new install: do not drip out the old backlog later
            news["seen"] += [h["link"] for h in fresh]
            print(f"News: first run, posted {posted} headline(s) and marked the rest as seen.")
    save_news(news)


# ---------- Extra daily content: lesson, levels to watch, day recap ----------
LESSONS = [
    ("What is a pip?", "A pip is the standard small step in a price. On EUR/USD it is 0.0001. If EUR/USD moves from 1.1000 to 1.1050, that is 50 pips. Your stop loss and targets are measured in pips."),
    ("What is a stop loss?", "A stop loss is an order that closes your trade at a set price if the market goes against you. It caps your loss before you even enter. Trading without one is how accounts blow up."),
    ("What does 1:2 mean?", "1:2 (risk to reward) means you risk 1 to try to make 2. With a 1:2 setup you can be wrong more often than right and still make money. At 1:2, winning just 40% of trades is enough to come out ahead."),
    ("Why risk only 1% per trade?", "If you risk 1% per trade, ten losses in a row cost about 10% of your account. If you risk 10%, ten losses wipe you out. Small risk keeps you in the game long enough for your edge to show."),
    ("What is the spread?", "The spread is the small gap between the buy price and the sell price. It is a cost you pay the moment you enter. It gets wider around big news, so we avoid trading right through it."),
    ("What is leverage?", "Leverage lets you control a bigger position than your cash. It magnifies wins AND losses. High leverage does not make you richer faster. It makes mistakes cost more."),
    ("What is a trend?", "An uptrend is price making higher highs and higher lows. A downtrend is the reverse. Trading with the trend is usually easier than fighting it, which is why our strategy only takes breakouts that agree with the trend."),
    ("What is a breakout?", "A breakout is when price closes beyond a level it struggled to pass, like yesterday's high. Sometimes it keeps running. Sometimes it fails and snaps back. That is why every trade needs a stop loss."),
    ("What is break-even?", "When TP1 is hit, we close half the trade and move the stop to the entry price. The remaining half can no longer lose money, and it can still run to TP2."),
    ("Why do losses happen?", "Every strategy has losing trades, even good ones. What matters is the result over many trades, not one trade. Judge a strategy on at least 100 trades, never on 3."),
    ("Why we skip big news", "Around big releases like NFP, prices jump in both directions and spreads widen. Stops can be hit by a spike that reverses seconds later. Skipping those minutes avoids a lot of random losses."),
    ("What is a safe haven?", "When investors are scared, money often flows into the US dollar, the Japanese yen, the Swiss franc and gold. That is why those can rise on bad news elsewhere."),
    ("Why does gold move opposite the dollar?", "Gold is priced in dollars. When the dollar gets stronger, gold costs more for everyone else, so demand and price often fall. The link is common but not perfect."),
    ("What are interest rates doing to currencies?", "Money tends to flow where it earns more interest. Higher rates usually strengthen a currency. Cuts usually weaken it. This is why central bank days are so important."),
]


def run_extras(candles, state, now):
    """Levels to watch (morning) and a day recap (evening). Real data only: nothing here is invented."""
    news = load_news()
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5:
        return
    changed = False

    # 1) levels to watch: the exact lines the Daily Break strategy is waiting for
    if 7 <= now.hour < 18 and news["levels_date"] != today:
        blocks, panels = [], []
        for name in STRATEGY_PAIRS.get("Daily Break", []):
            df = candles.get(name)
            inst = next((i for i in INSTRUMENTS if i["name"] == name), None)
            if df is None or inst is None or len(df) < 450:
                continue
            minutes = max(1, int(round((df["datetime"].iloc[-1] - df["datetime"].iloc[-2]).total_seconds() / 60)))
            d = lab_prepare(df.reset_index(drop=True), minutes)
            groups = sorted(lab_day_groups(d).items())
            if len(groups) < 2 or str(groups[-1][0]) != today:
                continue
            prev, cur = groups[-2][1], groups[-1][1]
            hi, lo, dec = float(d["h"][prev].max()), float(d["l"][prev].min()), decimals(inst)
            i = d["n"] - 1
            up, down = lab_trend(d, i, "BUY"), lab_trend(d, i, "SELL")
            broke = any(7 * 60 <= d["mod"][j] <= 16 * 60 and (d["c"][j] > hi or d["c"][j] < lo) for j in cur)
            if broke:
                note = "Price already closed beyond yesterday's range. Now watching for a new high or low of the day, with the trend."
                short, pstate, watch = "already broke yesterday's range: watching new highs/lows", "broke", "Watching for a new high or low of the day"
            elif up:
                note = f"📈 Trend up. Watching a 30-min close ABOVE <code>{hi:.{dec}f}</code> for a BUY."
                short, pstate = f"trend UP: watching a close above {hi:.{dec}f} (BUY)", "up"
                watch = f"Waiting for a 30-min close above {hi:.{dec}f}  →  BUY"
            elif down:
                note = f"📉 Trend down. Watching a 30-min close BELOW <code>{lo:.{dec}f}</code> for a SELL."
                short, pstate = f"trend DOWN: watching a close below {lo:.{dec}f} (SELL)", "down"
                watch = f"Waiting for a 30-min close below {lo:.{dec}f}  →  SELL"
            else:
                note = "➖ Trend unclear. No trade unless the trend becomes clear."
                short, pstate, watch = "trend unclear: no trade unless it clears", "none", "No trade unless the trend becomes clear"
            panels.append({"name": name, "df": df, "hi": hi, "lo": lo, "dec": dec, "note": short,
                           "state": pstate, "watch": watch})
            blocks.append(f"<b>{name}</b>\n   Yesterday: high <code>{hi:.{dec}f}</code> · low <code>{lo:.{dec}f}</code>\n   {note}")
        if blocks:
            news["levels_date"] = today   # mark first, so a retry or a second run cannot post it twice
            save_news(news)
            head = f"📍 <b>{BRAND} LEVELS TO WATCH</b> · {now.astimezone(WAT):%a %d %b}"
            intro = ("Our Daily Break strategy waits for a 30-minute close beyond yesterday's high or low, "
                     "in the direction of the trend (signals only come in the morning, 08:00 to 12:00, and the evening, 18:00 to 19:30 Nigeria time, at most one each, and everything is closed by 21:00).")
            full = f"{head}\n\n{intro} These are today's lines:\n\n" + "\n\n".join(blocks) \
                   + "\n\nNo breakout means no trade. Waiting is part of the plan."
            short = f"{head}\n\n{intro}\n\nNo breakout means no trade. Waiting is part of the plan."
            send_image_or_text(lambda: make_levels_chart(panels, f"LEVELS TO WATCH · {now.astimezone(WAT):%a %d %b}"),
                               full, caption=short)
            changed = True

    # 2) day recap
    if now.hour >= 20 and news["recap_date"] != today:
        names = {i["name"] for i in INSTRUMENTS}   # ignore retired instruments (old synthetics)
        live = [t for t in state["closed"] + state["open"] if t.get("live", True) and t["pair"] in names]
        opened = [t for t in live if str(t.get("opened_at", ""))[:10] == today]
        closed = [t for t in state["closed"] if t.get("live", True) and t["pair"] in names
                  and str(t.get("closed_at", ""))[:10] == today]
        still = [t for t in state["open"] if t.get("live", True) and t["pair"] in names]
        lines = [f"📊 <b>{BRAND} DAY RECAP</b> · {now.astimezone(WAT):%a %d %b}", ""]
        lines.append(f"Signals posted today: {len(opened)}")
        if closed:
            icon = {"TP2": "✅", "TP1": "🟡", "SL": "❌", "EXPIRED": "⏱", "DAYEND": "⏰"}
            word = {"TP2": "full target hit", "TP1": "half won, rest at break-even", "SL": "stopped out", "EXPIRED": "expired, no gain or loss", "DAYEND": "closed at the end of the day"}
            lines += [f"{icon.get(t['result'], '•')} {t['side']} {t['pair']}: {t['r']:+.1f}R ({word.get(t['result'], '')})" for t in closed]
            lines.append(f"Closed today: {sum(t['r'] for t in closed):+.1f}R")
        if still:
            lines.append("Still open: " + ", ".join(f"{t['side']} {t['pair']}" for t in still))
        if not opened and not closed:
            lines += ["", "No setups today. Daily Break only trades a clean breakout of yesterday's high or low that agrees "
                      "with the trend. No setup means no trade, and that is the plan."]
        done = [t for t in state["closed"] if t.get("live", True) and t.get("strategy") == "Daily Break"]
        if done:
            w, l, rate, r = record(done)
            lines += ["", f"📊 Daily Break record: {w}W / {l}L ({rate:.0f}%) | {r:+.1f}R"]
        else:
            lines += ["", "📊 Daily Break record: building (no closed trades yet)"]
        if closed or opened:
            lines += ["", "R = the amount you risk on a trade. +2R means you made double your risk, "
                      "-1R means the stop loss was hit, 0R means it closed at break-even."]
        news["recap_date"] = today   # mark first, so a retry or a second run cannot post it twice
        save_news(news)
        rows = [("Signals posted today: %d" % len(opened), "head")]
        kind_of = {"TP2": "win", "TP1": "win", "SL": "loss", "EXPIRED": "flat", "DAYEND": "info"}
        words = {"TP2": "full target hit", "TP1": "half won, rest at break-even", "SL": "stopped out",
                 "EXPIRED": "expired, no gain or loss", "DAYEND": "closed at the end of the day"}
        for t in closed:
            rows.append((f"{t['side']} {t['pair']}   {t['r']:+.1f}R   {words.get(t['result'], '')}", kind_of.get(t["result"], "info")))
        if closed:
            rows.append((f"Closed today: {sum(t['r'] for t in closed):+.1f}R", "head"))
        if still:
            rows.append(("Still open: " + ", ".join(f"{t['side']} {t['pair']}" for t in still), "dim"))
        if not opened and not closed:
            rows.append(("No setups today. Daily Break only trades a clean breakout of yesterday's high or low "
                         "that agrees with the trend. No setup means no trade, and that is the plan.", "info"))
        rows.append(("Daily Break record: " + (f"{w}W / {l}L ({rate:.0f}%) | {r:+.1f}R" if done else "building (no closed trades yet)"), "head"))
        if closed or opened:
            rows.append(("R = the amount you risk on a trade. +2R = made double your risk, -1R = stop loss hit, 0R = break-even.", "dim"))
        total = sum(t["r"] for t in closed)
        tiles = [("SIGNALS TODAY", str(len(opened)), "info"),
                 ("CLOSED TODAY", f"{total:+.1f}R" if closed else "none yet", "win" if total > 0 else ("loss" if total < 0 else "flat") if closed else "dim"),
                 ("DAILY BREAK RECORD", f"{w}W / {l}L" if done else "building", "info")]
        card_rows = [r_ for r_ in rows if r_[1] != "head"]
        send_image_or_text(lambda: make_recap_card(f"DAY RECAP · {now.astimezone(WAT):%a %d %b}", card_rows, tiles),
                           "\n".join(lines), caption=f"📊 <b>{BRAND} DAY RECAP</b> · {now.astimezone(WAT):%a %d %b}")
        changed = True

    if changed:
        save_news(news)


def run_extras_safe(candles, state, now):
    if not NEWS:
        return
    try:
        run_extras(candles, state, now)
    except Exception as e:  # extras must never stop the signals
        print(f"Extras step failed: {str(e)[:150]}")


def run_news_safe(force=False, repost=False):
    if not NEWS:
        return
    try:
        run_news(force, repost)
    except Exception as e:  # news must never stop the signals
        print(f"News step failed: {str(e)[:150]}")


def retire_removed_instruments(state):
    """Close (without scoring) any open trade on an instrument that has been switched off."""
    names = {i["name"] for i in INSTRUMENTS}
    gone = [t for t in state["open"] if t["pair"] not in names]
    if not gone:
        return
    now = datetime.now(timezone.utc).isoformat()
    for t in gone:
        t["result"], t["r"], t["closed_at"] = "EXPIRED", 0.0, now
        state["closed"].append(t)
    state["open"] = [t for t in state["open"] if t["pair"] in names]
    if any(t.get("live", True) for t in gone):
        send("ℹ️ Synthetic index signals have been discontinued. Any open synthetic trade is no longer tracked.")
    save_state(state)


# ---------- Timeframe test: London Breakout with 30 / 15 minute entries ----------
TF_PAIRS = ["EUR/USD", "GBP/USD", "XAU/USD"]
TF_MINUTES = [60, 30, 15]
TF_PAGES = {60: 3, 30: 6, 15: 10}   # pages of 5000 candles (about 1 to 2 years each)
TF_FILE = Path("state/tf.txt")


def fetch_tf(symbol, minutes, pages):
    """Several pages of candles at any size, going back in time (slow: obeys the rate limit)."""
    interval = {60: "1h", 30: "30min", 15: "15min"}[minutes]
    frames, end = [], None
    for page in range(pages):
        params = {"symbol": symbol, "interval": interval, "outputsize": 5000, "timezone": "UTC", "apikey": TD_KEY}
        if end:
            params["end_date"] = end
        data = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=90).json()
        if "values" not in data:
            if page == 0:
                raise RuntimeError(data.get("message", str(data)))
            break
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        frames.append(df)
        if len(df) < 100:
            break
        end = df["datetime"].min().strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(8)
    df = pd.concat(frames).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    now = datetime.utcnow()
    return df[df["datetime"] + timedelta(minutes=minutes) <= now].reset_index(drop=True)


def tf_breakout_trades(inst, df, minutes):
    """London Breakout on any candle size. Range, ATR and trend are always the same H1-based ones,
    so only the ENTRY candle changes (the first 30/15 minute close beyond the overnight range)."""
    k = 60 // minutes
    h1 = df.set_index("datetime").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev_close = h1["close"].shift(1)
    tr = pd.concat([h1["high"] - h1["low"], (h1["high"] - prev_close).abs(), (h1["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr_h = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().shift(1)     # last COMPLETED hour only
    ema_h = h1["close"].ewm(span=TREND_EMA, adjust=False).mean().shift(1)
    ema_h_old = ema_h.shift(10)
    hour_idx = df["datetime"].dt.floor("1h")
    atr = atr_h.reindex(hour_idx).values
    ema = ema_h.reindex(hour_idx).values
    ema_old = ema_h_old.reindex(hour_idx).values
    dts, highs, lows, closes = df["datetime"], df["high"].values, df["low"].values, df["close"].values
    days = dts.dt.date.values
    hours = dts.dt.hour.values
    expire = max(1, EXPIRE_HOURS * 60 // minutes)
    out, busy = [], -1
    minute_of_day = hours * 60 + dts.dt.minute.values
    split = int(len(df) * 0.7)
    a, b = STRATEGY_TARGETS["London Breakout"]
    starts = {}
    for i, d in enumerate(days):
        starts.setdefault(d, []).append(i)
    for day, idx in starts.items():
        idx = np.array(idx)
        asia = idx[hours[idx] < SESSION_START]
        sess = idx[(minute_of_day[idx] >= SESSION_START * 60) & (minute_of_day[idx] <= SESSION_END * 60)]
        if len(asia) < ASIA_RANGE_HOURS * k or len(sess) == 0:
            continue
        hi, lo = highs[asia].max(), lows[asia].min()
        i = None
        for c in sess:
            if closes[c] > hi or closes[c] < lo:
                i = c
                break
        if i is None or i <= busy or i >= len(df) - 1:
            continue
        at, e, eo = atr[i], ema[i], ema_old[i]
        if not (at > 0) or e != e or eo != eo:
            continue
        if not (0.5 * at <= hi - lo <= 4 * at):
            continue
        entry, mid = closes[i], (hi + lo) / 2
        side = "BUY" if entry > hi else "SELL"
        sl = mid - 0.1 * at if side == "BUY" else mid + 0.1 * at
        if side == "BUY" and not (entry > e and e > eo):
            continue
        if side == "SELL" and not (entry < e and e < eo):
            continue
        risk = abs(entry - sl)
        if risk > 3 * at:
            continue
        if risk < 0.5 * at:
            sl = entry - 0.5 * at if side == "BUY" else entry + 0.5 * at
        risk = abs(entry - sl)
        sign = 1 if side == "BUY" else -1
        end = min(len(df), i + 1 + expire)
        result, j, tp1_hit = walk_trade(side, entry, sl, entry + sign * a * risk, entry + sign * b * risk,
                                        highs[i + 1:end], lows[i + 1:end])
        if result is None:
            if end - (i + 1) < expire:
                continue
            result, j = ("TP1" if tp1_hit else "EXPIRED"), expire - 1
        busy = i + 1 + j
        r = trade_r(result, a, b) - cost_r(inst, entry, sl)
        out.append((i < split, r, r > 0))
    return out


def run_tf():
    results = {}   # minutes -> list of (is_early, r, won)
    for inst in INSTRUMENTS:
        if inst["name"] not in TF_PAIRS:
            continue
        for minutes in TF_MINUTES:
            try:
                time.sleep(8)
                df = fetch_tf(inst["symbol"], minutes, TF_PAGES[minutes])
            except Exception as e:
                print(f"Could not fetch {inst['name']} {minutes}m: {str(e)[:150]}")
                continue
            trades = tf_breakout_trades(inst, df, minutes)
            results.setdefault(minutes, []).extend(trades)
            print(f"{inst['name']} {minutes}m: {len(df)} candles, {len(trades)} trades")
    lines = ["==== TIMEFRAME TEST: London Breakout, 200-EMA+slope filter, TP1 1:1 / TP2 1:2, after spreads ====",
             "Range, ATR and trend are the same H1 ones everywhere. Only the entry candle (60 / 30 / 15 min) changes.",
             "Different timeframes cover different amounts of history, so compare R/trade and t, not total R.", "",
             f"{'entry':>6s} {'trades':>6s} {'win%':>5s} {'total R':>8s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
    for minutes in TF_MINUTES:
        t = results.get(minutes, [])
        if not t:
            continue
        n = len(t)
        wins = sum(1 for x in t if x[2])
        total = sum(x[1] for x in t)
        early = sum(x[1] for x in t if x[0])
        late = sum(x[1] for x in t if not x[0])
        ts = t_score([x[1] for x in t])
        good = "  good" if early > 0 and late > 0 and n >= 30 and ts >= 3 else ""
        lines.append(f"{minutes:>4d}m {n:6d} {100 * wins / n:5.0f} {total:+8.1f} {total / n:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{good}")
    lines += ["", "Spreads are subtracted; slippage is not. Lower timeframes pay the spread more often per move,",
              "so a 15/30 minute row only matters if it clearly beats the 60 minute row."]
    text = "\n".join(lines)
    print(text)
    TF_FILE.parent.mkdir(parents=True, exist_ok=True)
    TF_FILE.write_text(text + "\n")


# ---------- Strategy lab: several strategies x 60/30/15 minute entries, after spreads ----------
LAB_FILE = Path("state/lab.txt")
LAB_TARGETS = [(1.0, 1.5), (1.0, 2.0), (2.0, 3.0)]
LAB_STRATEGIES = ["Pullback", "Sweep Reversal", "NY Breakout", "Daily Break"]


def lab_prepare(df, minutes):
    k = 60 // minutes
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    ef = c.ewm(span=50 * k, adjust=False).mean()
    es = c.ewm(span=200 * k, adjust=False).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    dts = df["datetime"]
    return {
        "bbu": (mid + 2 * sd).values, "bbl": (mid - 2 * sd).values, "rsi": (100 - 100 / (1 + gain / loss)).values,
        "k": k, "o": df["open"].values, "h": h.values, "l": l.values, "c": c.values,
        "atr": atr.values, "atr_h": (atr * (k ** 0.5)).values, "ef": ef.values, "es": es.values,
        "es_old": es.shift(10 * k).values, "dt": dts, "days": dts.dt.date.values,
        "hour": dts.dt.hour.values, "mod": (dts.dt.hour * 60 + dts.dt.minute).values, "n": len(df),
    }


def lab_day_groups(d):
    groups = {}
    for i, day in enumerate(d["days"]):
        groups.setdefault(day, []).append(i)
    return {day: np.array(ix) for day, ix in groups.items()}


def lab_trend(d, i, side):
    e, eo, c = d["es"][i], d["es_old"][i], d["c"][i]
    if e != e or eo != eo:
        return False
    return (c > e and e > eo) if side == "BUY" else (c < e and e < eo)


def lab_fix_risk(side, entry, sl, atr_min, atr_max):
    risk = abs(entry - sl)
    if risk > atr_max:
        return None
    if risk < atr_min:
        sl = entry - atr_min if side == "BUY" else entry + atr_min
    return sl


def lab_pullback(d):
    out, k = [], d["k"]
    h, l, c, o, ef, atr = d["h"], d["l"], d["c"], d["o"], d["ef"], d["atr"]
    for i in range(200 * k + 10 * k, d["n"] - 1):
        if not (420 <= d["mod"][i] <= 1020) or not atr[i] > 0:
            continue
        for side in ("BUY", "SELL"):
            if not lab_trend(d, i, side) or (ef[i] > d["es"][i]) != (side == "BUY"):
                continue
            if side == "BUY" and l[i - 1] <= ef[i - 1] and c[i] > ef[i] and c[i] > o[i]:
                sl = lab_fix_risk(side, c[i], l[i - 4:i + 1].min() - 0.2 * atr[i], 0.5 * atr[i], 3 * atr[i])
            elif side == "SELL" and h[i - 1] >= ef[i - 1] and c[i] < ef[i] and c[i] < o[i]:
                sl = lab_fix_risk(side, c[i], h[i - 4:i + 1].max() + 0.2 * atr[i], 0.5 * atr[i], 3 * atr[i])
            else:
                continue
            if sl is not None:
                out.append((i, side, sl))
    return out


def lab_sweep_reversal(d):
    out, k = [], d["k"]
    for day, idx in lab_day_groups(d).items():
        asia = idx[d["hour"][idx] < SESSION_START]
        sess = idx[(d["mod"][idx] >= SESSION_START * 60) & (d["mod"][idx] <= SESSION_END * 60)]
        if len(asia) < ASIA_RANGE_HOURS * k or len(sess) == 0:
            continue
        hi, lo = d["h"][asia].max(), d["l"][asia].min()
        for i in sess:
            if i >= d["n"] - 1:
                break
            c, atr = d["c"][i], d["atr"][i]
            if c > hi or c < lo:        # a real breakout today: not a fake-out day
                break
            if not atr > 0:
                break
            side = sl = None
            if d["h"][i] > hi:
                side, sl = "SELL", d["h"][i] + 0.1 * atr
            elif d["l"][i] < lo:
                side, sl = "BUY", d["l"][i] - 0.1 * atr
            if side:
                sl = lab_fix_risk(side, c, sl, 0.5 * atr, 3 * atr)
                if sl is not None:
                    out.append((i, side, sl))
                break
    return out


def lab_ny_breakout(d):
    out, k = [], d["k"]
    for day, idx in lab_day_groups(d).items():
        rng = idx[(d["mod"][idx] >= 720) & (d["mod"][idx] < 810)]
        win = idx[(d["mod"][idx] >= 810) & (d["mod"][idx] <= 960)]
        if len(rng) < int(1.5 * k) or len(win) == 0:
            continue
        hi, lo = d["h"][rng].max(), d["l"][rng].min()
        for i in win:
            if i >= d["n"] - 1:
                break
            c, ah = d["c"][i], d["atr_h"][i]
            if not ah > 0 or not (0.5 * ah <= hi - lo <= 4 * ah):
                break
            if c > hi or c < lo:
                side = "BUY" if c > hi else "SELL"
                mid = (hi + lo) / 2
                sl = mid - 0.1 * ah if side == "BUY" else mid + 0.1 * ah
                if lab_trend(d, i, side):
                    sl = lab_fix_risk(side, c, sl, 0.5 * ah, 3 * ah)
                    if sl is not None:
                        out.append((i, side, sl))
                break
    return out


DAILY_END_MIN = 960   # Daily Break entries are allowed until this minute of the UTC day (960 = 16:00 UTC = 17:00 Nigeria)


def lab_daily_break(d, live=False, end_min=None, sl_mult=1.5):
    end_min = end_min or DAILY_END_MIN
    out = []
    days = sorted(lab_day_groups(d).items())
    prev = None
    for day, idx in days:
        if prev is not None:
            hi, lo = d["h"][prev].max(), d["l"][prev].min()
            for i in idx:
                if i >= d["n"] - 1 and not live:
                    break
                if not (420 <= d["mod"][i] <= end_min):
                    continue
                c, ah = d["c"][i], d["atr_h"][i]
                if not ah > 0:
                    break
                if c > hi or c < lo:
                    side = "BUY" if c > hi else "SELL"
                    if lab_trend(d, i, side):
                        sl = c - sl_mult * ah if side == "BUY" else c + sl_mult * ah
                        out.append((i, side, sl))
                    break
        prev = idx
    return out


def lab_asian_fade(d):
    """Quiet Asian hours: fade a stretch beyond the Bollinger band once price closes back inside it."""
    out = []
    c, o, h, l, atr = d["c"], d["o"], d["h"], d["l"], d["atr"]
    for i in range(30, d["n"] - 1):
        if not (0 <= d["mod"][i] < 360) or not atr[i] > 0:
            continue
        if c[i - 1] < d["bbl"][i - 1] and c[i] > d["bbl"][i] and c[i] > o[i] and d["rsi"][i - 1] < 35:
            side, sl = "BUY", l[i - 2:i + 1].min() - 0.2 * atr[i]
        elif c[i - 1] > d["bbu"][i - 1] and c[i] < d["bbu"][i] and c[i] < o[i] and d["rsi"][i - 1] > 65:
            side, sl = "SELL", h[i - 2:i + 1].max() + 0.2 * atr[i]
        else:
            continue
        sl = lab_fix_risk(side, c[i], sl, 0.5 * atr[i], 2.5 * atr[i])
        if sl is not None:
            out.append((i, side, sl))
    return out


def lab_momentum(d):
    """A big candle closing near its high/low, in the direction of the trend, during London/New York hours."""
    out = []
    c, o, h, l, atr = d["c"], d["o"], d["h"], d["l"], d["atr"]
    for i in range(300, d["n"] - 1):
        if not (420 <= d["mod"][i] <= 1020) or not atr[i - 1] > 0:
            continue
        rng = h[i] - l[i]
        if rng <= 0 or abs(c[i] - o[i]) < 1.3 * atr[i - 1]:
            continue
        if c[i] > o[i] and (c[i] - l[i]) / rng >= 0.75 and lab_trend(d, i, "BUY"):
            side, sl = "BUY", c[i] - 1.5 * atr[i - 1]
        elif c[i] < o[i] and (h[i] - c[i]) / rng >= 0.75 and lab_trend(d, i, "SELL"):
            side, sl = "SELL", c[i] + 1.5 * atr[i - 1]
        else:
            continue
        out.append((i, side, sl))
    return out


IDEA_STRATEGIES = ["Pullback", "Asian Fade", "Momentum Bar"]
LAB_FUNCS = {"Asian Fade": lab_asian_fade, "Momentum Bar": lab_momentum, "Pullback": lab_pullback, "Sweep Reversal": lab_sweep_reversal,
             "NY Breakout": lab_ny_breakout, "Daily Break": lab_daily_break}


def lab_evaluate(inst, d, cands, minutes, a, b):
    """Run the candidate entries through the same trade rules as everywhere else. Returns (early, r, won, hours)."""
    expire = max(1, EXPIRE_HOURS * 60 // minutes)
    split, busy, out = int(d["n"] * 0.7), -1, []
    for i, side, sl in cands:
        if i <= busy:
            continue
        entry = d["c"][i]
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        sign = 1 if side == "BUY" else -1
        end = min(d["n"], i + 1 + expire)
        result, j, tp1_hit = walk_trade(side, entry, sl, entry + sign * a * risk, entry + sign * b * risk,
                                        d["h"][i + 1:end], d["l"][i + 1:end])
        if result is None:
            if end - (i + 1) < expire:
                continue
            result, j = ("TP1" if tp1_hit else "EXPIRED"), expire - 1
        busy = i + 1 + j
        r = trade_r(result, a, b) - cost_r(inst, entry, sl)
        out.append((i < split, r, r > 0, (j + 1) * minutes / 60))
    return out


def lab_daily_break_late(d):
    """Same as Daily Break but entries allowed until 20:00 UTC."""
    return lab_daily_break(d, end_min=1200)


def lab_daily_break_free(d):
    """Daily Break without the trend filter: more signals, but fewer 'with the trend' trades."""
    out = []
    prev = None
    for day, idx in sorted(lab_day_groups(d).items()):
        if prev is not None:
            hi, lo = d["h"][prev].max(), d["l"][prev].min()
            for i in idx:
                if i >= d["n"] - 1:
                    break
                if not (420 <= d["mod"][i] <= DAILY_END_MIN):
                    continue
                c, ah = d["c"][i], d["atr_h"][i]
                if not ah > 0:
                    break
                if c > hi or c < lo:
                    side = "BUY" if c > hi else "SELL"
                    out.append((i, side, c - 1.5 * ah if side == "BUY" else c + 1.5 * ah))
                    break
        prev = idx
    return out


def lab_daily_break_multi(d, live=False):
    """Daily Break, then up to 2 more trend-following signals the same day when price makes a new day high/low."""
    out, k = [], d["k"]
    prev = None
    for day, idx in sorted(lab_day_groups(d).items()):
        if prev is not None:
            hi, lo = d["h"][prev].max(), d["l"][prev].min()
            fired, last_i = 0, -999
            for pos, i in enumerate(idx):
                if i >= d["n"] - (0 if live else 1) or fired >= 3:
                    break
                if not (420 <= d["mod"][i] <= (LIVE_END_MIN if live else 1080)):
                    continue
                c, ah = d["c"][i], d["atr_h"][i]
                if not ah > 0:
                    break
                if fired == 0:
                    side = "BUY" if c > hi else ("SELL" if c < lo else None)
                else:
                    if i - last_i < 2 * k:
                        continue
                    td_hi, td_lo = d["h"][idx[:pos]].max(), d["l"][idx[:pos]].min()
                    side = "BUY" if c > td_hi else ("SELL" if c < td_lo else None)
                if side and lab_trend(d, i, side):
                    out.append((i, side, c - 1.5 * ah if side == "BUY" else c + 1.5 * ah))
                    fired += 1
                    last_i = i
                elif side and fired == 0:
                    break
        prev = idx
    return out


def lab_asia_break(d):
    """Break of the Asian-session range (00:00-07:00 UTC) during London/New York, with the trend."""
    out, k = [], d["k"]
    for day, idx in sorted(lab_day_groups(d).items()):
        asia = idx[d["mod"][idx] < 420]
        win = idx[(d["mod"][idx] >= 420) & (d["mod"][idx] <= DAILY_END_MIN)]
        if len(asia) < 5 * k or len(win) == 0:
            continue
        hi, lo = d["h"][asia].max(), d["l"][asia].min()
        for i in win:
            if i >= d["n"] - 1:
                break
            c, ah = d["c"][i], d["atr_h"][i]
            if not ah > 0:
                break
            if c > hi or c < lo:
                side = "BUY" if c > hi else "SELL"
                if lab_trend(d, i, side):
                    out.append((i, side, c - 1.5 * ah if side == "BUY" else c + 1.5 * ah))
                break
    return out


MORE_STRATEGIES = ["Daily Break", "Break Late", "Break No Filter", "Break Multi", "Asia Break"]
LAB_FUNCS.update({"Break Late": lab_daily_break_late, "Break No Filter": lab_daily_break_free,
                  "Break Multi": lab_daily_break_multi, "Asia Break": lab_asia_break})
MORE_FILE = Path("state/more.txt")


def run_more():
    run_lab(["XAU/USD"], MORE_FILE, "GOLD MORE SIGNALS (XAU/USD)", MORE_STRATEGIES, [60, 30])


def _lab_retest(d, level_fn, end_min=960):
    """Breakout, then wait for price to come back, touch the broken level and hold it. One entry per day, with the trend."""
    out, k = [], d["k"]
    prev = None
    for day, idx in sorted(lab_day_groups(d).items()):
        lv = level_fn(d, idx, prev)
        prev = idx
        if lv is None:
            continue
        hi, lo = lv
        side = level = brk = None
        for pos, i in enumerate(idx):
            if i >= d["n"] - 1:
                break
            m = d["mod"][i]
            if m < 420 or m > end_min + 120:
                continue
            c, o, ah = d["c"][i], d["o"][i], d["atr_h"][i]
            if not ah > 0:
                break
            if side is None:
                if m <= end_min and (c > hi or c < lo):
                    side, level, brk = ("BUY", hi, pos) if c > hi else ("SELL", lo, pos)
                continue
            if pos - brk > 4 * k:
                break
            if side == "BUY":
                if c < level - 0.3 * ah:
                    break
                if d["l"][i] <= level + 0.15 * ah and c > level and c > o and lab_trend(d, i, "BUY"):
                    sl = lab_fix_risk("BUY", c, d["l"][idx[brk:pos + 1]].min() - 0.2 * ah, 0.5 * ah, 3 * ah)
                    if sl is not None:
                        out.append((i, "BUY", sl))
                    break
            else:
                if c > level + 0.3 * ah:
                    break
                if d["h"][i] >= level - 0.15 * ah and c < level and c < o and lab_trend(d, i, "SELL"):
                    sl = lab_fix_risk("SELL", c, d["h"][idx[brk:pos + 1]].max() + 0.2 * ah, 0.5 * ah, 3 * ah)
                    if sl is not None:
                        out.append((i, "SELL", sl))
                    break
    return out


def lab_retest_daily(d):
    return _lab_retest(d, lambda d, idx, prev: None if prev is None else (d["h"][prev].max(), d["l"][prev].min()))


def lab_retest_asia(d):
    def lv(d, idx, prev):
        asia = idx[d["mod"][idx] < 420]
        return (d["h"][asia].max(), d["l"][asia].min()) if len(asia) >= 5 * d["k"] else None
    return _lab_retest(d, lv)


RETEST_STRATEGIES = ["Retest Daily", "Retest Asia"]
LAB_FUNCS.update({"Retest Daily": lab_retest_daily, "Retest Asia": lab_retest_asia})
RETEST_FILE = Path("state/retest.txt")


def run_retest():
    pairs = [i["name"] for i in INSTRUMENTS if i["kind"] == "forex"]
    run_lab(pairs, RETEST_FILE, "BREAKOUT + RETEST (all 7 pairs pooled)", RETEST_STRATEGIES, [60, 30])


VOL_PAGES = {60: 4, 30: 8}   # pages of 5000 candles (30m: about 2 years)


def fetch_deriv_tf(symbol, minutes, pages):
    """Several pages of Deriv candles at any size, going back in time."""
    import websocket
    frames, end = [], "latest"
    for _ in range(pages):
        data, errors = None, []
        for url in DERIV_URLS:
            try:
                ws = websocket.create_connection(url, timeout=30)
                try:
                    ws.send(json.dumps({"ticks_history": symbol, "adjust_start_time": 1, "count": 5000, "end": end,
                                        "granularity": minutes * 60, "style": "candles"}))
                    for _ in range(5):
                        data = json.loads(ws.recv())
                        if "candles" in data or "error" in data:
                            break
                finally:
                    ws.close()
                if data and "candles" in data:
                    break
                errors.append(str((data or {}).get("error", "no candles"))[:70])
            except Exception as e:
                errors.append(str(e)[:70])
            data = None
        if data is None:
            if not frames:
                raise RuntimeError(" | ".join(errors))
            break
        df = pd.DataFrame(data["candles"])
        frames.append(df)
        if len(df) < 100:
            break
        end = int(df["epoch"].min()) - 1
        time.sleep(1)
    df = pd.concat(frames).drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[df["datetime"] + timedelta(minutes=minutes) <= datetime.utcnow()].reset_index(drop=True)


def lab_vol_break(d, multi=False, live=False):
    """Daily Break for markets that never close: yesterday's high/low (UTC day), any hour, with the trend."""
    out, k = [], d["k"]
    prev = None
    for day, idx in sorted(lab_day_groups(d).items()):
        if prev is not None:
            hi, lo = d["h"][prev].max(), d["l"][prev].min()
            fired, last_i = 0, -999
            for pos, i in enumerate(idx):
                if i >= d["n"] - (0 if live else 1) or fired >= (3 if multi else 1):
                    break
                c, ah = d["c"][i], d["atr_h"][i]
                if not ah > 0:
                    break
                if fired == 0:
                    side = "BUY" if c > hi else ("SELL" if c < lo else None)
                else:
                    if i - last_i < 2 * k:
                        continue
                    td_hi, td_lo = d["h"][idx[:pos]].max(), d["l"][idx[:pos]].min()
                    side = "BUY" if c > td_hi else ("SELL" if c < td_lo else None)
                if side and lab_trend(d, i, side):
                    out.append((i, side, c - 1.5 * ah if side == "BUY" else c + 1.5 * ah))
                    fired += 1
                    last_i = i
                elif side and fired == 0:
                    break
        prev = idx
    return out


VOL_STRATEGIES = ["Vol Break", "Vol Break Multi"]
LAB_FUNCS.update({"Vol Break": lambda d: lab_vol_break(d), "Vol Break Multi": lambda d: lab_vol_break(d, multi=True)})
VOL_FILE = Path("state/vol.txt")


def run_vol():
    insts = [{"name": n, "kind": "synthetic", "symbol": sym} for n, sym in SYNTHETICS_OFF.items()]
    run_lab([i["name"] for i in insts], VOL_FILE, "VOLATILITY INDICES (breakout, 24/7)", VOL_STRATEGIES, [60, 30],
            source=lambda inst, m: fetch_deriv_tf(inst["symbol"], m, VOL_PAGES[m]), instruments=insts)


GOLD_FILE = Path("state/gold.txt")


def run_gold():
    run_lab(["XAU/USD"], GOLD_FILE, "GOLD LAB (XAU/USD only)")


IDEAS_FILE = Path("state/ideas.txt")


def run_ideas():
    pairs = [i["name"] for i in INSTRUMENTS if i["kind"] == "forex"]
    run_lab(pairs, IDEAS_FILE, "IDEAS LAB (all 7 pairs)", IDEA_STRATEGIES, [30])


def run_lab(pairs=None, out_file=None, title="STRATEGY LAB (3 pairs pooled)", strategies=None, minutes_list=None, source=None, instruments=None):
    pairs = pairs or TF_PAIRS
    out_file = out_file or LAB_FILE
    pair_strats = strategies or ["Daily Break"]
    strategies = strategies or LAB_STRATEGIES
    minutes_list = minutes_list or TF_MINUTES
    rows = {}   # (strategy, minutes, target) -> trades
    by_pair = {}   # (strategy, minutes, pair) -> trades at the 1/2 target
    day_counts = {}   # minutes -> trading days per pair
    for inst in (instruments or INSTRUMENTS):
        if inst["name"] not in pairs:
            continue
        for minutes in minutes_list:
            try:
                if source:
                    df = source(inst, minutes)
                else:
                    time.sleep(8)
                    df = fetch_tf(inst["symbol"], minutes, TF_PAGES[minutes])
            except Exception as e:
                print(f"Could not fetch {inst['name']} {minutes}m: {str(e)[:150]}")
                continue
            d = lab_prepare(df, minutes)
            day_counts.setdefault(minutes, []).append(max(1, len(lab_day_groups(d))))
            for strat in strategies:
                cands = LAB_FUNCS[strat](d)
                for tgt in LAB_TARGETS:
                    res = lab_evaluate(inst, d, cands, minutes, *tgt)
                    rows.setdefault((strat, minutes, tgt), []).extend(res)
                    if tgt == (1.0, 2.0):
                        by_pair[(strat, minutes, inst["name"])] = res
            print(f"{inst['name']} {minutes}m done ({len(df)} candles)")
    lines = [f"==== {title}, after typical spreads ====",
             "'sig/d' = signals per day (all pairs in this table added together).",
             "Half closes at TP1, stop to entry, rest runs to TP2. 'hrs' = average hours a trade stays open.",
             "'good' = t of 3+, positive in BOTH the early 70% and late 30%, 30+ trades.",
             "Many variants are tried here, so a lucky winner is likely. Trust only 'good' rows with similar neighbours."]
    for strat in strategies:
        lines += ["", f"-- {strat} --",
                  f"{'entry':>5s} {'TP1/TP2':>8s} {'trades':>6s} {'sig/d':>5s} {'win%':>5s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s} {'hrs':>5s}"]
        for minutes in minutes_list:
            for tgt in LAB_TARGETS:
                t = rows.get((strat, minutes, tgt), [])
                if not t:
                    continue
                n = len(t)
                tot = sum(x[1] for x in t)
                early = sum(x[1] for x in t if x[0])
                late = sum(x[1] for x in t if not x[0])
                ts = t_score([x[1] for x in t])
                good = "  good" if early > 0 and late > 0 and n >= 30 and ts >= 3 else ""
                lines.append(f"{minutes:>4d}m {tgt[0]:g}/{tgt[1]:g}".ljust(15)
                             + f"{n:6d} {n / (sum(day_counts[minutes]) / len(day_counts[minutes])):5.2f} {100 * sum(1 for x in t if x[2]) / n:5.0f} {tot / n:+8.2f} {ts:5.1f} "
                             f"{early:+8.1f} {late:+8.1f} {sum(x[3] for x in t) / n:5.1f}{good}")
    lines += ["", "-- PER PAIR (TP1/TP2 1/2): is one pair carrying the result? --",
              f"{'strategy':15s} {'entry':>5s} {'pair':8s} {'trades':>6s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
    for (strat, minutes, pair), t in sorted(by_pair.items()):
        if strat not in pair_strats or not t:
            continue
        good = ""
        n = len(t)
        lines.append(f"{strat:15s} {minutes:>4d}m {pair:8s} {n:6d} {sum(x[1] for x in t) / n:+8.2f} "
                     f"{t_score([x[1] for x in t]):5.1f} {sum(x[1] for x in t if x[0]):+8.1f} {sum(x[1] for x in t if not x[0]):+8.1f}")
    text = "\n".join(lines)
    print(text)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text + "\n")


# ---------- Images: signal charts ----------
CHARTS = True          # draw a chart image for every signal (falls back to plain text if anything goes wrong)
CHART_CANDLES = 70     # how many recent candles the chart shows


def _plt():
    """Load matplotlib; install it on the fly if requirements.txt was not updated yet."""
    try:
        import matplotlib
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", "matplotlib"],
                       check=False)
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _theme():
    return dict(bg1="#020a0e", bg2="#06232b", panel="#06171d", edge="#1b6670", fg="#e6fbfa", dim="#78a8af",
                up="#19e6b0", down="#ff5c7a", amber="#ffc14d", blue="#38b6ff", cyan="#1fd8d0",
                mono="DejaVu Sans Mono", sans="DejaVu Sans")


def _spaced(text):
    """'OWN YOUR EDGE' -> 'O W N   Y O U R   E D G E' (the wide-letter tagline look)."""
    return "   ".join(" ".join(w) for w in text.split())


def _canvas(plt, w, h, accent):
    """Dark teal stage: soft glow behind the card, a light beam on the floor, faint dot texture. Units are inches."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    t = _theme()
    fig = plt.figure(figsize=(w, h), dpi=150)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    bg.set_autoscale_on(False)
    grad = np.linspace(1, 0, 256).reshape(-1, 1)
    bg.imshow(grad, extent=[0, w, 0, h], cmap=LinearSegmentedColormap.from_list("bg", [t["bg2"], t["bg1"]]),
              aspect="auto", zorder=0)
    yy, xx = np.mgrid[0:1:200j, 0:1:300j]
    rgba = np.zeros((200, 300, 4))
    rgba[..., :3] = to_rgb(accent)
    rr = np.sqrt(((xx - 0.5) * 1.5) ** 2 + (yy - 0.98) ** 2)
    glow = np.clip(1 - rr / 0.9, 0, 1) ** 2 * 0.34                      # big soft glow at the top
    beam = np.exp(-(((xx - 0.5) / 0.30) ** 2) - (((yy - 0.028) / 0.017) ** 2)) * 0.85   # thin light on the floor
    haze = np.exp(-(((xx - 0.5) / 0.46) ** 2) - (((yy - 0.028) / 0.065) ** 2)) * 0.20
    rgba[..., 3] = np.clip(glow + beam + haze, 0, 0.95)
    bg.imshow(rgba, extent=[0, w, 0, h], aspect="auto", zorder=1, origin="lower")
    gx, gy = np.meshgrid(np.arange(0.2, w, 0.26), np.arange(0.2, h, 0.26))   # faint dot texture
    bg.scatter(gx.ravel(), gy.ravel(), s=0.5, color=t["fg"], alpha=0.07, linewidths=0, zorder=1.5)
    bg.set_xlim(0, w)
    bg.set_ylim(0, h)
    return fig, bg, t


def _panel(bg, x, y, w, h, t, accent=None, r=0.14):
    """Glass panel: soft outer glow, translucent body, bright rim, diagonal sheen."""
    from matplotlib.patches import FancyBboxPatch
    rim = accent or t["edge"]
    for lw, a in ((9, 0.05), (5, 0.08), (2.6, 0.12)):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=rim,
                                    lw=lw, alpha=a, zorder=2))
    body = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=t["panel"], ec=rim,
                          lw=1.1, alpha=0.86, zorder=2.2)
    bg.add_patch(body)
    sheen = np.zeros((40, 80, 4))
    sheen[..., :3] = 1.0
    gy, gx = np.mgrid[0:1:40j, 0:1:80j]
    sheen[..., 3] = np.clip(1 - (gx * 0.75 + (1 - gy) * 0.55), 0, 1) ** 2 * 0.14
    im = bg.imshow(sheen, extent=[x, x + w, y, y + h], aspect="auto", zorder=2.4, origin="lower")
    im.set_clip_path(body)
    if accent:
        bg.plot([x + r, x + w - r], [y + h, y + h], color=accent, lw=2.0, alpha=0.95, zorder=3, solid_capstyle="round")


def _pill(bg, x, y, w, h, color, text, fs=10, tc="#04110d", t=None):
    from matplotlib.patches import FancyBboxPatch
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=color, ec="none", zorder=4))
    bg.text(x + w / 2, y + h / 2 - 0.005, text, color=tc, fontsize=fs, fontweight="bold", ha="center", va="center",
            family=(t or _theme())["sans"], zorder=5)


def _wordmark(fig, bg, x_right, y, t, accent):
    """PIP + PULSE wordmark with a little heartbeat line."""
    p2 = bg.text(x_right, y, "PULSE", color=accent, fontsize=15, fontweight="bold", ha="right", va="center", family=t["sans"], zorder=5)
    fig.canvas.draw()
    width = p2.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    p1 = bg.text(x_right - width, y, "PIP", color=t["fg"], fontsize=15, fontweight="bold", ha="right", va="center", family=t["sans"], zorder=5)
    fig.canvas.draw()
    w1 = p1.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    x0 = x_right - width - w1 - 0.68
    xs = [0, 0.10, 0.17, 0.24, 0.31, 0.38, 0.52]
    ys = [0, 0, 0.13, -0.11, 0.07, 0, 0]
    bg.plot([x0 + a for a in xs], [y + b for b in ys], color=accent, lw=1.6, zorder=5,
            path_effects=[_pe().Stroke(linewidth=5, foreground=accent, alpha=0.22), _pe().Normal()])


def _pe():
    import matplotlib.patheffects as pe
    return pe


def _glow(color, w=6, a=0.2):
    return [_pe().Stroke(linewidth=w, foreground=color, alpha=a), _pe().Normal()]


def _dots(bg, x, y, width, color, lit, n=16, size=7):
    """A row of dots, the first `lit` share of them bright (dot-matrix gauge)."""
    xs = np.linspace(x, x + width, n)
    k = int(round(max(0.0, min(1.0, lit)) * n))
    bg.scatter(xs[:k], [y] * k, s=size, color=color, linewidths=0, alpha=0.95, zorder=5)
    bg.scatter(xs[k:], [y] * (n - k), s=size, color=color, linewidths=0, alpha=0.20, zorder=5)


def _card(bg, x, y, w, h, t, label, value, sub, color, lit=None):
    _panel(bg, x, y, w, h, t, accent=color, r=0.10)
    bg.text(x + 0.14, y + h - 0.17, label, color=t["dim"], fontsize=6.6, fontweight="bold", va="center", family=t["mono"], zorder=5)
    bg.text(x + 0.14, y + h / 2 - 0.02, value, color=color, fontsize=12.5, fontweight="bold", va="center", family=t["mono"], zorder=5)
    if lit is not None:
        _dots(bg, x + 0.16, y + 0.17, w - 0.34, color, lit)
    elif sub:
        bg.text(x + 0.14, y + 0.15, sub, color=t["dim"], fontsize=6.8, va="center", family=t["sans"], zorder=5)
    if lit is not None and sub:
        bg.text(x + w - 0.14, y + h - 0.17, sub, color=t["dim"], fontsize=6.4, va="center", ha="right", family=t["mono"], zorder=5)


def _candles(ax, plt, d, t):
    for i, r in d.reset_index(drop=True).iterrows():
        col = t["up"] if r["close"] >= r["open"] else t["down"]
        ax.plot([i, i], [r["low"], r["high"]], color=col, linewidth=0.8, zorder=3, alpha=0.9)
        lo, hi = min(r["open"], r["close"]), max(r["open"], r["close"])
        ax.add_patch(plt.Rectangle((i - 0.34, lo), 0.68, max(hi - lo, 1e-9), facecolor=col, edgecolor=col,
                                   linewidth=0.4, alpha=0.92, zorder=4))


def _zone(ax, x0, x1, y_a, y_b, color, strong_at_a):
    """Gradient-filled box between two prices, strongest next to y_a."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    r, g, b = to_rgb(color)
    cm = LinearSegmentedColormap.from_list("z", [(r, g, b, 0.03), (r, g, b, 0.34)])
    lo, hi = min(y_a, y_b), max(y_a, y_b)
    strong_low = (strong_at_a and y_a < y_b) or (not strong_at_a and y_b < y_a)
    arr = np.linspace(1, 0, 128) if strong_low else np.linspace(0, 1, 128)
    ax.imshow(arr.reshape(-1, 1), extent=[x0, x1, lo, hi], origin="lower", aspect="auto", cmap=cm, zorder=1)


def _style_axes(ax, t):
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color=t["edge"], linewidth=0.6, alpha=0.55, linestyle=(0, (2, 3)))
    ax.tick_params(axis="y", colors=t["dim"], labelsize=7, length=0)
    ax.tick_params(axis="x", colors=t["dim"], labelsize=7, length=0)
    for lab in ax.get_yticklabels():
        lab.set_family(t["mono"])


def _tag(ax, x, price, text, color, t, fg="#04110d"):
    ax.text(x, price, text, color=fg, fontsize=7.6, fontweight="bold", va="center", ha="left", family=t["mono"],
            zorder=9, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=color, ec="none"))


def make_signal_chart(s, df, path="signal_chart.png"):
    """Glass-card signal: candles with profit/loss zones, glowing levels, and four numbered stat tiles with dot gauges."""
    plt = _plt()
    W, H = 9.0, 5.7
    t = _theme()
    side_col = t["up"] if s["side"] == "BUY" else t["down"]
    fig, bg, t = _canvas(plt, W, H, t["cyan"])
    exp = bool(s.get("exp"))
    # header
    _pill(bg, 0.4, H - 0.78, 0.95, 0.38, side_col, s["side"], fs=12, t=t)
    bg.text(1.5, H - 0.58, s["pair"], color=t["fg"], fontsize=22, fontweight="bold", va="center", family=t["mono"], zorder=5)
    bg.text(0.42, H - 0.98, f"{s['strategy'].upper()}  ·  30M  ·  {'EXPERIMENTAL SIGNAL' if exp else 'VIP SIGNAL'}",
            color=t["dim"], fontsize=7.6, fontweight="bold", va="center", family=t["mono"], zorder=5)
    _wordmark(fig, bg, W - 0.4, H - 0.58, t, t["cyan"])
    # chart panel
    py0, ph = 2.02, 2.62
    _panel(bg, 0.3, py0, W - 0.6, ph, t, accent=t["cyan"])
    d = df.tail(CHART_CANDLES).reset_index(drop=True)
    n, pad = len(d), 12
    ax = fig.add_axes([0.62 / W, (py0 + 0.16) / H, 6.55 / W, (ph - 0.32) / H])
    _style_axes(ax, t)
    _candles(ax, plt, d, t)
    entry, sl, tp1, tp2 = s["entry"], s["sl"], s["tp1"], s["tp2"]
    x0, x1 = n - 1, n - 1 + pad
    _zone(ax, x0, x1, entry, tp2, t["up"], True)
    _zone(ax, x0, x1, entry, sl, t["down"], True)
    for price, col, ls, lw in ((tp2, t["up"], (0, (4, 3)), 1.2), (tp1, t["up"], (0, (4, 3)), 1.2),
                               (entry, t["fg"], "-", 1.4), (sl, t["down"], (0, (4, 3)), 1.2)):
        ax.plot([x0 - 10, x1], [price, price], color=col, linestyle=ls, linewidth=lw, zorder=6, path_effects=_glow(col, 5, 0.18))
    ax.scatter([n - 1], [entry], s=230, color=side_col, alpha=0.18, zorder=7)
    ax.scatter([n - 1], [entry], s=36, color=t["fg"], edgecolor=side_col, linewidth=1.6, zorder=8)
    lows = list(d["low"]) + [sl, entry, tp1, tp2]
    highs = list(d["high"]) + [sl, entry, tp1, tp2]
    span = max(highs) - min(lows)
    ax.set_ylim(min(lows) - span * 0.07, max(highs) + span * 0.07)
    ax.set_xlim(-1, x1 + 0.2)
    step = max(1, n // 5)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([d["datetime"].iloc[i].strftime("%d %b %H:%M") for i in ticks])
    dec = len(str(s["entry"]).split(".")[-1])
    for lab, price, col in (("TP2", tp2, t["up"]), ("TP1", tp1, t["up"]), ("ENTRY", entry, t["fg"]), ("SL", sl, t["down"])):
        _tag(ax, x1 + 0.7, price, f"{lab} {price:.{dec}f}", col, t)
    # stat tiles (dot gauge = size of the move compared with 3R)
    cw, gap, y0, ch = (W - 0.6 - 3 * 0.14) / 4, 0.14, 0.78, 1.0
    unit = s["unit"]
    cards = [("01  ENTRY", f"{entry:.{dec}f}", "30m close", t["fg"], None),
             ("02  STOP LOSS", f"{sl:.{dec}f}", f"-{s['risk_pips']} {unit}", t["down"], 1.0 / 3),
             ("03  TP1 · 1:%g" % s["tp1_r"], f"{tp1:.{dec}f}", f"+{round(s['risk_pips'] * s['tp1_r'], 1)} {unit}", t["up"], s["tp1_r"] / 3),
             ("04  TP2 · 1:%g" % s["tp2_r"], f"{tp2:.{dec}f}", f"+{round(s['risk_pips'] * s['tp2_r'], 1)} {unit}", t["up"], s["tp2_r"] / 3)]
    for k, (lab, val, sub, col, lit) in enumerate(cards):
        _card(bg, 0.3 + k * (cw + gap), y0, cw, ch, t, lab, val, sub, col, lit)
    foot = ("Experimental, unverified  ·  Risk 0.25% max  ·  At TP1 close half and move your stop to entry" if exp
            else "Risk 1% max per trade  ·  At TP1 close half and move your stop to entry  ·  Trade at your own risk")
    bg.text(W / 2, 0.60, foot, color=t["dim"], fontsize=6.6, ha="center", va="center", family=t["sans"], zorder=5)
    bg.text(W / 2, 0.36, _spaced("TRADE THE PLAN."), color=t["cyan"], fontsize=6.8, ha="center", va="center",
            family=t["mono"], alpha=0.9, zorder=5)
    fig.savefig(path, facecolor=t["bg1"])
    plt.close(fig)
    return path


def send_photo(path, caption=""):
    with open(path, "rb") as f:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
            data={"chat_id": TG_CHAT, "caption": caption, "parse_mode": "HTML"},
            files={"photo": f}, timeout=60,
        )
    r.raise_for_status()


def send_signal(sig, text, df):
    """Post a signal as chart + caption. If the picture fails for any reason, post the plain text instead."""
    if CHARTS and df is not None:
        try:
            path = make_signal_chart(sig, df)
            if len(text) <= 1000:
                send_photo(path, text)
            else:
                send_photo(path)
                send(text)
            return
        except Exception as e:
            print(f"Chart failed, sending text only: {str(e)[:150]}")
    send(text)


# ---------- Images: levels chart, recap card, test chart ----------
def make_levels_chart(panels, title, path="levels_chart.png"):
    """One glass panel per pair: candles, yesterday's high/low, and what the bot is waiting for."""
    plt = _plt()
    n_p = len(panels)
    W, ph = 9.0, 2.75
    H = 1.35 + ph * n_p + 0.65
    fig, bg, t = _canvas(plt, W, H, _theme()["cyan"])
    _pill(bg, 0.35, H - 0.72, 2.0, 0.38, t["cyan"], "LEVELS TO WATCH", fs=10, t=t)
    bg.text(2.55, H - 0.53, title.split("·")[-1].strip(), color=t["fg"], fontsize=12, fontweight="bold", va="center", family=t["mono"], zorder=5)
    _wordmark(fig, bg, W - 0.4, H - 0.53, t, t["cyan"])
    state_style = {"up": ("TREND UP", t["up"]), "down": ("TREND DOWN", t["down"]),
                   "none": ("NO CLEAR TREND", t["amber"]), "broke": ("ALREADY BROKE OUT", t["dim"])}
    for k, p in enumerate(panels):
        top = H - 1.0 - k * ph
        y0 = top - ph + 0.15
        _panel(bg, 0.3, y0, W - 0.6, ph - 0.2, t)
        label, col = state_style[p["state"]]
        bg.text(0.55, top - 0.22, p["name"], color=t["fg"], fontsize=13, fontweight="bold", va="center", family=t["mono"], zorder=5)
        _pill(bg, 1.95, top - 0.37, 1.55, 0.3, col, label, fs=7.5, tc="#06101c", t=t)
        bg.text(3.7, top - 0.22, p["watch"], color=t["dim"], fontsize=8, va="center", family=t["sans"], zorder=5)
        d = p["df"].tail(60).reset_index(drop=True)
        m = len(d)
        ax = fig.add_axes([0.62 / W, (y0 + 0.12) / H, 6.55 / W, (ph - 0.2 - 0.62) / H])
        _style_axes(ax, t)
        _candles(ax, plt, d, t)
        for price, name, c2, hot in ((p["hi"], "HIGH", t["up"], p["state"] == "up"), (p["lo"], "LOW", t["down"], p["state"] == "down")):
            ax.plot([-1, m + 6], [price, price], color=c2, linestyle="-" if hot else (0, (4, 3)),
                    linewidth=1.8 if hot else 1.1, zorder=6, path_effects=_glow(c2, 7 if hot else 4, 0.25 if hot else 0.15))
            _tag(ax, m + 7.0, price, f"{name} {price:.{p['dec']}f}", c2, t)
        lows, highs = list(d["low"]) + [p["lo"]], list(d["high"]) + [p["hi"]]
        span = max(highs) - min(lows)
        ax.set_ylim(min(lows) - span * 0.08, max(highs) + span * 0.08)
        ax.set_xlim(-1, m + 6.5)
        ax.set_xticks([])
    bg.text(W / 2, 0.62, "No breakout means no trade. Waiting is part of the plan.", color=t["dim"], fontsize=7.4,
            ha="center", va="center", family=t["sans"], zorder=5)
    bg.text(W / 2, 0.36, _spaced("WAIT FOR THE EDGE."), color=t["cyan"], fontsize=6.8, ha="center", va="center",
            family=t["mono"], alpha=0.9, zorder=5)
    fig.savefig(path, facecolor=t["bg1"])
    plt.close(fig)
    return path


def make_recap_card(title, rows, tiles, path="recap_card.png"):
    """Stat tiles on top, one coloured line per closed trade, and small notes underneath."""
    import textwrap
    plt = _plt()
    colors = {"win": "up", "loss": "down", "flat": "amber", "info": "fg", "dim": "dim"}
    t0 = _theme()
    lines = []
    for text, kind in rows:
        for part in (textwrap.wrap(text, 70) or [""]):
            lines.append((part, kind))
    W = 8.2
    H = 3.1 + 0.36 * len(lines)
    fig, bg, t = _canvas(plt, W, H, t0["cyan"])
    _pill(bg, 0.35, H - 0.72, 1.55, 0.38, t["cyan"], "DAY RECAP", fs=10, t=t)
    bg.text(2.1, H - 0.53, title.split("·")[-1].strip(), color=t["fg"], fontsize=12, fontweight="bold", va="center", family=t["mono"], zorder=5)
    _wordmark(fig, bg, W - 0.4, H - 0.53, t, t["cyan"])
    cw, gap = (W - 0.6 - (len(tiles) - 1) * 0.14) / len(tiles), 0.14
    for k, (lab, val, kind) in enumerate(tiles):
        _card(bg, 0.3 + k * (cw + gap), H - 1.95, cw, 0.95, t, lab, val, "", t[colors.get(kind, "fg")])
    y = H - 2.3
    for text, kind in lines:
        bg.text(0.45, y, text, color=t[colors.get(kind, "fg")], fontsize=9.5 if kind in ("win", "loss", "flat") else 8.6,
                fontweight="bold" if kind in ("win", "loss", "flat") else "normal", va="center", family=t["mono"] if kind in ("win", "loss", "flat") else t["sans"], zorder=5)
        y -= 0.36
    bg.text(W / 2, 0.36, _spaced("OWN YOUR NUMBERS."), color=t["cyan"], fontsize=6.8, ha="center", va="center",
            family=t["mono"], alpha=0.9, zorder=5)
    fig.savefig(path, facecolor=t["bg1"])
    plt.close(fig)
    return path


def send_image_or_text(make_image, text, caption=None):
    """Try to post a picture (caption defaults to the full text). If anything fails, post the plain text."""
    if CHARTS:
        try:
            path = make_image()
            cap = text if caption is None else caption
            if len(cap) <= 1000:
                send_photo(path, cap)
            else:
                send_photo(path)
                send(cap)
            return
        except Exception as e:
            print(f"Image failed, sending text only: {str(e)[:150]}")
    send(text)


def run_test_chart():
    """Posts a clearly labelled TEST chart built from real recent prices, to check the picture pipeline."""
    inst = next(i for i in INSTRUMENTS if i["name"] == "EUR/USD")
    df = add_indicators(fetch_twelvedata(inst["symbol"], LIVE_COUNT, LIVE_INTERVAL, LIVE_MINUTES * 60))
    last = df.iloc[-1]
    entry, atr = float(last.close), float(last.atr)
    sig = build_signal(inst, "TEST", "BUY", entry, entry - 1.5 * atr, last.datetime)
    path = make_signal_chart(sig, df, "test_chart.png")
    send_photo(path, "🧪 <b>TEST CHART</b>: this is NOT a real signal. It only checks that chart images work.")
    print("Test chart sent.")


# ---------- Widening test: more pairs and a longer window for Daily Break ----------
WIDEN_FILE = Path("state/widen.txt")
WIDEN_ENDS = [(960, "until 17:00 Nigeria time (today)"), (1200, "until 21:00 Nigeria time")]


MARKETS_FILE = Path("state/markets.txt")
EXTRA_MARKETS = ["XAG/USD", "BTC/USD", "ETH/USD", "WTI/USD"]   # tested with --markets (silver, bitcoin, ethereum, oil)
SPREADS.update({"XAG/USD": 0.03, "BTC/USD": 20.0, "ETH/USD": 1.5, "WTI/USD": 0.03})   # rough typical spreads


def run_markets():
    insts = [{"name": m, "kind": "forex", "symbol": m} for m in EXTRA_MARKETS]
    run_widen(insts, MARKETS_FILE, "MARKETS TEST: Daily Break on other markets")


def run_widen(insts=None, out_file=None, title="WIDEN TEST: Daily Break"):
    insts = insts or [i for i in INSTRUMENTS if i["kind"] == "forex"]
    out_file = out_file or WIDEN_FILE
    results = {}   # (end_min, pair) -> trades
    days = {}      # pair -> number of trading days in the data
    for inst in insts:
        try:
            time.sleep(8)
            df = fetch_tf(inst["symbol"], 30, TF_PAGES[30])
        except Exception as e:
            print(f"Could not fetch {inst['name']}: {str(e)[:150]}")
            continue
        d = lab_prepare(df, 30)
        days[inst["name"]] = max(1, len(lab_day_groups(d)))
        for end_min, _ in WIDEN_ENDS:
            cands = lab_daily_break(d, end_min=end_min)
            results[(end_min, inst["name"])] = lab_evaluate(inst, d, cands, 30, 1.0, 2.0)
        print(f"{inst['name']}: {len(df)} candles, {days[inst['name']]} days")
    lines = [f"==== {title}, 30-minute entries, TP1 1:1 / TP2 1:2, after spreads ====",
             "'sig/day' = average signals per day on that pair. 'good' = t of 3+, positive early AND late, 30+ trades."]
    for end_min, label in WIDEN_ENDS:
        lines += ["", f"-- Window: {label} --",
                  f"{'pair':8s} {'trades':>6s} {'sig/day':>7s} {'win%':>5s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
        pooled, per_day = [], 0.0
        for name in sorted(days):
            t = results.get((end_min, name), [])
            if not t:
                continue
            n = len(t)
            pooled += t
            per_day += n / days[name]
            early, late = sum(x[1] for x in t if x[0]), sum(x[1] for x in t if not x[0])
            ts = t_score([x[1] for x in t])
            good = "  good" if early > 0 and late > 0 and n >= 30 and ts >= 3 else ""
            lines.append(f"{name:8s} {n:6d} {n / days[name]:7.2f} {100 * sum(1 for x in t if x[2]) / n:5.0f} "
                         f"{sum(x[1] for x in t) / n:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{good}")
        if pooled:
            n = len(pooled)
            early, late = sum(x[1] for x in pooled if x[0]), sum(x[1] for x in pooled if not x[0])
            lines.append(f"{'ALL':8s} {n:6d} {per_day:7.2f} {100 * sum(1 for x in pooled if x[2]) / n:5.0f} "
                         f"{sum(x[1] for x in pooled) / n:+8.2f} {t_score([x[1] for x in pooled]):5.1f} {early:+8.1f} {late:+8.1f}")
    lines += ["", "More pairs mean more signals, but only add a pair if it holds up on its own (positive early and late)."]
    text = "\n".join(lines)
    print(text)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text + "\n")


WIDE_FILE = Path("state/wide.txt")
WIDE_STOPS = [1.5, 2.5, 3.5]                    # stop distance in hourly ATRs
WIDE_TARGETS = [(1.0, 2.0), (1.0, 3.0), (2.0, 3.0)]


def run_wide():
    """Forex only: does a WIDER stop (spread becomes a smaller share) and bigger targets give an edge?"""
    insts = [i for i in INSTRUMENTS if i["kind"] == "forex"]
    rows, by_pair, days = {}, {}, {}
    for inst in insts:
        for minutes in (60, 30):
            try:
                time.sleep(8)
                df = fetch_tf(inst["symbol"], minutes, TF_PAGES[minutes])
            except Exception as e:
                print(f"Could not fetch {inst['name']} {minutes}m: {str(e)[:150]}")
                continue
            d = lab_prepare(df, minutes)
            days[(minutes, inst["name"])] = max(1, len(lab_day_groups(d)))
            for sm in WIDE_STOPS:
                cands = lab_daily_break(d, sl_mult=sm)
                for tgt in WIDE_TARGETS:
                    res = lab_evaluate(inst, d, cands, minutes, *tgt)
                    rows.setdefault((minutes, sm, tgt), []).extend(res)
                    by_pair[(minutes, sm, tgt, inst["name"])] = res
            print(f"{inst['name']} {minutes}m done ({len(df)} candles)")

    def stat(t):
        n = len(t)
        early, late = sum(x[1] for x in t if x[0]), sum(x[1] for x in t if not x[0])
        ts = t_score([x[1] for x in t])
        good = early > 0 and late > 0 and n >= 30 and ts >= 3
        return n, sum(x[1] for x in t) / n, ts, early, late, good

    lines = ["==== WIDE-STOP TEST: Daily Break on the 7 forex pairs, after spreads ====",
             "Stop = number of hourly ATRs. Wider stop = spread is a smaller share of each trade.",
             "'sig/d' = signals per day, all 7 pairs together. 'good' = t of 3+, positive early AND late, 30+ trades.",
             "", f"{'entry':>5s} {'stop':>5s} {'TP1/TP2':>8s} {'trades':>6s} {'sig/d':>5s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
    best = None
    for (minutes, sm, tgt), t in sorted(rows.items()):
        if not t:
            continue
        n, avg, ts, early, late, good = stat(t)
        dpp = [v for (m, _), v in days.items() if m == minutes]
        sigd = n / (sum(dpp) / len(dpp)) if dpp else 0
        lines.append(f"{minutes:>4d}m {sm:>4g}x {tgt[0]:g}/{tgt[1]:g}".ljust(21)
                     + f"{n:6d} {sigd:5.2f} {avg:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{'  good' if good else ''}")
        if best is None or ts > best[0]:
            best = (ts, (minutes, sm, tgt))
    if best:
        minutes, sm, tgt = best[1]
        lines += ["", f"-- PER PAIR for the best pooled row ({minutes}m, stop {sm:g}x, {tgt[0]:g}/{tgt[1]:g}) --",
                  f"{'pair':8s} {'trades':>6s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
        for inst in insts:
            t = by_pair.get((minutes, sm, tgt, inst["name"]), [])
            if t:
                n, avg, ts, early, late, good = stat(t)
                lines.append(f"{inst['name']:8s} {n:6d} {avg:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{'  good' if good else ''}")
    goods = []
    for (minutes, sm, tgt, name), t in by_pair.items():
        if t:
            n, avg, ts, early, late, good = stat(t)
            if good:
                goods.append((ts, minutes, sm, tgt, name, n, avg))
    lines += ["", "-- ANY single pair that passed on its own (best first) --"]
    for ts, minutes, sm, tgt, name, n, avg in sorted(goods, reverse=True)[:12]:
        lines.append(f"{name:8s} {minutes:>3d}m stop {sm:g}x {tgt[0]:g}/{tgt[1]:g}: {n} trades, {avg:+.2f}R each, t {ts:.1f}")
    if not goods:
        lines.append("None. No forex pair passed with a wider stop either.")
    text = "\n".join(lines)
    print(text)
    WIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    WIDE_FILE.write_text(text + "\n")


GOLDMORE_FILE = Path("state/goldmore.txt")


def lab_break_variant(d, ref="prev", start=420, end=960, cont=False, trend="full", sl_mult=1.5):
    """Daily-Break family on one instrument. ref: 'prev' = yesterday's range, 'asia' = today's 00-07 UTC range.
    cont: keep looking after a break that fails the trend filter. trend: 'full' (price vs EMA and slope), 'price' (price vs EMA only), 'off'."""
    out, prev = [], None
    for day, idx in sorted(lab_day_groups(d).items()):
        if ref == "prev":
            rng_idx = prev
        else:
            rng_idx = idx[d["hour"][idx] < 7]
            if len(rng_idx) < 8:
                rng_idx = None
        prev = idx
        if rng_idx is None or len(rng_idx) == 0:
            continue
        hi, lo = d["h"][rng_idx].max(), d["l"][rng_idx].min()
        for i in idx:
            if i >= d["n"] - 1:
                break
            if not (start <= d["mod"][i] <= end):
                continue
            c, ah = d["c"][i], d["atr_h"][i]
            if not ah > 0:
                break
            if c > hi or c < lo:
                side = "BUY" if c > hi else "SELL"
                if trend == "full":
                    ok = lab_trend(d, i, side)
                elif trend == "price":
                    e = d["es"][i]
                    ok = e == e and ((c > e) if side == "BUY" else (c < e))
                else:
                    ok = True
                if ok:
                    out.append((i, side, c - sl_mult * ah if side == "BUY" else c + sl_mult * ah))
                    break
                if not cont:
                    break
    return out


GOLDMORE_VARIANTS = [
    ("A. Keep looking after a counter-trend break", dict(ref="prev", cont=True)),
    ("B. Looser trend filter (price vs trend line only)", dict(ref="prev", cont=True, trend="price")),
    ("C. Longer window (05:00-20:00 UTC)", dict(ref="prev", cont=True, start=300, end=1200)),
    ("D. Asian-range break (a different setup)", dict(ref="asia", cont=True)),
    ("E. Asian-range break, longer window", dict(ref="asia", cont=True, start=300, end=1200)),
]


def run_goldmore():
    inst = next(i for i in INSTRUMENTS if i["name"] == "XAU/USD")
    rows = []
    for minutes in (30, 60):
        time.sleep(8)
        df = fetch_tf(inst["symbol"], minutes, TF_PAGES[minutes])
        d = lab_prepare(df, minutes)
        ndays = max(1, len(lab_day_groups(d)))
        base = lab_break_variant(d)
        base_i = {c[0] for c in base}
        bt = lab_evaluate(inst, d, base, minutes, 1.0, 2.0)
        rows.append((minutes, "BASE: what is live now", len(bt), len(bt) / ndays, bt))
        for label, kw in GOLDMORE_VARIANTS:
            cands = lab_break_variant(d, **kw)
            allr = lab_evaluate(inst, d, cands, minutes, 1.0, 2.0)
            extra = [c for c in cands if c[0] not in base_i]
            ext = lab_evaluate(inst, d, extra, minutes, 1.0, 2.0)
            rows.append((minutes, label + " (all signals)", len(allr), len(allr) / ndays, allr))
            rows.append((minutes, label + " (ONLY the extra ones)", len(ext), len(ext) / ndays, ext))
        print(f"{minutes}m done ({len(df)} candles, {ndays} days)")
    lines = ["==== MORE GOLD SIGNALS (XAU/USD), after spreads, TP1 1:1 / TP2 1:2, stop 1.5x ====",
             "'extra' = signals the live setup does not already give. Those must pass ON THEIR OWN, otherwise they only add noise.",
             "'good' = t of 3+, positive early AND late, 30+ trades. 'per wk' = signals per week (5 trading days).", ""]
    cur = None
    for minutes, label, n, per_day, t in rows:
        if minutes != cur:
            cur = minutes
            lines += [f"-- {minutes}-minute entries --",
                      f"{'setup':58s} {'trades':>6s} {'per wk':>6s} {'R/trade':>8s} {'t':>5s} {'early R':>8s} {'late R':>8s}"]
        if n == 0:
            lines.append(f"{label:58s} {0:6d}")
            continue
        early, late = sum(x[1] for x in t if x[0]), sum(x[1] for x in t if not x[0])
        ts = t_score([x[1] for x in t])
        good = "  good" if early > 0 and late > 0 and n >= 30 and ts >= 3 else ""
        lines.append(f"{label:58s} {n:6d} {per_day * 5:6.1f} {sum(x[1] for x in t) / n:+8.2f} {ts:5.1f} {early:+8.1f} {late:+8.1f}{good}")
        if "ONLY the extra" in label:
            lines.append("")
    text = "\n".join(lines)
    print(text)
    GOLDMORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDMORE_FILE.write_text(text + "\n")


# ---------- EXPERIMENTAL section: Volatility indices (kept completely apart from the main record) ----------
EXP_ENABLED = False   # experimental volatility section removed
EXP_FILE = Path("state/experimental.json")
EXP_INDICES = {"Volatility 75": "R_75", "Volatility 75 (1s)": "1HZ75V", "Volatility 25": "R_25", "Volatility 100": "R_100"}
EXP_WINDOW, EXP_MIN, EXP_PAUSE_R, EXP_RESUME_R = 20, 10, -5.0, -1.0   # auto-pause when the last 20 trades are -5R or worse
EXP_MAX_PER_DAY = 4
EXP_RISK = "0.25% max per trade"


def exp_load():
    try:
        st = json.loads(EXP_FILE.read_text())
    except Exception:
        st = {}
    for k, v in (("open", []), ("closed", []), ("last_signal", {}), ("paused", False), ("announced", False)):
        st.setdefault(k, v)
    return st


def exp_save(st):
    EXP_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXP_FILE.write_text(json.dumps(st, indent=2))


def exp_record_line(st):
    posted = [t for t in st["closed"] if t.get("live", True)]
    w, l, rate, r = record(posted)
    if w + l == 0:
        return "🧪 Experimental record: building (no closed trades yet)"
    return f"🧪 Experimental record: {w}W / {l}L ({rate:.0f}%) | {r:+.1f}R"


def exp_paused(st):
    done = sorted([t for t in st["closed"] if t["result"] in SCORED], key=lambda t: t.get("closed_at", ""))[-EXP_WINDOW:]
    if len(done) < EXP_MIN:
        return False
    total = sum(t["r"] for t in done)
    return (total < EXP_RESUME_R) if st["paused"] else (total < EXP_PAUSE_R)


def exp_resolve(st, candles):
    now, still = datetime.now(timezone.utc), []
    for t in st["open"]:
        df = candles.get(t["pair"])
        if df is None:
            still.append(t)
            continue
        after = df[df["datetime"] > pd.Timestamp(t["candle"])]
        result, _, tp1_hit = walk_trade(t["side"], t["entry"], t["sl"], t["tp1"], t["tp2"], after["high"].values, after["low"].values)
        if result is None and now - datetime.fromisoformat(t["opened_at"]) > timedelta(hours=EXPIRE_HOURS):
            result = "TP1" if tp1_hit else "EXPIRED"
        if result is None:
            if tp1_hit and not t.get("tp1_notified"):
                t["tp1_notified"] = True
                if t.get("live", True):
                    send(f"🧪 <b>Experimental · TP1 hit:</b> {t['side']} {t['pair']}\nClose half and move your stop to entry. The rest runs to TP2.")
            elif t.get("live", True):
                note = progress_note(t, after, "🧪 ")
                if note:
                    send(note)
            still.append(t)
            continue
        t["result"], t["r"], t["closed_at"] = result, trade_r(result, t["tp1_r"], t["tp2_r"]), now.isoformat()
        st["closed"].append(t)
        if t.get("live", True):
            label = {"TP2": "✅ TP2 hit (full target)", "TP1": "🟡 TP1 hit, rest stopped at break-even",
                     "SL": "❌ SL hit", "EXPIRED": "⏱ Expired"}[result]
            send(f"🧪 <b>Experimental</b> · {label}: <b>{t['side']} {t['pair']}</b>\nResult: {t['r']:+.1f}R\n\n{exp_record_line(st)}")
    st["open"] = still


def exp_signal(inst, df):
    """Daily Break for markets that never close: first 30-minute close beyond yesterday's high/low, with the trend."""
    if len(df) < 450:
        return None
    d = lab_prepare(df.reset_index(drop=True), 30)
    cands = lab_vol_break(d, live=True)
    if not cands or cands[-1][0] != d["n"] - 1:
        return None
    if datetime.utcnow() - df["datetime"].iloc[-1].to_pydatetime() > timedelta(hours=2):   # stale data
        return None
    i, side, sl = cands[-1]
    sig = build_signal(inst, "Vol Break", side, float(d["c"][i]), sl, df["datetime"].iloc[-1])
    sig["exp"] = True
    return sig


def format_exp(s, st):
    arrow = "📈" if s["side"] == "BUY" else "📉"
    return (
        f"🧪 <b>{BRAND} EXPERIMENTAL SIGNAL</b>  ·  VOLATILITY INDEX\n\n"
        f"<b>Index:</b> {s['pair']}\n"
        f"<b>Direction:</b> {arrow} {s['side']}\n"
        f"<b>Setup:</b> Daily Break (30-minute close beyond yesterday's high/low, with the trend)\n\n"
        f"<b>Entry:</b> <code>{s['entry']}</code>\n"
        f"<b>Stop Loss:</b> <code>{s['sl']}</code> ({s['risk_pips']} pts)\n"
        f"<b>TP1 (1:{s['tp1_r']:g}):</b> <code>{s['tp1']}</code>\n"
        f"<b>TP2 (1:{s['tp2_r']:g}):</b> <code>{s['tp2']}</code>\n\n"
        f"<b>Risk:</b> {EXP_RISK}\n"
        f"{exp_record_line(st)}\n\n"
        f"⚠️ EXPERIMENTAL: our tests found no reliable edge on these indices yet. This is NOT part of the VIP record. "
        f"Use very small size and treat it as a live test."
    )


EXP_INTRO = (
    f"🧪 <b>{BRAND} EXPERIMENTAL SECTION</b>\n\n"
    "We are starting a separate, clearly labelled test on the Volatility indices (V75, V75 1s, V25, V100). "
    "These are posted with the tag 🧪 EXPERIMENTAL and tracked in their own record.\n\n"
    "• Our back-tests showed NO reliable edge here, so this is a live test, not a recommendation.\n"
    "• It is NOT part of the VIP record (that stays gold only).\n"
    "• Use tiny size (0.25% risk or less). If the last 20 trades are -5R or worse, the section pauses itself."
)


def run_experimental(now):
    if not EXP_ENABLED:
        return
    st = exp_load()
    insts = [{"name": n, "kind": "synthetic", "symbol": sym} for n, sym in EXP_INDICES.items()]
    candles = {}
    for inst in insts:
        try:
            candles[inst["name"]] = fetch_deriv_tf(inst["symbol"], 30, 1)
        except Exception as e:
            print(f"Experimental: could not fetch {inst['name']}: {str(e)[:150]}")
    if not candles:
        return
    exp_resolve(st, candles)
    st["paused"] = exp_paused(st)
    if not st["announced"] and not st["paused"]:
        send(EXP_INTRO)
        st["announced"] = True
    today = now.strftime("%Y-%m-%d")
    posted_today = sum(1 for t in st["open"] + st["closed"] if t.get("live", True) and str(t.get("opened_at", ""))[:10] == today)
    for inst in insts:
        df = candles.get(inst["name"])
        if df is None or any(t["pair"] == inst["name"] for t in st["open"]):
            continue
        sig = exp_signal(inst, df)
        if not sig or st["last_signal"].get(inst["name"]) == sig["candle"]:
            continue
        sig["live"] = (not st["paused"]) and posted_today < EXP_MAX_PER_DAY
        if sig["live"]:
            send_signal(sig, format_exp(sig, st), add_indicators(df))
            posted_today += 1
        else:
            print(f"Experimental shadow trade (not posted): {inst['name']} {sig['side']} @ {sig['entry']}")
        st["open"].append(sig)
        st["last_signal"][inst["name"]] = sig["candle"]
    exp_save(st)


def run_experimental_safe(now):
    try:
        run_experimental(now)
    except Exception as e:   # the experimental section must never stop the main signals
        print(f"Experimental step failed: {str(e)[:200]}")


# ---------- Main ----------
def main():
    if "--test" in sys.argv:
        send("✅ Bot is connected and working.")
        print("Test message sent.")
        return
    if "--backtest" in sys.argv:
        run_backtest()
        return
    if "--sweep" in sys.argv:
        run_sweep()
        return
    if "--ideas" in sys.argv:
        run_ideas()
        return
    if "--markets" in sys.argv:
        run_markets()
        return
    if "--gold" in sys.argv:
        run_gold()
        return
    if "--goldmore" in sys.argv:
        run_goldmore()
        return
    if "--vol" in sys.argv:
        run_vol()
        return
    if "--retest" in sys.argv:
        run_retest()
        return
    if "--more" in sys.argv:
        run_more()
        return
    if "--wide" in sys.argv:
        run_wide()
        return
    if "--widen" in sys.argv:
        run_widen()
        return
    if "--test-chart" in sys.argv:
        run_test_chart()
        return
    if "--news-repost" in sys.argv:   # testing only: ignores the once-a-day rules, so it can post duplicates
        run_news_safe(force=True, repost=True)
        print("News repost done.")
        return
    if "--news" in sys.argv:
        run_news_safe(force=True)
        print("News step done.")
        return
    if "--nfp" in sys.argv:
        run_nfp()
        return
    if "--tf" in sys.argv:
        run_tf()
        return
    if "--lab" in sys.argv:
        run_lab()
        return

    run_news_safe()

    now = datetime.now(timezone.utc)
    weekend = now.weekday() >= 5

    state = load_state()
    retire_removed_instruments(state)
    backtest = load_backtest()
    cuts = cut_combos(backtest)
    if cuts:
        print(f"Removed by backtest ({len(cuts)}): " + ", ".join(sorted(cuts)))

    # an instrument is skipped when every strategy on it was removed (unless a trade is still open)
    still_open = {t["pair"] for t in state["open"]}
    todo = []
    for inst in INSTRUMENTS:
        if weekend and inst["kind"] == "forex":
            continue
        wanted = any(live_ok(s, inst) and combo_key(s, inst["name"]) not in cuts for s in STRATEGIES)
        if wanted or inst["name"] in still_open:
            todo.append(inst)
    if weekend:
        print("Weekend: skipping forex, running synthetic indices only.")
    candles = {}
    first_forex = True
    for inst in todo:
        try:
            if inst["kind"] == "forex":
                if not first_forex:
                    time.sleep(8)  # stay under the free-tier rate limit
                first_forex = False
                df = fetch_twelvedata(inst["symbol"], LIVE_COUNT, LIVE_INTERVAL, LIVE_MINUTES * 60)
            else:
                df = fetch_deriv(inst["symbol"])
            candles[inst["name"]] = add_indicators(df)
        except Exception as e:
            print(f"Could not fetch {inst['name']}: {str(e)[:200]}")

    print(f"Fetched {len(candles)} of {len(todo)} instruments.")
    if todo and not candles:
        print("Nothing could be fetched. Check your API keys and connections.")
        sys.exit(1)

    resolve_open(state, candles)
    save_state(state)
    run_extras_safe(candles, state, now)

    for inst in todo:
        df = candles.get(inst["name"])
        if df is None:
            continue
        for strategy in STRATEGIES:
            key = combo_key(strategy, inst["name"])
            if key in cuts or not live_ok(strategy, inst):
                continue  # removed by the backtest, or not a live pair for this strategy
            ev = blackout_event(inst, now)
            if ev:
                print(f"News pause: no {inst['name']} signals around {ev['ccy']} {ev['title']}")
                continue
            if any(t["pair"] == inst["name"] and t["strategy"] == strategy for t in state["open"]):
                continue  # one open trade per instrument per strategy
            sig = STRATEGY_FUNCS[strategy](inst, df)
            if sig and strategy == "Daily Break":
                slot = signal_slot(sig["candle"])
                if slot is None or slot_used(state, slot, sig["candle"]):
                    print(f"Skipped {inst['name']} {sig['side']}: {'outside the signal slots' if slot is None else slot + ' signal already sent today'}")
                    continue
            if sig and state["last_signal"].get(key) != sig["candle"]:
                sig["live"] = combo_active(state, key, backtest)
                if sig["live"]:
                    send_signal(sig, format_signal(sig, winrate_30d_line(state)), df)
                else:
                    print(f"Shadow trade (paused, not posted): {key} {sig['side']} @ {sig['entry']}")
                state["open"].append(sig)
                state["last_signal"][key] = sig["candle"]
                save_state(state)

    # keep every combination's pause state fresh, even when it had no new signal
    for strategy in STRATEGIES:
        for inst in INSTRUMENTS:
            key = combo_key(strategy, inst["name"])
            if key not in cuts:
                combo_active(state, key, backtest)
    save_state(state)
    print_leaderboard(state, backtest, cuts)
    run_experimental_safe(now)
    print("Run complete.")


if __name__ == "__main__":
    main()
