"""Fast-scalp signal generator for gold (XAU/USD). Ranked setups (not random), 1:2 or 1:3 targets, about 60 a day,
split into morning / afternoon / evening sessions. Posts to your channel (TELEGRAM_CHAT_ID), or to a separate chat if you add a SCALP_CHAT_ID secret.
Run every 30 minutes on weekdays: python scalp_bot.py"""
import json, os, random, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import signal_bot as b

SYMBOL = "XAU/USD"
STATE = Path("state/scalp.json")
PER_SESSION = 20   # upper limit per session (one signal every 15 minutes gives about 18-20)
SESSIONS = {"MORNING":   (6 * 60, 10 * 60 + 30),
            "AFTERNOON": (11 * 60, 15 * 60 + 30),
            "EVENING":   (16 * 60, 21 * 60)}
RATIOS = [2.0, 3.0]
MAX_PER_RUN = 1    # most signals one run posts (1 = a skipped run is NOT made up with extra signals)
MAX_WAIT_SEC = 330  # a run waits at most this long for its slot (the timer wakes the bot every 5 minutes)
CADENCE_MIN = 15   # a new signal every this many minutes (None = old pacing)
GAP_MIN, GAP_MAX = 20, 40      # seconds between signals in one run (short, so no signal goes stale while waiting)
EXPIRE_MIN = 90
PRE_SIGNAL = True    # before every new signal: tell you to close trades that are still open (or a short heads-up)
RANDOM_MODE = True   # True = random BUY/SELL at the live price so the session always gets its signals. False = ranked setups
SHOW_SESSION_SUMMARY = False
RECAP_AFTER_MIN = 21 * 60 + 10
SPREAD = 0.30
BE_TRIGGER_R = 0.5   # send 'Set BE' when price is this far in profit, as a share of the stop distance (0.5 = half way to a 1R win)
SESSION_TEXT = {   # opening and closing messages (times are Nigeria time)
    "MORNING": (
        "☀️🌅 <b>Good morning traders!</b>\n"
        "Rise and shine, it's another day to go after the money again 💰\n\n"
        "🔥 <b>PROFITS SANTA AI · MORNING SESSION is LIVE</b>\n"
        "🕖 07:00 – 11:30 (Nigeria time)\n"
        "📊 Gold (XAU/USD) scalps · 1:2 and 1:3\n\n"
        "Charts open? Coffee ready? 😎 Let's get it. Ready? 🚀",
        "☕ <b>Morning session done!</b>\n"
        "Let's take a 30-minute break 😮‍💨 Stretch, drink some water, reset your mind.\n\n"
        "🔔 Signals will resume shortly. The <b>Afternoon Session</b> opens at 12:00. Stay ready! 🔥"),
    "AFTERNOON": (
        "🌤💪 <b>Good afternoon traders!</b>\n"
        "Break's over and we're back for round two 💰\n\n"
        "🔥 <b>PROFITS SANTA AI · AFTERNOON SESSION is LIVE</b>\n"
        "🕛 12:00 – 16:30 (Nigeria time)\n"
        "📊 Gold (XAU/USD) scalps · 1:2 and 1:3\n\n"
        "Eyes on the charts, stay disciplined. Ready? 🚀",
        "🌇 <b>Afternoon session done!</b>\n"
        "Time for another 30-minute break 😮‍💨 Breathe, hydrate, reset.\n\n"
        "🔔 Signals will resume shortly. The <b>Evening Session</b> opens at 17:00. Stay ready! 🔥"),
    "EVENING": (
        "🌙✨ <b>Good evening traders!</b>\n"
        "Final round of the day, let's finish strong 💰\n\n"
        "🔥 <b>PROFITS SANTA AI · EVENING SESSION is LIVE</b>\n"
        "🕔 17:00 – 22:00 (Nigeria time)\n"
        "📊 Gold (XAU/USD) scalps · 1:2 and 1:3\n\n"
        "Last session of the day. Stay sharp! 🚀",
        "🌃 <b>That's a wrap, traders!</b> 🏁\n"
        "The evening session is closed. Day recap coming up next 📊\n\n"
        "Rest well, recharge, and we go again tomorrow at 07:00 ☀️🔥"),
}


def load():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {}
    s.setdefault("day", "")
    s.setdefault("count", {})
    s.setdefault("open", [])
    s.setdefault("closed", [])
    s.setdefault("summary_sent", [])
    s.setdefault("used", [])
    s.setdefault("recap_days", [])
    return s


def save(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    s["closed"] = s["closed"][-600:]
    s["used"] = s.get("used", [])[-400:]
    STATE.write_text(json.dumps(s, indent=1))


def session_of(now):
    m = now.hour * 60 + now.minute
    for name, (a, c) in SESSIONS.items():
        if a <= m < c:
            return name, a, c
    return None, 0, 0


def fetch():
    df = b.fetch_twelvedata(SYMBOL, 120, "5min", 300)
    return b.only_closed(df, 300)


def live_price(fallback):
    try:
        r = b.requests.get("https://api.twelvedata.com/price", params={"symbol": SYMBOL, "apikey": b.TD_KEY}, timeout=20)
        return float(r.json()["price"])
    except Exception as e:
        print(f"Live price failed, using the last candle close: {str(e)[:100]}")
        return fallback


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def _rsi(s, n=7):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-9))


def add_indicators(df):
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    if "volume" in df.columns and df["volume"].notna().any() and df["volume"].sum() > 0:
        v = df["volume"].astype(float)
    else:
        v = (h - l).abs().replace(0, 1e-9)

    df["ema9"]  = c.ewm(span=9,  adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["rsi7"]  = _rsi(c, 7)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_med"] = df["atr"].rolling(200, min_periods=30).median()

    tp = (h + l + c) / 3
    df["vwap"] = (tp * v).rolling(78, min_periods=20).sum() / v.rolling(78, min_periods=20).sum().replace(0, 1e-9)
    df["vma"]  = v.rolling(20, min_periods=5).mean()
    return df


def score_at(df, i, side):
    if i < 55:
        return 0.0, []
    r, prev = df.iloc[i], df.iloc[i - 1]
    sign = 1 if side == "BUY" else -1
    s, why = 0.0, []

    if sign * (r.close - r.ema50) > 0:
        s += 10
        why.append("trend")
    if sign * (r.ema9 - r.ema21) > 0:
        s += 10
        why.append("stack")
    if sign * (r.ema50 - df.ema50.iloc[i - 5]) > 0:
        s += 10
        why.append("slope")

    if side == "BUY" and r.rsi7 > 55:
        s += 10
        why.append(f"rsi{r.rsi7:.0f}")
    if side == "SELL" and r.rsi7 < 45:
        s += 10
        why.append(f"rsi{r.rsi7:.0f}")
    if sign * (r.ema9 - prev.ema9) > 0:
        s += 10
        why.append("push")

    if side == "BUY" and r.close > df.high.iloc[i - 20:i].max():
        s += 15
        why.append("brk")
    if side == "SELL" and r.close < df.low.iloc[i - 20:i].min():
        s += 15
        why.append("brk")
    if sign * (r.close - r.vwap) > 0 and sign * (prev.close - prev.vwap) <= 0:
        s += 10
        why.append("vwap")

    if pd.notna(r.atr) and pd.notna(r.atr_med) and r.atr > r.atr_med:
        s += 15
        why.append("atr")
    if pd.notna(r.atr) and r.atr and r.atr < SPREAD * 3:
        s -= 20
        why.append("tight")

    if pd.notna(r.vma) and pd.notna(prev.vma) and r.vma > 1.2 * prev.vma:
        s += 10
        why.append("vol")

    return s, why


def pick_signals(df, s, n, max_age_min=15, lookback=6, skip=frozenset()):
    used = set(s.get("used", [])) | set(skip)
    now = datetime.now(timezone.utc)
    best = {}
    for i in range(max(56, len(df) - lookback), len(df)):
        ts = pd.Timestamp(df.datetime.iloc[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if (now - ts.to_pydatetime()).total_seconds() > max_age_min * 60:
            continue
        key = ts.isoformat()
        if key in used:
            continue
        bs, bside, bwhy = 0.0, None, []
        for side in ("BUY", "SELL"):
            sc, why = score_at(df, i, side)
            if sc > bs:
                bs, bside, bwhy = sc, side, why
        if bside and bs > 0:
            best[i] = (bs, bside, i, key, bwhy)
    return sorted(best.values(), key=lambda x: -x[0])[:n]


def make_trade(df, session, number, side, trigger_close, score):
    entry = live_price(trigger_close)
    sign = 1 if side == "BUY" else -1
    risk = max(round(atr(df) * 1.0, 2), 1.5)
    if not RANDOM_MODE and abs(entry - trigger_close) > 0.5 * risk:    # price already ran away from the setup: do not post it
        print(f"Skipped {side}: price moved {abs(entry - trigger_close):.2f} since the setup")
        return None
    ratio = 3.0 if score >= 55 else random.choice(RATIOS)
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    started = now_naive - timedelta(minutes=now_naive.minute % 5)
    return {"n": number, "session": session, "side": side, "entry": round(entry, 2),
            "sl": round(entry - sign * risk, 2), "tp": round(entry + sign * risk * ratio, 2),
            "ratio": ratio, "risk": risk, "score": round(score),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "candle": started.isoformat()}


def make_scalp_image_basic(tr, path="scalp.png"):
    plt = b._plt()
    from matplotlib.patches import FancyBboxPatch, Circle
    W, H = 6.4, 5.4
    BG, PANEL, EDGE, FG, DIM = "#08090b", "#111317", "#2a2d33", "#ffffff", "#9aa0a8"
    GREEN, RED = "#1ed760", "#ff3b4e"
    col = GREEN if tr["side"] == "BUY" else RED
    mono, sans = "DejaVu Sans Mono", "DejaVu Sans"
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, fc, ec="none", lw=1.0, r=0.12, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

    box(0.25, 0.25, W - 0.5, H - 0.5, PANEL, EDGE, 1.2, 0.2, 1)
    ax.text(0.55, H - 0.7, "PROFITS SANTA AI", color=FG, fontsize=17, fontweight="bold", va="center", family=sans, zorder=5)
    ax.plot([0.55, 0.85, 1.0, 1.25], [H - 0.98, H - 0.98, H - 0.9, H - 0.86], color=FG, lw=1.6, zorder=5)
    ax.text(W - 0.55, H - 0.7, f"#{tr['n']:02d} / 60", color=DIM, fontsize=10, fontweight="bold", ha="right", va="center", family=mono, zorder=5)
    ax.text(W - 0.55, H - 1.0, f"{tr['session']} SESSION", color=DIM, fontsize=7.5, ha="right", va="center", family=mono, zorder=5)
    ax.plot([0.5, W - 0.5], [H - 1.25, H - 1.25], color=EDGE, lw=1, zorder=3)

    rx, ry = 4.25, H - 3.2
    ax.plot([rx + 0.8, rx + 0.8], [ry + 1.55, ry + 1.85], color=FG, lw=1.8, zorder=4)
    ax.add_patch(Circle((rx + 0.8, ry + 1.92), 0.07, fc=FG, ec="none", zorder=4))
    box(rx - 0.12, ry + 0.5, 0.12, 0.45, "#1d2026", FG, 1.0, 0.04, 3)
    box(rx + 1.6, ry + 0.5, 0.12, 0.45, "#1d2026", FG, 1.0, 0.04, 3)
    box(rx, ry, 1.6, 1.55, "#15181d", FG, 1.6, 0.28, 3)
    box(rx + 0.12, ry + 0.52, 1.36, 0.82, "#050607", EDGE, 1.0, 0.18, 3.2)
    for ex in (rx + 0.28, rx + 0.92):
        box(ex, ry + 0.78, 0.4, 0.3, FG, "none", 1, 0.1, 4)
    for k in range(5):
        box(rx + 0.32 + k * 0.2, ry + 0.15, 0.13, 0.2, DIM if k % 2 else FG, "none", 1, 0.03, 4)

    arrow = "\u25b2" if tr["side"] == "BUY" else "\u25bc"
    box(0.55, H - 2.5, 3.3, 0.95, col, "none", 1, 0.2, 4)
    ax.text(2.2, H - 2.02, f"{arrow}  {tr['side']}", color="#ffffff" if tr["side"] == "SELL" else "#04120a", fontsize=29,
            fontweight="bold", ha="center", va="center", family=sans, zorder=6)
    ax.text(0.57, H - 2.8, "XAU/USD  \u00b7  GOLD SCALP", color=FG, fontsize=9, fontweight="bold", va="center", family=mono, zorder=5)
    box(2.35, H - 3.05, 1.5, 0.36, "none", FG, 1.2, 0.18, 4)
    ax.text(3.1, H - 2.87, f"RR 1:{tr['ratio']:g}", color=FG, fontsize=11, fontweight="bold", ha="center", va="center", family=sans, zorder=6)

    risk = tr["risk"]
    cards = [("ENTRY", f"{tr['entry']:.2f}", "market price"),
             ("STOP LOSS", f"{tr['sl']:.2f}", f"-${risk:.2f}"),
             ("TAKE PROFIT", f"{tr['tp']:.2f}", f"+${risk * tr['ratio']:.2f}")]
    cw, gap = (W - 1.1 - 2 * 0.14) / 3, 0.14
    for k, (lab, val, sub) in enumerate(cards):
        x = 0.55 + k * (cw + gap)
        box(x, 0.75, cw, 1.35, "#0c0e11", EDGE, 1.2, 0.12, 3)
        ax.text(x + 0.14, 1.88, lab, color=DIM, fontsize=7.5, fontweight="bold", va="center", family=mono, zorder=5)
        ax.text(x + 0.14, 1.4, val, color=FG, fontsize=15, fontweight="bold", va="center", family=mono, zorder=5)
        ax.text(x + 0.14, 0.98, sub, color=DIM, fontsize=8, va="center", family=mono, zorder=5)
    ax.text(W / 2, 0.45, "R O C K   A I   \u00b7   E X E C U T E   F A S T", color=DIM, fontsize=6.5, ha="center", va="center", family=mono, zorder=5)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def make_scalp_image(tr, path="scalp.png"):
    try:
        import friend_card
        return friend_card.render(tr, path)
    except Exception as e:
        print(f"Main card failed, using the plain one: {str(e)[:150]}")
        return make_scalp_image_basic(tr, path)


def caption(tr):
    return (f"{'🟢' if tr['side'] == 'BUY' else '🔴'} <b>{tr['side']} XAU/USD</b>\n"
            f"<b>Entry</b> <code>{tr['entry']:.2f}</code>\n"
            f"<b>SL</b> <code>{tr['sl']:.2f}</code>\n"
            f"<b>TP</b> <code>{tr['tp']:.2f}</code>  (1:{tr['ratio']:g})\n\n"
            f"👆 <i>Tap on any price number to copy</i>")


def resolve(s, df, force_close=False):
    """Follow every open trade candle by candle. Once price moves BE_TRIGGER_R x the risk our way we send a
    'Set BE' alert; if price then comes back to the entry the trade closes as BE (0R, protected, a partial win)."""
    now = datetime.now(timezone.utc)
    done = []
    dfdt = pd.to_datetime(df["datetime"], utc=True)
    for tr in list(s["open"]):
        candle_ts = pd.Timestamp(tr["candle"])
        if candle_ts.tzinfo is None:
            candle_ts = candle_ts.tz_localize("UTC")
        after = df[dfdt > candle_ts]
        buy = tr["side"] == "BUY"
        sign = 1 if buy else -1
        trigger = tr["entry"] + sign * BE_TRIGGER_R * tr["risk"]
        res, be_on = None, False
        for hi, lo in zip(after["high"].values, after["low"].values):
            if not be_on:
                if (lo <= tr["sl"]) if buy else (hi >= tr["sl"]):
                    res = "SL"
                    break
                if (hi >= tr["tp"]) if buy else (lo <= tr["tp"]):
                    res = "TP"
                    break
                if (hi >= trigger) if buy else (lo <= trigger):
                    be_on = True
                    if (lo <= tr["entry"]) if buy else (hi >= tr["entry"]):   # same candle came back: assume the worst
                        res = "BE"
                        break
            else:
                if (lo <= tr["entry"]) if buy else (hi >= tr["entry"]):
                    res = "BE"
                    break
                if (hi >= tr["tp"]) if buy else (lo <= tr["tp"]):
                    res = "TP"
                    break
        if be_on and res is None and not tr.get("be_notified"):
            tr["be_notified"] = True
            b.send(f"🛡 <b>Set BE now</b> · {tr['side']} XAU/USD\n"
                   f"Price has moved in your favour. Move your stop loss to your entry:\n"
                   f"<code>{tr['entry']:.2f}</code>\n\n"
                   f"👆 <i>Tap on the price number to copy</i>")
        if res is None and (force_close or (now - datetime.fromisoformat(tr["opened_at"])).total_seconds() > EXPIRE_MIN * 60):
            last = float(df["close"].iloc[-1])
            floor = 0.0 if (be_on or tr.get("be_notified")) else -1.0     # a protected trade cannot finish below 0
            r = max(min(sign * (last - tr["entry"]) / tr["risk"], tr["ratio"]), floor)
            tr["result"], tr["r"] = "TIMEOUT", round(r, 2)
        elif res:
            tr["result"], tr["r"] = res, {"TP": tr["ratio"], "SL": -1.0, "BE": 0.0}[res]
        else:
            continue
        tr["closed_at"] = now.isoformat()
        s["open"].remove(tr)
        s["closed"].append(tr)
        done.append(tr)
    return done


DEDUPE_MIN = 20      # look back this many minutes for a similar signal
DEDUPE_R = 0.7       # a same-direction signal must be at least this many "risks" (about $1+) away from the earlier one


def too_close(s, tr):
    """True if a same-direction signal was sent recently at nearly the same price (no near-duplicate signals)."""
    now = datetime.now(timezone.utc)
    for t in s["open"] + s["closed"][-40:]:
        if t["side"] != tr["side"]:
            continue
        try:
            age = (now - datetime.fromisoformat(t["opened_at"])).total_seconds() / 60
        except Exception:
            continue
        if age <= DEDUPE_MIN and abs(t["entry"] - tr["entry"]) < DEDUPE_R * tr["risk"]:
            return True
    return False


def pre_signal(s, df):
    """Goes out right before every new signal. Any trade still open (no TP, SL or BE yet) is closed at the live price and
    the message tells you to close it; if nothing is open, a short heads-up is sent instead."""
    now = datetime.now(timezone.utc)
    live = live_price(float(df["close"].iloc[-1]))
    lines = []
    for tr in list(s["open"]):
        sign = 1 if tr["side"] == "BUY" else -1
        floor = 0.0 if tr.get("be_notified") else -1.0
        r = max(min(sign * (live - tr["entry"]) / tr["risk"], tr["ratio"]), floor)
        tr["result"], tr["r"], tr["closed_at"] = "CLOSE", round(r, 2), now.isoformat()
        s["open"].remove(tr)
        s["closed"].append(tr)
        lines.append(f"\U0001F514 <b>Close #{tr['n']} {tr['side']}</b> now")
    if lines:
        return ("\u23F0 <b>Close your open trade before the next signal</b>\n" + "\n".join(lines) +
                f"\nMarket price: <code>{live:.2f}</code>\n\n\U0001F446 <i>Tap on the price number to copy</i>")
    return "\u26A1 <b>New signal coming up...</b> get ready"


def random_side(s):
    """Random BUY/SELL, but if the last signal went out less than 3 minutes ago, take the other direction (no twin signals)."""
    side = random.choice(("BUY", "SELL"))
    last = (s["open"] + s["closed"][-5:])
    last = max(last, key=lambda t: t["opened_at"], default=None)
    if last:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last["opened_at"])).total_seconds() / 60
        if age < 3:
            side = "SELL" if last["side"] == "BUY" else "BUY"
    return side


def results_text(done):
    if not done:
        return None
    lines = []
    for t in done:
        if t["result"] == "TP":
            lines.append(f"✅ #{t['n']} {t['side']} TP hit {t['r']:+.1f}R")
        elif t["result"] == "SL":
            print(f"(not posted) #{t['n']} {t['side']} SL hit {t['r']:+.1f}R")   # still recorded in the log, just not sent to Telegram
            continue
        elif t["result"] == "BE":
            lines.append(f"🛡 #{t['n']} {t['side']} hit BE · partial win, trade protected (0R)")
        else:
            if t["r"] < 0:
                print(f"(not posted) #{t['n']} {t['side']} closed {t['r']:+.1f}R")
                continue
            lines.append(f"⏱ #{t['n']} {t['side']} closed {t['r']:+.1f}R")
    if not lines:
        return None
    return "📋 <b>Results</b>\n" + "\n".join(lines)


def session_summary(s, name, day):
    ts = [t for t in s["closed"] if t.get("day") == day and t["session"] == name]
    if not ts:
        return None
    w = sum(1 for t in ts if t["r"] > 0)
    l = len(ts) - w
    return (f"📊 <b>{name} session closed</b>\nTrades: {len(ts)} · Wins {w} · Losses {l} ({w / len(ts) * 100:.0f}%)\n"
            f"Total: {sum(t['r'] for t in ts):+.1f}R")


def _tg_send(text):
    r = b.requests.post(f"https://api.telegram.org/bot{b.TG_TOKEN}/sendMessage",
                        json={"chat_id": b.TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=30)
    if not r.ok:
        print(f"TELEGRAM ERROR (message): {r.status_code} {r.text[:300]}")   # says exactly what is wrong, e.g. 'chat not found'
    r.raise_for_status()


def _tg_photo(path, caption=""):
    with open(path, "rb") as f:
        r = b.requests.post(f"https://api.telegram.org/bot{b.TG_TOKEN}/sendPhoto",
                            data={"chat_id": b.TG_CHAT, "caption": caption, "parse_mode": "HTML"}, files={"photo": f}, timeout=60)
    if not r.ok:
        print(f"TELEGRAM ERROR (photo): {r.status_code} {r.text[:300]}")
    r.raise_for_status()


def main():
    b.send, b.send_photo = _tg_send, _tg_photo
    chat = os.environ.get("SCALP_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not chat:
        print("No chat id found. Check the TELEGRAM_CHAT_ID secret.")
        sys.exit(1)
    b.TG_CHAT = chat
    b.TD_KEY = b.TD_KEY or os.environ.get("TD_KEY", "")   # accept the key from either secret name
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        print("Weekend: gold is closed.")
        return
    s = load()
    day = now.strftime("%Y-%m-%d")
    if s["day"] != day:
        s["day"], s["count"] = day, {}
        s["used"] = []
    df = fetch()
    for t in s["open"]:
        t.setdefault("day", t["opened_at"][:10])
    force = now.hour * 60 + now.minute >= RECAP_AFTER_MIN
    done = resolve(s, df, force_close=force)
    txt = results_text(done)
    if txt:
        b.send(txt)
    if force and s["count"] and day not in s.setdefault("recap_days", []):
        wins = [t for t in s["closed"] if t.get("day") == day and t["r"] > 0]
        be_n = sum(1 for t in s["closed"] if t.get("day") == day and t["result"] == "BE")
        label = (now + timedelta(hours=1)).strftime("%a %d %b")
        cap = f"\U0001F3C6 <b>PROFITS SANTA AI DAY RECAP</b> \u00b7 {label}\n{len(wins)} winning trades today"
        try:
            import friend_recap
            b.send_photo(friend_recap.render_recap(label, wins, be_count=be_n), cap)
        except Exception as e:
            print(f"Recap card failed, sending text: {str(e)[:150]}")
            b.send(cap)
        s["recap_days"].append(day)
    session, a, c = session_of(now)
    greeted = s.setdefault("greeted", [])
    if session and f"{day}:{session}:open" not in greeted:        # opening message, once per session
        b.send(SESSION_TEXT[session][0])
        greeted.append(f"{day}:{session}:open")
    for name, (sa, sc) in SESSIONS.items():                       # closing message, once, only for sessions we opened
        if now.hour * 60 + now.minute >= sc and f"{day}:{name}:open" in greeted and f"{day}:{name}:close" not in greeted:
            msg = SESSION_TEXT[name][1]
            if name == "EVENING" and now.weekday() == 4:          # Friday: see you Monday
                msg = msg.replace("we go again tomorrow at 07:00", "enjoy your weekend and we go again on Monday at 07:00")
            b.send(msg)
            greeted.append(f"{day}:{name}:close")
    s["greeted"] = greeted[-30:]
    for name, (sa, sc) in SESSIONS.items():
        key = f"{day}:{name}"
        if SHOW_SESSION_SUMMARY and now.hour * 60 + now.minute >= sc and key not in s["summary_sent"] and s["count"].get(name):
            m = session_summary(s, name, day)
            if m and not [t for t in s["open"] if t["session"] == name]:
                b.send(m)
                s["summary_sent"].append(key)
    if session:
        have = s["count"].get(session, 0)
        mins_in = now.hour * 60 + now.minute - a
        should = int(PER_SESSION * min(1.0, (mins_in + 30) / (c - a - 30)))
        todo = max(0, min(should - have, MAX_PER_RUN, PER_SESSION - have))
        if CADENCE_MIN:
            session_start = now.replace(hour=a // 60, minute=a % 60, second=0, microsecond=0)
            lp = s.get("last_post")
            lp = datetime.fromisoformat(lp) if lp else None
            todo = 0
            if have < PER_SESSION:
                if lp is None or lp < session_start:
                    todo = 1                                   # first signal of the session
                else:
                    wait = (lp + timedelta(minutes=CADENCE_MIN) - now).total_seconds()
                    if wait <= MAX_WAIT_SEC:                   # next slot is due now or within a few minutes: wait for it
                        if wait > 0:
                            print(f"Waiting {wait:.0f}s for the next signal slot")
                            time.sleep(wait)
                        todo = 1
        if todo > 0:
            df = add_indicators(df)
            # Fill the quota: first the freshest setups (last 30 min, best score first), then, if we are still
            # short, older ones (up to 2 hours). A setup whose price has already run away is skipped, and the next one is tried.
            tier1 = pick_signals(df, s, 60, 30, 12)
            tier2 = pick_signals(df, s, 60, 120, 24, skip={p[3] for p in tier1})
            picks = tier1 + tier2
            if RANDOM_MODE:
                picks = [(0.0, random.choice(("BUY", "SELL")), len(df) - 1, f"rnd{time.time():.0f}-{k}", ["random"]) for k in range(todo)]
            trig = {k: float(df.close.iloc[i]) for (_, _, i, k, _) in picks}   # setup prices, taken before any refetch
            if not picks:
                print(f"{session}: no scored candles at all, nothing to post")
            posted = 0
            for (sc, side, i, key, why) in picks:
                if posted >= todo:
                    break
                if posted:
                    time.sleep(random.randint(GAP_MIN, GAP_MAX))
                    df = add_indicators(fetch())
                    resolve_msg = results_text(resolve(s, df))
                    if resolve_msg:
                        b.send(resolve_msg)
                if PRE_SIGNAL:
                    b.send(pre_signal(s, df))
                    time.sleep(3)
                n = sum(s["count"].values()) + 1
                if RANDOM_MODE:
                    side = random_side(s)
                tr = make_trade(df, session, n, side, trig[key], sc)
                if tr is None:
                    s.setdefault("used", []).append(key)
                    continue
                if not RANDOM_MODE and too_close(s, tr):
                    print(f"skipped #{n} {side}: too close to a signal already sent")
                    s.setdefault("used", []).append(key)
                    continue
                tr["day"] = day
                b.send_image_or_text(lambda: make_scalp_image(tr), caption(tr))
                s["open"].append(tr)
                s["count"][session] = s["count"].get(session, 0) + 1
                s.setdefault("used", []).append(key)
                posted += 1
                s["last_post"] = datetime.now(timezone.utc).isoformat()
                save(s)
            print(f"{session}: posted {posted} of {todo} wanted this run")
    save(s)
    print("Scalp run done:", s["count"], "open:", len(s["open"]))


if __name__ == "__main__":
    main()
