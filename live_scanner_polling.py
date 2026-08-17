"""
Live Scanner - Bank Nifty Stocks
==================================
Run this AFTER the morning prep script, once market opens (9:15 AM).

What it does:
1. Loads today's prep JSON (S/R zones + RVOL baseline curve per stock).
2. Connects to Upstox Market Data Feed V3 WebSocket, subscribes to the
   12 Bank Nifty stocks in "full" mode (needed for LTP + volume).
3. Maintains live per-symbol state: cumulative volume, cumulative
   price*volume (for VWAP), last price.
4. Every SCORING_INTERVAL_SEC, evaluates the rule per stock:
     a) price within X% of a support/resistance zone
     b) RVOL (live cumulative volume vs the time-matched baseline) above
        threshold
     c) classifies into one of four patterns based on VWAP position -
        ALL FOUR are surfaced, not just the "confirming" two, since heavy
        volume against the expected direction is just as worth flagging
        as confirming volume:
          support + above VWAP    -> support_bounce (bullish)
          support + below VWAP    -> support_breakdown_warning (bearish)
          resistance + above VWAP -> resistance_breakout (bullish)
          resistance + below VWAP -> resistance_rejection_warning (bearish)
     d) order-book aggression (tbq/tsq imbalance from the depth feed) as a
        second confirming factor - resting buy quantity vs sell quantity
        across the top depth levels. Agreement with the pattern's bias
        boosts the score; disagreement (e.g. selling pressure showing up
        on a "support_bounce") discounts it as a caution flag. This is a
        resting-order proxy for aggression, not true tick-by-tick
        trade-initiator data, which Upstox's API doesn't expose.
     d) OI confirmation (via each stock's current-month futures contract):
        classifies the move as fresh_long / short_covering / fresh_short /
        long_unwinding, and weights the score accordingly - fresh buildup
        in the expected direction is boosted, covering/unwinding is
        discounted since it's a weaker, less committed move.
     e) daily trend regime (EMA50/200 crossover, RSI(14) zone, MACD state -
        computed once at prep time from daily candles, not live) as a
        fifth confirming/conflicting weight - agreeing with the intraday
        pattern's bias boosts the score, fighting the daily trend (e.g. a
        bullish pattern while EMA50 < EMA200) discounts it. See
        SymbolState.trend_weight() for the exact logic.
   and prints a ranked candidate list.
5. Writes to eod_logs/ CONTINUOUSLY throughout the session (not just at
   the end): eod_summary_<date>.csv is overwritten with fresh snapshot
   values every scoring cycle, and eod_candidates_<date>.csv gets each
   candidate appended the moment it's flagged. This means the files on
   disk are always current - nothing depends on a clean shutdown, so a
   Ctrl+C or a crash can't lose the day's data.

IMPORTANT: This is the REST-POLLING version, built after extensive live
testing today found the WebSocket connection opened successfully but
never actually delivered tick data (confirmed via message-count debug
logging, small-subscription tests, and network isolation tests) despite
REST calls to the same Upstox account working reliably throughout the
same session. This version fetches live quotes via a single batched REST
call (Upstox's Full Market Quotes endpoint, up to 500 instruments per
call) every POLL_INTERVAL_SEC seconds instead of a persistent WebSocket
stream. Trade-off: ~10s latency instead of true push-based real-time
updates - but it actually works, which the WebSocket path did not today.

No protobuf/.proto file needed for this version at all - only requests.

Install deps:
    pip install requests

Notes carried over from prior sessions:
- File paths anchored via os.path.dirname(os.path.abspath(__file__)).
- config.txt keeps ACCESS_TOKEN out of source code.
"""

import os
import json
import csv
import glob
import sys
import time
import threading
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

# Always use IST explicitly for market-hours/time-bucket logic, regardless
# of the machine's system timezone - see streamlit_scanner_banknifty.py
# for the full explanation (this bit us on Streamlit Cloud, which runs
# UTC; harmless here if your PC's clock is already IST, but keeping both
# scripts consistent and correct regardless of where this ever runs).
IST = ZoneInfo("Asia/Kolkata")

import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.txt")
PREP_DIR = os.path.join(SCRIPT_DIR, "prep_output")
EOD_LOG_DIR = os.path.join(SCRIPT_DIR, "eod_logs")


SCORING_INTERVAL_SEC = 60          # how often to re-evaluate the rule
NEAR_ZONE_PCT = 0.75               # consider "near" S/R if within this % of a zone
RVOL_THRESHOLD = 1.5               # live cum volume must be >= 1.5x baseline
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def load_config():
    config = {}
    with open(CONFIG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            config[key.strip()] = val.strip()
    return config


def get_access_token(config):
    return config.get("ACCESS_TOKEN") or config.get("UPSTOX_ACCESS_TOKEN")


def load_today_prep():
    """Loads the most recent prep JSON (falls back to latest available if
    today's file isn't there yet)."""
    pattern = os.path.join(PREP_DIR, "banknifty_prep_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No prep files found in {PREP_DIR}. Run morning_prep_banknifty.py first."
        )
    latest = files[-1]
    print(f"Loading prep data from: {latest}")
    with open(latest, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LIVE STATE
# ---------------------------------------------------------------------------
class SymbolState:
    """Tracks running VWAP + cumulative volume + OI for one symbol during
    the live session."""
    def __init__(self, symbol, instrument_key, futures_instrument_key, sr_data, rvol_baseline, trend_indicators=None, prev_close=None):
        self.symbol = symbol
        self.instrument_key = instrument_key
        self.futures_instrument_key = futures_instrument_key
        self.resistance_zones = sr_data.get("resistance_zones", [])
        self.support_zones = sr_data.get("support_zones", [])
        self.rvol_baseline = rvol_baseline  # dict: "HH:MM" -> avg cum volume
        # Daily trend regime, computed once at prep time (not live) -
        # EMA50/200 crossover, RSI(14) zone, MACD(12,26,9) state.
        self.trend_indicators = trend_indicators or {}
        # Previous close matching whichever instrument live LTP comes from
        # (futures, normally - see prep script's prev_close resolution).
        self.prev_close = prev_close

        self.last_price = None
        self.atp = None
        self.cum_volume = 0
        self.oi_open = None     # first OI value seen this session
        self.oi_current = None
        self.tbq = None         # total buy quantity (resting bids across depth)
        self.tsq = None         # total sell quantity (resting asks across depth)
        self.lock = threading.Lock()

    def change_pct(self):
        """% change of live price vs previous close (same instrument as
        the live feed). None if either value isn't available yet."""
        with self.lock:
            if self.last_price is None or not self.prev_close:
                return None
            return (self.last_price - self.prev_close) / self.prev_close * 100

    def update(self, ltp, atp, total_traded_volume, tbq=None, tsq=None):
        with self.lock:
            self.last_price = ltp
            if atp:
                self.atp = atp
            if total_traded_volume and total_traded_volume > self.cum_volume:
                self.cum_volume = total_traded_volume
            if tbq is not None:
                self.tbq = tbq
            if tsq is not None:
                self.tsq = tsq

    def aggression_ratio(self):
        """tbq/tsq - how much resting buy interest outweighs resting sell
        interest (or vice versa) right now. None if depth data unavailable."""
        with self.lock:
            if not self.tbq or not self.tsq:
                return None
            return self.tbq / self.tsq

    def aggression_label(self):
        """Classifies current order-book pressure. This is resting-order
        imbalance (tbq/tsq), a real-time proxy for aggression - not a
        substitute for true tick-by-tick trade-initiator data, which
        Upstox's API doesn't expose."""
        ratio = self.aggression_ratio()
        if ratio is None:
            return None
        if ratio > 1.3:
            return "buying_aggression"
        if ratio < (1 / 1.3):
            return "selling_aggression"
        return "balanced"

    def update_oi(self, oi):
        if not oi:
            return
        with self.lock:
            if self.oi_open is None:
                self.oi_open = oi
            self.oi_current = oi

    def oi_direction(self):
        """Returns 'up', 'down', 'flat', or None if OI data isn't available yet."""
        with self.lock:
            if self.oi_open is None or self.oi_current is None:
                return None
            change_pct = (self.oi_current - self.oi_open) / self.oi_open * 100
            if change_pct > 0.5:
                return "up"
            elif change_pct < -0.5:
                return "down"
            return "flat"

    def oi_classification(self, price_up):
        """Combines price direction with OI direction to classify what's
        actually happening: fresh buildup vs unwinding/covering."""
        direction = self.oi_direction()
        if direction is None:
            return None
        if price_up and direction == "up":
            return "fresh_long"       # real buying, strongest bullish signal
        if price_up and direction == "down":
            return "short_covering"   # weaker bullish, prone to fade
        if not price_up and direction == "up":
            return "fresh_short"      # real selling, strongest bearish signal
        if not price_up and direction == "down":
            return "long_unwinding"   # weaker bearish, less committed
        return "neutral"

    def vwap(self):
        with self.lock:
            return self.atp

    def rvol(self):
        """Compares current cumulative volume to the baseline curve's value
        at the nearest earlier time bucket."""
        now_str = datetime.now(IST).strftime("%H:%M")
        buckets = sorted(self.rvol_baseline.keys())
        matched = None
        for b in buckets:
            if b <= now_str:
                matched = b
            else:
                break
        if matched is None or self.rvol_baseline.get(matched, 0) == 0:
            return None
        with self.lock:
            return self.cum_volume / self.rvol_baseline[matched]

    def nearest_zone(self):
        """Returns (zone_type, level, distance_pct) for the closest S/R zone
        to the current price."""
        if self.last_price is None:
            return None
        candidates = []
        for z in self.resistance_zones:
            dist_pct = (z["level"] - self.last_price) / self.last_price * 100
            candidates.append(("resistance", z["level"], dist_pct, z["strength"]))
        for z in self.support_zones:
            dist_pct = (self.last_price - z["level"]) / self.last_price * 100
            candidates.append(("support", z["level"], dist_pct, z["strength"]))

        # Only zones price is approaching from below (resistance) or above (support)
        valid = [c for c in candidates if c[2] >= 0]
        if not valid:
            return None
        valid.sort(key=lambda c: c[2])
        return valid[0]  # closest zone

    def trend_weight(self, bias):
        """Weighs the score based on daily EMA50/200 + RSI + MACD trend
        regime agreeing or conflicting with the intraday pattern's bias.
        This is a slower filter than OI/aggression (computed once at prep
        time from daily candles, not live), so it acts as a broader
        "is this fighting the daily trend" check rather than a fast signal.
        Missing indicator data (e.g. recently-listed stock) is neutral."""
        ti = self.trend_indicators
        weight = 1.0
        reasons = []

        ema_trend = ti.get("ema_trend")
        if ema_trend in ("bullish", "golden_cross"):
            weight *= 1.15 if bias == "bullish" else 0.8
            reasons.append(f"ema:{ema_trend}")
        elif ema_trend in ("bearish", "death_cross"):
            weight *= 1.15 if bias == "bearish" else 0.8
            reasons.append(f"ema:{ema_trend}")

        rsi_zone = ti.get("rsi_zone")
        if rsi_zone == "overbought":
            weight *= 0.85 if bias == "bullish" else 1.1  # chasing an overbought bullish move is riskier
            reasons.append("rsi:overbought")
        elif rsi_zone == "oversold":
            weight *= 0.85 if bias == "bearish" else 1.1  # chasing an oversold bearish move is riskier
            reasons.append("rsi:oversold")

        macd_state = ti.get("macd_state")
        if macd_state in ("bullish", "bullish_cross"):
            weight *= 1.1 if bias == "bullish" else 0.85
            reasons.append(f"macd:{macd_state}")
        elif macd_state in ("bearish", "bearish_cross"):
            weight *= 1.1 if bias == "bearish" else 0.85
            reasons.append(f"macd:{macd_state}")

        return round(weight, 3), ",".join(reasons) if reasons else "no_data"


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------
def evaluate(states):
    results = []
    for sym, state in states.items():
        if state.last_price is None:
            continue

        zone = state.nearest_zone()
        if zone is None:
            continue
        zone_type, level, dist_pct, strength = zone

        if dist_pct > NEAR_ZONE_PCT:
            continue  # not near enough

        rvol = state.rvol()
        if rvol is None or rvol < RVOL_THRESHOLD:
            continue

        vwap = state.vwap()
        if vwap is None:
            continue

        price_above_vwap = state.last_price > vwap

        # All four zone/VWAP combinations are meaningful, not just two -
        # heavy volume against the "expected" direction is a warning sign
        # (rejection/breakdown), just as worth flagging as confirmation is.
        if zone_type == "support" and price_above_vwap:
            pattern = "support_bounce"
            bias = "bullish"
        elif zone_type == "support" and not price_above_vwap:
            pattern = "support_breakdown_warning"
            bias = "bearish"
        elif zone_type == "resistance" and price_above_vwap:
            pattern = "resistance_breakout"
            bias = "bullish"
        else:  # resistance and not price_above_vwap
            pattern = "resistance_rejection_warning"
            bias = "bearish"

        oi_class = state.oi_classification(price_above_vwap)

        # OI confirmation weighting: fresh buildup in the expected direction
        # gets boosted, covering/unwinding gets a mild penalty since it's a
        # weaker, less committed move. No OI data yet -> neutral (1.0).
        oi_weight = {
            "fresh_long": 1.3,
            "fresh_short": 1.3,
            "short_covering": 0.8,
            "long_unwinding": 0.8,
            "neutral": 1.0,
            None: 1.0,
        }.get(oi_class, 1.0)

        # Order-book aggression (tbq/tsq imbalance) as a second confirming
        # factor - buying_aggression matching a bullish pattern (or
        # selling_aggression matching a bearish one) is boosted; a mismatch
        # (e.g. selling_aggression on a "support_bounce") is a caution flag
        # and gets discounted, since the resting order book disagrees with
        # the price/VWAP read.
        aggression = state.aggression_label()
        if aggression == "buying_aggression":
            aggression_weight = 1.2 if bias == "bullish" else 0.75
        elif aggression == "selling_aggression":
            aggression_weight = 1.2 if bias == "bearish" else 0.75
        else:  # 'balanced' or None (no depth data yet)
            aggression_weight = 1.0

        # Daily trend regime (EMA50/200 + RSI + MACD) as a fourth confirming
        # factor - see SymbolState.trend_weight() for the full logic. This
        # is the slowest-moving filter here, computed once at prep time.
        trend_wt, trend_reason = state.trend_weight(bias)

        score = (1 / max(dist_pct, 0.01)) * rvol * (1 + strength * 0.1) * oi_weight * aggression_weight * trend_wt
        results.append({
            "symbol": sym,
            "pattern": pattern,
            "bias": bias,
            "ltp": round(state.last_price, 2),
            "zone_level": level,
            "distance_pct": round(dist_pct, 2),
            "rvol": round(rvol, 2),
            "vwap": round(vwap, 2),
            "zone_strength": strength,
            "oi_signal": oi_class or "no_data",
            "aggression": aggression or "no_data",
            "trend_signal": trend_reason,
            "score": round(score, 2),
        })

    # Sort by RVOL first (highest relative volume at top - matches the
    # EOD performance script's approach: prioritize stocks trading heaviest
    # vs their own average, since that's what makes a move actionable).
    # score is still computed and shown for reference, just not the sort key.
    results.sort(key=lambda r: r["rvol"], reverse=True)
    return results


def scoring_loop(states, stop_event):
    csv_paths = init_daily_csvs()

    while not stop_event.is_set():
        now = datetime.now(IST).time()

        if now < MARKET_OPEN or now > MARKET_CLOSE:
            time.sleep(5)
            continue

        candidates = evaluate(states)
        timestamp = datetime.now(IST).strftime("%H:%M:%S")
        print(f"\n--- {timestamp} | Candidates: {len(candidates)} (sorted by RVOL, high to low) ---")
        for c in candidates:
            flag = "  " if c["bias"] == "bullish" else "**"  # simple visual cue for warnings
            print(f"{flag}{c['symbol']:12s} RVOL={c['rvol']:<6} {c['pattern']:<28s} LTP={c['ltp']:<9} "
                  f"zone={c['zone_level']:<9} dist%={c['distance_pct']:<6} "
                  f"VWAP={c['vwap']:<9} OI={c['oi_signal']:<15} AGGR={c['aggression']:<18} "
                  f"TREND={c['trend_signal']:<30} score={c['score']}")
            c["timestamp"] = timestamp
            append_candidate_row(csv_paths["candidates"], c)

        write_summary_snapshot(csv_paths["summary"], states)

        stop_event.wait(SCORING_INTERVAL_SEC)

    # Final snapshot on the way out, best-effort - the data is already safe
    # on disk from the periodic writes above even if this one gets cut off.
    try:
        write_summary_snapshot(csv_paths["summary"], states)
    except Exception:
        pass


CANDIDATE_FIELDNAMES = [
    "timestamp", "symbol", "rvol", "pattern", "bias", "ltp", "zone_level", "distance_pct",
    "vwap", "zone_strength", "oi_signal", "aggression", "trend_signal", "score",
]


def init_daily_csvs():
    """Creates today's summary + candidates CSVs (with headers) up front,
    so files exist on disk from the very start of the session rather than
    only at the end - nothing to lose if the process dies unexpectedly."""
    os.makedirs(EOD_LOG_DIR, exist_ok=True)
    date_str = datetime.now(IST).strftime("%Y-%m-%d")
    summary_path = os.path.join(EOD_LOG_DIR, f"eod_summary_{date_str}.csv")
    candidates_path = os.path.join(EOD_LOG_DIR, f"eod_candidates_{date_str}.csv")

    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "symbol", "final_ltp", "final_vwap", "final_volume",
            "oi_open", "oi_current", "oi_change_pct", "oi_direction", "last_updated",
        ])

    with open(candidates_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CANDIDATE_FIELDNAMES).writeheader()

    print(f"EOD logs will be written continuously to:\n  {summary_path}\n  {candidates_path}")
    return {"summary": summary_path, "candidates": candidates_path}


def write_summary_snapshot(summary_path, states):
    """Overwrites the summary CSV with current values. Called every scoring
    cycle (not just at exit), so the file is always close to up to date on
    disk regardless of how/when the process ends.

    NOTE: does NOT wrap the loop body in `with state.lock:` - oi_direction()
    acquires state.lock internally already. Nesting a second acquire on a
    plain (non-reentrant) threading.Lock from the same thread deadlocks it
    forever - this was the actual bug (same pattern fixed earlier today in
    streamlit_scanner_banknifty.py) that silently prevented this function
    from ever writing any data rows."""
    rows = []
    for symbol, state in states.items():
        oi_change_pct = None
        if state.oi_open and state.oi_current:
            oi_change_pct = round((state.oi_current - state.oi_open) / state.oi_open * 100, 2)
        rows.append([
            symbol, state.last_price, state.atp, state.cum_volume,
            state.oi_open, state.oi_current, oi_change_pct, state.oi_direction(),
            datetime.now(IST).strftime("%H:%M:%S"),
        ])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "symbol", "final_ltp", "final_vwap", "final_volume",
            "oi_open", "oi_current", "oi_change_pct", "oi_direction", "last_updated",
        ])
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def append_candidate_row(candidates_path, candidate):
    """Appends a single candidate row immediately, so it's on disk the
    moment it's flagged - no batching, nothing lost on interrupt."""
    with open(candidates_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDNAMES)
        writer.writerow({k: candidate.get(k, "") for k in CANDIDATE_FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------------------------------
FULL_QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
POLL_INTERVAL_SEC = 10  # how often to fetch fresh quotes for all stocks


def fetch_all_quotes(instrument_keys, access_token):
    """Single REST call fetching LTP/volume/avg_price/OI/depth for ALL
    instruments at once (up to 500 per Upstox's limit - well over our
    ~208). Returns the raw response dict, keyed by 'EXCHANGE:SYMBOL' with
    each entry also containing 'instrument_token' matching our key format,
    which is what we actually match against (not the outer key)."""
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    params = {"instrument_key": ",".join(instrument_keys)}
    resp = requests.get(FULL_QUOTES_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", {})


def update_states_from_quotes(quotes_data, states_by_key):
    """Applies one polling cycle's quote data to every matching
    SymbolState, using the exact same update()/update_oi() methods the
    (now-abandoned for today) WebSocket path used - so all downstream
    scoring logic is completely unchanged."""
    updated_count = 0
    for entry in quotes_data.values():
        instrument_key = entry.get("instrument_token")
        state = states_by_key.get(instrument_key)
        if state is None:
            continue
        ltp = entry.get("last_price")
        atp = entry.get("average_price")
        volume = entry.get("volume") or 0
        tbq = entry.get("total_buy_quantity")
        tsq = entry.get("total_sell_quantity")
        oi = entry.get("oi")
        state.update(ltp, atp, volume, tbq, tsq)
        state.update_oi(oi)
        updated_count += 1
    return updated_count


def polling_thread(instrument_keys, states_by_key, access_token, stop_event):
    """Runs forever (until stop_event is set), fetching fresh quotes for
    all stocks every POLL_INTERVAL_SEC seconds via REST - this is a
    complete replacement for the WebSocket streaming path, built because
    the WebSocket connection was confirmed (via extensive testing) to
    open successfully but never actually deliver tick data, while REST
    calls worked reliably throughout the same session. This trades true
    push-based real-time updates for a proven-reliable ~10s-latency
    polling approach."""
    while not stop_event.is_set():
        try:
            quotes = fetch_all_quotes(instrument_keys, access_token)
            n = update_states_from_quotes(quotes, states_by_key)
            print(f"[POLL] Updated {n}/{len(states_by_key)} stocks from REST quotes")
        except Exception as e:
            print(f"[POLL] ERROR fetching quotes: {type(e).__name__}: {e}")
        stop_event.wait(POLL_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    config = load_config()
    access_token = get_access_token(config)

    prep_data = load_today_prep()

    states = {}             # symbol -> SymbolState
    states_by_key = {}       # instrument_key (subscribed one, per stock) -> SymbolState
    instrument_keys = []      # single list for subscription - one key per stock now

    for symbol, info in prep_data["stocks"].items():
        instrument_key = info["instrument_key"]
        futures_key = info.get("futures_instrument_key")
        state = SymbolState(
            symbol=symbol,
            instrument_key=instrument_key,
            futures_instrument_key=futures_key,
            sr_data={
                "resistance_zones": info.get("resistance_zones", []),
                "support_zones": info.get("support_zones", []),
            },
            rvol_baseline=info.get("rvol_baseline", {}),
            trend_indicators=info.get("trend_indicators", {}),
            prev_close=info.get("prev_close"),
        )
        states[symbol] = state

        # Subscribe to futures for LTP/VWAP/volume/OI together (see
        # make_on_message docstring for why) - falls back to cash market
        # only if no futures contract was resolved for this stock at all.
        if futures_key:
            states_by_key[futures_key] = state
            instrument_keys.append(futures_key)
        else:
            states_by_key[instrument_key] = state
            instrument_keys.append(instrument_key)
            print(f"  Note: no futures contract for {symbol}, falling back to cash market for live data.")

    print(f"Loaded {len(states)} symbols for live scanning.")
    print(f"Using REST POLLING mode (every {POLL_INTERVAL_SEC}s) instead of WebSocket -")
    print(f"see live_scanner_polling.py header comment for why.")

    stop_event = threading.Event()
    scorer_thread = threading.Thread(target=scoring_loop, args=(states, stop_event), daemon=True)
    scorer_thread.start()

    try:
        polling_thread(instrument_keys, states_by_key, access_token, stop_event)
    finally:
        # No cleanup write needed here - the scoring loop writes the summary
        # snapshot and appends each candidate to disk continuously, every
        # cycle, so today's CSVs in eod_logs/ are already up to date.
        sys.stdout.write("Stopping. Today's logs are in eod_logs/ (written continuously).\n")
        sys.stdout.flush()
        stop_event.set()


if __name__ == "__main__":
    main()
