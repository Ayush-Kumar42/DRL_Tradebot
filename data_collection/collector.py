import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from binance.client import Client
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()                          # reads .env in the same directory
API_KEY    = os.getenv("api_key")
API_SECRET = os.getenv("api_secret")

SYMBOLS = [
    "BTC","LINK", "ATOM",
    "CRO",  "DOGE", "EOS",  "ETH",  "IOTA", "LTC",
    "XMR",  "XEM",  "DOT",  "SOL",  "XLM",  "USDT",
    "TRX",  "UNI",  "USDC", "WBTC", "XRP",
]
QUOTE      = "USDT"          # quote asset; USDT/USDT pair will be skipped
INTERVAL   = Client.KLINE_INTERVAL_1MINUTE
OUTPUT_DIR = Path("data")

# Binance returns max 1000 candles per request.
# We page backwards from "now" all the way to each symbol's listing date.
CANDLES_PER_REQUEST = 1000
REQUEST_DELAY_SEC   = 0.25   # stay well within rate limits

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


def fetch_all_klines(client: Client, symbol: str) -> pd.DataFrame:
    """
    Paginate through the entire available 1-min history for *symbol*
    by repeatedly calling get_klines with an end_time that walks backwards,
    then reverse-sorting to chronological order.
    """
    all_rows = []
    end_ms   = None          # None → current time on first call

    log.info(f"  [{symbol}] starting full history download …")

    while True:
        kwargs = dict(symbol=symbol, interval=INTERVAL, limit=CANDLES_PER_REQUEST)
        if end_ms is not None:
            kwargs["endTime"] = end_ms - 1   # exclusive upper bound

        try:
            candles = client.get_klines(**kwargs)
        except Exception as exc:
            log.error(f"  [{symbol}] API error: {exc}  – skipping remainder")
            break

        if not candles:
            break

        all_rows.extend(candles)

        oldest_open_ms = candles[0][0]
        end_ms         = oldest_open_ms   # next page ends just before this

        oldest_dt = datetime.fromtimestamp(oldest_open_ms / 1000, tz=timezone.utc)
        log.info(f"  [{symbol}] fetched {len(candles):>4} candles  "
                 f"| oldest so far: {oldest_dt.strftime('%Y-%m-%d %H:%M')}"
                 f"  | total rows: {len(all_rows):,}")

        if len(candles) < CANDLES_PER_REQUEST:
            # Reached the beginning of available history
            break

        time.sleep(REQUEST_DELAY_SEC)

    if not all_rows:
        return pd.DataFrame(columns=KLINE_COLS)

    df = pd.DataFrame(all_rows, columns=KLINE_COLS)

    # Sort oldest → newest and deduplicate
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset="open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Type-cast
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]:
        df[col] = pd.to_numeric(df[col])
    df["num_trades"] = df["num_trades"].astype(int)
    df.drop(columns=["ignore"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY or not API_SECRET:
        raise EnvironmentError(
            "api_key / api_secret not found. "
            "Make sure a .env file exists next to this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client(API_KEY, API_SECRET)

    # Verify connectivity
    server_time = client.get_server_time()
    log.info(f"Connected to Binance. Server time: "
             f"{datetime.fromtimestamp(server_time['serverTime']/1000, tz=timezone.utc)}")

    # Resolve tradeable pairs (some symbols may not have a USDT pair)
    exchange_info = client.get_exchange_info()
    tradeable     = {s["symbol"] for s in exchange_info["symbols"] if s["status"] == "TRADING"}

    results = {}

    for base in SYMBOLS:
        if base == "USDT":
            log.warning("Skipping USDT/USDT – not a valid pair.")
            continue

        pair = f"{base}{QUOTE}"

        if pair not in tradeable:
            log.warning(f"Skipping {pair} – not listed / not trading on Binance.")
            continue

        log.info(f"=== {pair} ===")
        df = fetch_all_klines(client, pair)

        if df.empty:
            log.warning(f"  [{pair}] No data returned – skipping save.")
            continue

        out_path = OUTPUT_DIR / f"{pair}_1m.csv"
        df.to_csv(out_path, index=False)

        candle_count = len(df)
        first_ts     = df["open_time"].iloc[0]
        last_ts      = df["open_time"].iloc[-1]
        results[pair] = {"candles": candle_count, "from": first_ts, "to": last_ts}

        log.info(f"  [{pair}] Saved {candle_count:,} candles "
                 f"({first_ts.date()} → {last_ts.date()})  →  {out_path}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 60)
    total = 0
    for sym, info in results.items():
        log.info(f"  {sym:<12}  {info['candles']:>10,} candles   "
                 f"{str(info['from'].date()):>12} → {str(info['to'].date())}")
        total += info["candles"]
    log.info(f"\n  {'TOTAL':<12}  {total:>10,} candles across {len(results)} pairs")
    log.info("=" * 60)


if __name__ == "__main__":
    main()