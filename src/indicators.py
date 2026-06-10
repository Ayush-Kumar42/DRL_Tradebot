from __future__ import annotations
 
import numpy as np
import pandas as pd
import ta
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Column name contracts used elsewhere in the codebase
# ─────────────────────────────────────────────────────────────────────────────
 
SIGNAL_INDICATORS = ["RSI", "MACD", "MA", "HA", "STOCH", "BBANDS", "CCI", "OBV"]
CONTINUOUS_FEATURES = ["CMF", "VWAP", "ATR", "VOLATILITY", "PARKINSON", "PRICE_ACTION"]
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────
 
def _safe_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]; returns zeros if range is zero."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Signal indicators
# ─────────────────────────────────────────────────────────────────────────────
 
def _calc_RSI(df: pd.DataFrame, window: int = 14) -> None:
    """
    RSI via ta.momentum.RSIIndicator.
    Entry (buy signal)  : RSI < 30  (oversold)
    Exit  (sell signal) : RSI > 70  (overbought)
    Strength            : normalised distance from the threshold
    """
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    df["RSI"] = rsi
 
    entries = rsi < 30
    exits   = rsi > 70
 
    df["RSI_strength"] = 0.0
    df.loc[entries, "RSI_strength"] = ((30 - rsi) / 30).clip(0, 1)
    df.loc[exits,   "RSI_strength"] = ((rsi - 70) / 30).clip(0, 1)
 
    df["RSI_entries"] = entries
    df["RSI_exits"]   = exits
 
 
def _calc_MACD(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> None:
    """
    MACD via ta.trend.MACD.
    Entry : MACD line crosses above signal line
    Exit  : MACD line crosses below signal line
    Strength: normalised |MACD - signal| (only at crossover bars)
    """
    ind   = ta.trend.MACD(df["Close"], window_fast=fast, window_slow=slow, window_sign=signal)
    macd  = ind.macd()
    sig   = ind.macd_signal()
    diff  = macd - sig
 
    # Crossover: sign change between consecutive bars
    prev_diff = diff.shift(1)
    entries = (prev_diff < 0) & (diff >= 0)
    exits   = (prev_diff > 0) & (diff <= 0)
 
    macd_diff_abs = diff.abs()
    norm = _safe_norm(macd_diff_abs)
 
    df["MACD_strength"] = 0.0
    df.loc[entries | exits, "MACD_strength"] = norm.clip(0, 1)
 
    df["MACD_entries"] = entries
    df["MACD_exits"]   = exits
 
 
def _calc_MA(df: pd.DataFrame, short: int = 20, long_: int = 50) -> None:
    """
    Moving-average crossover.
    Entry : short MA > long MA
    Exit  : short MA < long MA
    Strength: normalised |short - long|
    """
    ma_short = df["Close"].rolling(short).mean()
    ma_long  = df["Close"].rolling(long_).mean()
 
    entries = ma_short > ma_long
    exits   = ma_short < ma_long
 
    ma_diff = (ma_short - ma_long).abs()
    mx = ma_diff.max()
    df["MA_strength"] = (ma_diff / (mx if mx != 0 else 1)).clip(0, 1)
 
    df["MA_entries"] = entries
    df["MA_exits"]   = exits
 
 
def _calc_HA(df: pd.DataFrame) -> None:
    """
    Heiken-Ashi candles, computed iteratively so HA_Open is accurate.
    Entry : green candle with flat bottom  (HA_Low == HA_Open)
    Exit  : red candle with flat top       (HA_High == HA_Open)
    Strength: body / full-range ratio
    """
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
 
    # Iterative HA_Open (cannot be vectorised due to self-reference)
    ha_open = df["Open"].copy().astype(float)
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
 
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)
 
    green_flat = (ha_close > ha_open) & (ha_low == ha_open)
    red_flat   = (ha_close < ha_open) & (ha_high == ha_open)
 
    ha_body  = (ha_close - ha_open).abs()
    ha_range = (ha_high - ha_low).replace(0, np.nan)
 
    df["HA_strength"] = 0.0
    df.loc[green_flat | red_flat, "HA_strength"] = (
        (ha_body / ha_range).clip(0, 1)
    )
 
    df["HA_entries"] = green_flat
    df["HA_exits"]   = red_flat
 
 
def _calc_OBV(df: pd.DataFrame) -> None:
    """
    On-Balance Volume via ta.volume.OnBalanceVolumeIndicator.
    Entry : OBV increasing (diff > 0)
    Exit  : OBV decreasing (diff < 0)
    Strength: normalised OBV diff magnitude (only at signal bars)
    """
    obv      = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    obv_diff = obv.diff()
 
    entries = obv_diff > 0
    exits   = obv_diff < 0
 
    norm = _safe_norm(obv_diff)
 
    df["OBV_strength"] = 0.0
    df.loc[entries | exits, "OBV_strength"] = norm.clip(0, 1)
 
    df["OBV_entries"] = entries
    df["OBV_exits"]   = exits
 
 
def _calc_STOCH(df: pd.DataFrame) -> None:
    """
    Stochastic oscillator via ta.momentum.StochasticOscillator.
    Entry : %K > %D
    Exit  : %K < %D
    Strength: |%K - %D| / 100
    """
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    k = stoch.stoch()
    d = stoch.stoch_signal()
 
    entries = k > d
    exits   = k < d
 
    df["STOCH_strength"] = ((k - d).abs() / 100).clip(0, 1)
 
    df["STOCH_entries"] = entries
    df["STOCH_exits"]   = exits
 
 
def _calc_BBANDS(df: pd.DataFrame) -> None:
    """
    Bollinger Bands via ta.volatility.BollingerBands.
    Entry : Close < lower band  (mean-reversion buy)
    Exit  : Close > upper band  (mean-reversion sell)
    Strength: distance beyond band, scaled by bandwidth
    """
    bb     = ta.volatility.BollingerBands(df["Close"])
    lower  = bb.bollinger_lband()
    upper  = bb.bollinger_hband()
    bwidth = bb.bollinger_wband()    # (upper - lower) / mid
 
    entries = df["Close"] < lower
    exits   = df["Close"] > upper
 
    entry_strength = ((lower - df["Close"]) / bwidth.replace(0, np.nan)).fillna(0)
    exit_strength  = ((df["Close"] - upper) / bwidth.replace(0, np.nan)).fillna(0)
 
    df["BBANDS_strength"] = 0.0
    df.loc[entries, "BBANDS_strength"] = entry_strength.clip(0, 1)
    df.loc[exits,   "BBANDS_strength"] = exit_strength.clip(0, 1)
 
    df["BBANDS_entries"] = entries
    df["BBANDS_exits"]   = exits
 
 
def _calc_CCI(df: pd.DataFrame, window: int = 20, constant: float = 0.015) -> None:
    """
    Commodity Channel Index — computed raw (ta's CCI uses a different constant).
    Entry : |CCI| > 100  (breakout in either direction)
    Exit  : CCI returning toward zero  (-50 < CCI < 0 or 0 < CCI < 50)
    Strength: normalised to [0,1] via (CCI + 200) / 400
    """
    tp     = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_ma  = tp.rolling(window).mean()
    tp_md  = tp.rolling(window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci    = (tp - tp_ma) / (constant * tp_md.replace(0, np.nan))
 
    entries = (cci < -100) | (cci > 100)
    exits   = ((cci > -50) & (cci < 0)) | ((cci < 50) & (cci > 0))
 
    norm_cci = ((cci + 200) / 400).clip(0, 1)
 
    df["CCI_strength"] = 0.0
    df.loc[entries | exits, "CCI_strength"] = norm_cci
 
    df["CCI_entries"] = entries
    df["CCI_exits"]   = exits
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Continuous features
# ─────────────────────────────────────────────────────────────────────────────
 
def _calc_CMF(df: pd.DataFrame, window: int = 20) -> None:
    """Chaikin Money Flow: rolling (MFV sum) / (Volume sum), clipped to [-1, 1]."""
    mfm = (
        (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    ) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = mfm * df["Volume"]
    df["CMF"] = (
        mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()
    ).clip(-1, 1)
 
 
def _calc_VWAP(df: pd.DataFrame) -> None:
    """Cumulative VWAP. Resets to NaN on the first bar (cumsum from row 0)."""
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
 
 
def _calc_ATR(df: pd.DataFrame, window: int = 14) -> None:
    """Average True Range using simple rolling mean of TR."""
    tr = (
        pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"]  - df["Close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
    )
    df["ATR"] = tr.rolling(window).mean()
 
 
def _calc_VOLATILITY(df: pd.DataFrame, window: int = 20) -> None:
    """Rolling standard deviation of log returns."""
    df["VOLATILITY"] = df["Close"].pct_change().rolling(window).std()
 
 
def _calc_PARKINSON(df: pd.DataFrame, window: int = 20) -> None:
    """Parkinson volatility estimator, rolling mean."""
    parkinson = (1 / (4 * np.log(2))) * (np.log(df["High"] / df["Low"]) ** 2)
    df["PARKINSON"] = parkinson.rolling(window).mean()
 
 
def _calc_PRICE_ACTION(df: pd.DataFrame, window: int = 20) -> None:
    """
    Body-to-range ratio over the rolling window.
    Also adds dist_from_high and dist_from_low as auxiliary columns
    (useful as extra continuous features if needed).
    """
    high = df["High"].rolling(window).max()
    low  = df["Low"].rolling(window).min()
    df["dist_from_high"] = (df["Close"] - high) / high
    df["dist_from_low"]  = (df["Close"] - low)  / low
    body = (df["Close"] - df["Open"]).abs()
    rng  = (df["High"] - df["Low"]).replace(0, np.nan)
    df["PRICE_ACTION"] = (body / rng).clip(0, 1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Signal column builder  (entry/exit → -1 / 0 / +1)
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_signal_columns(df: pd.DataFrame) -> None:
    """
    For each signal indicator, convert the boolean *_entries / *_exits columns
    into a single integer *_signal column:
        +1  entry fired (and exit did not)
        -1  exit fired  (and entry did not)
         0  neither     (or both — entry takes priority)
    Then drop the raw *_entries / *_exits columns.
    """
    for ind in SIGNAL_INDICATORS:
        entry_col  = f"{ind}_entries"
        exit_col   = f"{ind}_exits"
        signal_col = f"{ind}_signal"
 
        if entry_col not in df.columns or exit_col not in df.columns:
            print(f"[indicators] WARNING: skipping {ind} — missing entry/exit columns")
            continue
 
        df[signal_col] = 0
        df.loc[df[entry_col].fillna(False),                                 signal_col] = 1
        df.loc[df[exit_col].fillna(False)  & (df[signal_col] != 1), signal_col] = -1
 
    # Drop raw boolean columns
    drop_cols = (
        [f"{ind}_entries" for ind in SIGNAL_INDICATORS if f"{ind}_entries" in df.columns]
        + [f"{ind}_exits" for ind in SIGNAL_INDICATORS if f"{ind}_exits"   in df.columns]
    )
    df.drop(columns=drop_cols, inplace=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Intermediate column cleanup
# ─────────────────────────────────────────────────────────────────────────────
 
_INTERMEDIATE_COLS = [
    "HA_Close", "HA_Open", "HA_High", "HA_Low",
    "MA_short", "MA_long",
    "%K", "%D", "BB_lower", "BB_upper", "BB_std",
    "OBV_diff", "CCI",
]
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def load_and_compute(csv_path: str) -> pd.DataFrame:
    """
    Load BTCUSDT_1m.csv (or any OHLCV CSV with a Datetime column) and
    compute all signal and continuous indicators.
 
    Expected CSV columns
    --------------------
    Datetime, Open, High, Low, Close, Volume
 
    Returns
    -------
    pd.DataFrame with:
      - Datetime index
      - Open, High, Low, Close, Volume
      - RSI_signal, MACD_signal, MA_signal, HA_signal,
        STOCH_signal, BBANDS_signal, CCI_signal, OBV_signal
      - RSI_strength, MACD_strength, MA_strength, HA_strength,
        STOCH_strength, BBANDS_strength, CCI_strength, OBV_strength
      - CMF, VWAP, ATR, VOLATILITY, PARKINSON, PRICE_ACTION
      - dist_from_high, dist_from_low  (auxiliary price-action columns)
      - RSI                             (raw RSI value, useful as continuous)
    """
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
 
    # Initialise strength columns so they exist even if an indicator errors
    for name in SIGNAL_INDICATORS:
        df[f"{name}_strength"] = 0.0
 
    # ── Signal indicators ────────────────────────────────────────────────
    _calc_RSI(df)
    _calc_MACD(df)
    _calc_MA(df)
    _calc_HA(df)
    _calc_OBV(df)
    _calc_STOCH(df)
    _calc_BBANDS(df)
    _calc_CCI(df)
 
    # ── Continuous features ──────────────────────────────────────────────
    _calc_CMF(df)
    _calc_VWAP(df)
    _calc_ATR(df)
    _calc_VOLATILITY(df)
    _calc_PARKINSON(df)
    _calc_PRICE_ACTION(df)
 
    # ── Convert entry/exit booleans → signal integers ────────────────────
    _build_signal_columns(df)
 
    # ── Drop intermediate work columns ───────────────────────────────────
    df.drop(columns=_INTERMEDIATE_COLS, errors="ignore", inplace=True)
 
    # ── Drop rows where any required column is NaN (warm-up period) ──────
    required_cols = (
        [f"{ind}_signal"   for ind in SIGNAL_INDICATORS]
        + [f"{ind}_strength" for ind in SIGNAL_INDICATORS]
        + CONTINUOUS_FEATURES
    )
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
 
    return df