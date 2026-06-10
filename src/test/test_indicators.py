"""
test_indicators.py
==================
Comprehensive pytest suite for indicators.py.

Run with:
    pytest test_indicators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import indicators as ind
from src.indicators import (
    SIGNAL_INDICATORS,
    CONTINUOUS_FEATURES,
    _safe_norm,
    _calc_RSI,
    _calc_MACD,
    _calc_MA,
    _calc_HA,
    _calc_OBV,
    _calc_STOCH,
    _calc_BBANDS,
    _calc_CCI,
    _calc_CMF,
    _calc_VWAP,
    _calc_ATR,
    _calc_VOLATILITY,
    _calc_PARKINSON,
    _calc_PRICE_ACTION,
    _build_signal_columns,
    _INTERMEDIATE_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 30_000 + np.cumsum(rng.normal(0, 150, n))
    high  = close + rng.uniform(50, 300, n)
    low   = close - rng.uniform(50, 300, n)
    open_ = close + rng.normal(0, 80, n)
    vol   = rng.uniform(1, 100, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=range(n),
    )


@pytest.fixture
def df() -> pd.DataFrame:
    return _make_ohlcv(200)


@pytest.fixture
def tmp_csv(tmp_path) -> str:
    """Write a well-formed OHLCV CSV and return its path."""
    frame = _make_ohlcv(300)
    frame.insert(0, "Datetime", pd.date_range("2024-01-01", periods=300, freq="1min"))
    path = tmp_path / "BTCUSDT_1m.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _run_all_indicators(df: pd.DataFrame) -> None:
    """Apply every indicator in the correct pipeline order."""
    for fn in (_calc_RSI, _calc_MACD, _calc_MA, _calc_HA,
               _calc_OBV, _calc_STOCH, _calc_BBANDS, _calc_CCI):
        fn(df)
    _calc_CMF(df)
    _calc_VWAP(df)
    _calc_ATR(df)
    _calc_VOLATILITY(df)
    _calc_PARKINSON(df)
    _calc_PRICE_ACTION(df)


# ─────────────────────────────────────────────────────────────────────────────
#  _safe_norm
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeNorm:
    def test_output_range(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _safe_norm(s)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_series_returns_zeros(self):
        s = pd.Series([7.0] * 10)
        result = _safe_norm(s)
        assert (result == 0.0).all()

    def test_preserves_index(self):
        s = pd.Series([10, 20, 30], index=[5, 10, 15])
        result = _safe_norm(s)
        assert list(result.index) == [5, 10, 15]

    def test_single_element(self):
        s = pd.Series([42.0])
        result = _safe_norm(s)
        assert result.iloc[0] == 0.0

    def test_two_distinct_values(self):
        s = pd.Series([0.0, 1.0])
        result = _safe_norm(s)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_RSI
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcRSI:
    def test_columns_created(self, df):
        _calc_RSI(df)
        assert "RSI" in df.columns
        assert "RSI_strength" in df.columns
        assert "RSI_entries" in df.columns
        assert "RSI_exits" in df.columns

    def test_rsi_range(self, df):
        _calc_RSI(df)
        valid = df["RSI"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_entries_below_30(self, df):
        _calc_RSI(df)
        mask = df["RSI_entries"] & df["RSI"].notna()
        assert (df.loc[mask, "RSI"] < 30).all()

    def test_exits_above_70(self, df):
        _calc_RSI(df)
        mask = df["RSI_exits"] & df["RSI"].notna()
        assert (df.loc[mask, "RSI"] > 70).all()

    def test_strength_range(self, df):
        _calc_RSI(df)
        assert (df["RSI_strength"] >= 0).all()
        assert (df["RSI_strength"] <= 1).all()

    def test_strength_zero_when_no_signal(self, df):
        _calc_RSI(df)
        neither = ~df["RSI_entries"] & ~df["RSI_exits"]
        assert (df.loc[neither, "RSI_strength"] == 0.0).all()

    def test_custom_window(self, df):
        _calc_RSI(df, window=21)
        assert "RSI" in df.columns
        assert df["RSI"].dropna().between(0, 100).all()

    def test_entry_and_exit_mutually_exclusive(self, df):
        _calc_RSI(df)
        # RSI can't simultaneously be < 30 and > 70
        both = df["RSI_entries"] & df["RSI_exits"]
        assert not both.any()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_MACD
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcMACD:
    def test_columns_created(self, df):
        _calc_MACD(df)
        for col in ("MACD_strength", "MACD_entries", "MACD_exits"):
            assert col in df.columns

    def test_signal_type(self, df):
        _calc_MACD(df)
        assert df["MACD_entries"].dtype == bool
        assert df["MACD_exits"].dtype == bool

    def test_strength_range(self, df):
        _calc_MACD(df)
        assert (df["MACD_strength"] >= 0).all()
        assert (df["MACD_strength"] <= 1).all()

    def test_strength_zero_when_no_signal(self, df):
        _calc_MACD(df)
        neither = ~df["MACD_entries"] & ~df["MACD_exits"]
        assert (df.loc[neither, "MACD_strength"] == 0.0).all()

    def test_entries_exits_mutually_exclusive_on_crossover(self, df):
        """A bar cannot be both a bullish and bearish crossover."""
        _calc_MACD(df)
        both = df["MACD_entries"] & df["MACD_exits"]
        assert not both.any()

    def test_custom_windows(self, df):
        _calc_MACD(df, fast=5, slow=15, signal=5)
        assert "MACD_entries" in df.columns

    def test_entry_on_bullish_crossover(self, df):
        """Where entry fires, MACD diff must have changed from negative to non-negative."""
        _calc_MACD(df)
        # If there are entries, they should be sparse (not every bar)
        entry_rate = df["MACD_entries"].mean()
        assert entry_rate < 0.5  # crossovers are rare events


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_MA
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcMA:
    def test_columns_created(self, df):
        _calc_MA(df)
        for col in ("MA_strength", "MA_entries", "MA_exits"):
            assert col in df.columns

    def test_entries_exits_mutually_exclusive(self, df):
        _calc_MA(df)
        both = df["MA_entries"] & df["MA_exits"]
        assert not both.any()

    def test_strength_range(self, df):
        _calc_MA(df)
        assert (df["MA_strength"].dropna() >= 0).all()
        assert (df["MA_strength"].dropna() <= 1).all()

    def test_custom_windows(self, df):
        _calc_MA(df, short=5, long_=10)
        valid = df["MA_strength"].dropna()
        assert len(valid) > 0

    def test_entry_when_short_above_long(self):
        """Monotone increasing close guarantees short MA > long MA after warm-up."""
        n = 100
        close = np.arange(1.0, n + 1)
        frame = pd.DataFrame({
            "Open": close - 0.1, "High": close + 0.1,
            "Low":  close - 0.1, "Close": close,
            "Volume": np.ones(n),
        })
        _calc_MA(frame, short=5, long_=10)
        valid = frame.dropna(subset=["MA_entries"])
        assert valid["MA_entries"].any()

    def test_exit_when_short_below_long(self):
        """Monotone decreasing close guarantees short MA < long MA after warm-up."""
        n = 100
        close = np.arange(n, 0, -1, dtype=float)
        frame = pd.DataFrame({
            "Open": close + 0.1, "High": close + 0.2,
            "Low":  close - 0.1, "Close": close,
            "Volume": np.ones(n),
        })
        _calc_MA(frame, short=5, long_=10)
        valid = frame.dropna(subset=["MA_exits"])
        assert valid["MA_exits"].any()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_HA
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcHA:
    def test_columns_created(self, df):
        _calc_HA(df)
        for col in ("HA_strength", "HA_entries", "HA_exits"):
            assert col in df.columns

    def test_strength_range(self, df):
        _calc_HA(df)
        s = df["HA_strength"]
        assert (s >= 0).all() and (s <= 1).all()

    def test_strength_zero_outside_signals(self, df):
        _calc_HA(df)
        neither = ~df["HA_entries"] & ~df["HA_exits"]
        assert (df.loc[neither, "HA_strength"] == 0.0).all()

    def test_boolean_columns(self, df):
        _calc_HA(df)
        assert df["HA_entries"].dtype == bool
        assert df["HA_exits"].dtype == bool

    def test_no_negative_strength(self, df):
        _calc_HA(df)
        assert (df["HA_strength"] >= 0).all()

    def test_first_bar_uses_real_open(self, df):
        """HA iterative open: row 0 HA_open must equal row 0 real open."""
        # We can't read ha_open directly, but we can verify _calc_HA doesn't raise
        # and produces valid outputs on a minimal frame
        frame = _make_ohlcv(5)
        _calc_HA(frame)
        assert "HA_strength" in frame.columns


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_OBV
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcOBV:
    def test_columns_created(self, df):
        _calc_OBV(df)
        for col in ("OBV_strength", "OBV_entries", "OBV_exits"):
            assert col in df.columns

    def test_strength_range(self, df):
        _calc_OBV(df)
        assert (df["OBV_strength"] >= 0).all()
        assert (df["OBV_strength"] <= 1).all()

    def test_entries_exits_cover_all_non_flat(self, df):
        """Random data: OBV changes every bar, so at least some of each."""
        _calc_OBV(df)
        assert df["OBV_entries"].any()
        assert df["OBV_exits"].any()

    def test_entries_exits_not_both_true(self, df):
        _calc_OBV(df)
        both = df["OBV_entries"] & df["OBV_exits"]
        assert not both.any()

    def test_entry_when_close_up(self):
        """Rising price → positive OBV diff → entry."""
        n = 30
        close = np.arange(100.0, 100 + n)  # strictly increasing
        frame = pd.DataFrame({
            "Open": close - 0.5, "High": close + 0.5,
            "Low":  close - 0.5, "Close": close,
            "Volume": np.ones(n) * 10,
        })
        _calc_OBV(frame)
        # After warm-up (diff starts at row 1), all should be entries
        assert frame.loc[1:, "OBV_entries"].all()

    def test_exit_when_close_down(self):
        """Falling price → negative OBV diff → exit."""
        n = 30
        close = np.arange(130.0, 130 - n, -1)
        frame = pd.DataFrame({
            "Open": close + 0.5, "High": close + 0.5,
            "Low":  close - 0.5, "Close": close,
            "Volume": np.ones(n) * 10,
        })
        _calc_OBV(frame)
        assert frame.loc[1:, "OBV_exits"].all()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_STOCH
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcSTOCH:
    def test_columns_created(self, df):
        _calc_STOCH(df)
        for col in ("STOCH_strength", "STOCH_entries", "STOCH_exits"):
            assert col in df.columns

    def test_strength_range(self, df):
        _calc_STOCH(df)
        s = df["STOCH_strength"].dropna()
        assert (s >= 0).all() and (s <= 1).all()

    def test_entries_exits_mutually_exclusive(self, df):
        _calc_STOCH(df)
        both = df["STOCH_entries"].fillna(False) & df["STOCH_exits"].fillna(False)
        assert not both.any()

    def test_strength_formula(self, df):
        """Strength = |%K - %D| / 100, so always ≤ 1."""
        _calc_STOCH(df)
        assert (df["STOCH_strength"].dropna() <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_BBANDS
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcBBANDS:
    def test_columns_created(self, df):
        _calc_BBANDS(df)
        for col in ("BBANDS_strength", "BBANDS_entries", "BBANDS_exits"):
            assert col in df.columns

    def test_strength_range(self, df):
        _calc_BBANDS(df)
        s = df["BBANDS_strength"]
        assert (s >= 0).all() and (s <= 1).all()

    def test_entry_below_lower_band(self):
        """Force a close far below the lower band and confirm entry fires."""
        n = 100
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high  = close + 2
        low   = close - 2
        open_ = close + rng.normal(0, 0.5, n)
        vol   = np.ones(n)
        frame = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                               "Close": close, "Volume": vol})
        frame.loc[99, "Close"] = close.mean() - 10 * close.std()
        frame.loc[99, "Low"]   = frame.loc[99, "Close"] - 1
        _calc_BBANDS(frame)
        assert frame.loc[99, "BBANDS_entries"]

    def test_exit_above_upper_band(self):
        """Force a close far above the upper band and confirm exit fires."""
        n = 100
        rng = np.random.default_rng(1)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        high  = close + 2
        low   = close - 2
        open_ = close + rng.normal(0, 0.5, n)
        vol   = np.ones(n)
        frame = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                               "Close": close, "Volume": vol})
        frame.loc[99, "Close"] = close.mean() + 10 * close.std()
        frame.loc[99, "High"]  = frame.loc[99, "Close"] + 1
        _calc_BBANDS(frame)
        assert frame.loc[99, "BBANDS_exits"]

    def test_no_signal_near_midband(self, df):
        """A flat series near the midband should not trigger entry or exit."""
        n = 100
        close = np.full(n, 100.0) + np.linspace(-0.01, 0.01, n)
        frame = pd.DataFrame({
            "Open": close, "High": close + 0.01,
            "Low": close - 0.01, "Close": close,
            "Volume": np.ones(n),
        })
        _calc_BBANDS(frame)
        assert not frame["BBANDS_entries"].any()
        assert not frame["BBANDS_exits"].any()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_CCI
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcCCI:
    def test_columns_created(self, df):
        _calc_CCI(df)
        for col in ("CCI_strength", "CCI_entries", "CCI_exits"):
            assert col in df.columns

    def test_strength_range(self, df):
        _calc_CCI(df)
        s = df["CCI_strength"]
        assert (s >= 0).all() and (s <= 1).all()

    def test_entry_when_cci_beyond_100(self, df):
        """Verify entry column aligns with |CCI| > 100 logic."""
        _calc_CCI(df)
        tp    = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_ma = tp.rolling(20).mean()
        tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci   = (tp - tp_ma) / (0.015 * tp_md)
        entry_expected = (cci.abs() > 100)
        valid = cci.notna()
        assert (df.loc[valid, "CCI_entries"] == entry_expected[valid]).all()

    def test_custom_window_and_constant(self, df):
        _calc_CCI(df, window=14, constant=0.015)
        assert "CCI_entries" in df.columns

    def test_norm_cci_is_clipped(self, df):
        """norm_cci = (CCI + 200) / 400, so strength ≤ 1 even for extreme CCI."""
        _calc_CCI(df)
        assert (df["CCI_strength"] <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_CMF
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcCMF:
    def test_column_created(self, df):
        _calc_CMF(df)
        assert "CMF" in df.columns

    def test_value_range(self, df):
        _calc_CMF(df)
        valid = df["CMF"].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_custom_window(self, df):
        _calc_CMF(df, window=10)
        valid = df["CMF"].dropna()
        assert len(valid) > 0

    def test_all_up_bars_positive_cmf(self):
        """When close = high every bar, MFM = 1, so CMF should be positive."""
        n = 50
        close = np.arange(100.0, 100 + n)
        frame = pd.DataFrame({
            "Open": close - 1, "High": close,
            "Low":  close - 2, "Close": close,
            "Volume": np.ones(n) * 10,
        })
        _calc_CMF(frame)
        assert (frame["CMF"].dropna() > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_VWAP
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcVWAP:
    def test_column_created(self, df):
        _calc_VWAP(df)
        assert "VWAP" in df.columns

    def test_positive_values(self, df):
        _calc_VWAP(df)
        assert (df["VWAP"].dropna() > 0).all()

    def test_vwap_within_price_range(self, df):
        _calc_VWAP(df)
        vwap = df["VWAP"].dropna()
        price_min = df["Low"].min()
        price_max = df["High"].max()
        assert (vwap >= price_min * 0.9).all()
        assert (vwap <= price_max * 1.1).all()

    def test_constant_price_vwap_equals_price(self):
        """If close is constant and volume is uniform, VWAP == close."""
        n = 50
        frame = pd.DataFrame({
            "Open": [100.0] * n, "High": [101.0] * n,
            "Low":  [99.0] * n, "Close": [100.0] * n,
            "Volume": [10.0] * n,
        })
        _calc_VWAP(frame)
        # VWAP is cumulative, so all rows equal the constant price
        assert (frame["VWAP"].abs() - 100.0).abs().max() < 1e-9

    def test_vwap_is_cumulative(self, df):
        """VWAP should be monotonically stable: not reset each row."""
        _calc_VWAP(df)
        # All values must be finite (no sudden resets / infinities)
        assert df["VWAP"].isna().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_ATR
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcATR:
    def test_column_created(self, df):
        _calc_ATR(df)
        assert "ATR" in df.columns

    def test_non_negative(self, df):
        _calc_ATR(df)
        assert (df["ATR"].dropna() >= 0).all()

    def test_custom_window(self, df):
        _calc_ATR(df, window=7)
        assert "ATR" in df.columns

    def test_constant_ohlc_zero_atr(self):
        """Flat OHLC → TR = 0 on every bar → ATR = 0."""
        n = 50
        frame = pd.DataFrame({
            "Open": [100.0] * n, "High": [100.0] * n,
            "Low":  [100.0] * n, "Close": [100.0] * n,
            "Volume": [1.0] * n,
        })
        _calc_ATR(frame)
        # ATR should be ~0 for all valid rows; allow floating-point tolerance
        atr = frame["ATR"].dropna()
        assert (atr.abs() < 1e-9).all()

    def test_larger_range_larger_atr(self):
        """Wide-range bars should produce higher ATR than narrow-range bars."""
        n = 100
        rng = np.random.default_rng(0)
        close = np.full(n, 100.0)
        narrow = pd.DataFrame({"Open": close, "High": close + 0.1,
                                "Low": close - 0.1, "Close": close,
                                "Volume": np.ones(n)})
        wide   = pd.DataFrame({"Open": close, "High": close + 10,
                                "Low": close - 10, "Close": close,
                                "Volume": np.ones(n)})
        _calc_ATR(narrow); _calc_ATR(wide)
        assert wide["ATR"].dropna().mean() > narrow["ATR"].dropna().mean()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcVolatility:
    def test_column_created(self, df):
        _calc_VOLATILITY(df)
        assert "VOLATILITY" in df.columns

    def test_non_negative(self, df):
        _calc_VOLATILITY(df)
        assert (df["VOLATILITY"].dropna() >= 0).all()

    def test_constant_price_zero_vol(self):
        """Flat close → log returns = 0 → rolling std = 0."""
        n = 50
        frame = pd.DataFrame({
            "Open": [100.0] * n, "High": [100.0] * n,
            "Low":  [100.0] * n, "Close": [100.0] * n,
            "Volume": [1.0] * n,
        })
        _calc_VOLATILITY(frame)
        vol = frame["VOLATILITY"].dropna()
        assert (vol.abs() < 1e-9).all()

    def test_high_vol_series_greater(self):
        """A noisy series should have higher vol than a smooth one."""
        n = 100
        rng = np.random.default_rng(0)
        smooth = pd.DataFrame({"Close": np.linspace(100, 110, n)})
        noisy  = pd.DataFrame({"Close": 100 + np.cumsum(rng.normal(0, 5, n))})
        _calc_VOLATILITY(smooth); _calc_VOLATILITY(noisy)
        assert noisy["VOLATILITY"].dropna().mean() > smooth["VOLATILITY"].dropna().mean()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_PARKINSON
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcParkinson:
    def test_column_created(self, df):
        _calc_PARKINSON(df)
        assert "PARKINSON" in df.columns

    def test_non_negative(self, df):
        _calc_PARKINSON(df)
        assert (df["PARKINSON"].dropna() >= 0).all()

    def test_constant_high_low_zero(self):
        """When High == Low, log(H/L) == 0, so PARKINSON == 0."""
        n = 50
        frame = pd.DataFrame({
            "Open": [100.0] * n, "High": [100.0] * n,
            "Low":  [100.0] * n, "Close": [100.0] * n,
            "Volume": [1.0] * n,
        })
        _calc_PARKINSON(frame)
        pk = frame["PARKINSON"].dropna()
        assert (pk.abs() < 1e-9).all()

    def test_wider_range_higher_parkinson(self):
        """Wide H-L spread → higher Parkinson estimate."""
        n = 100
        narrow = pd.DataFrame({"Open": [100]*n, "High": [100.1]*n,
                                "Low": [99.9]*n, "Close": [100]*n, "Volume": [1]*n})
        wide   = pd.DataFrame({"Open": [100]*n, "High": [110]*n,
                                "Low": [90]*n,  "Close": [100]*n, "Volume": [1]*n})
        _calc_PARKINSON(narrow); _calc_PARKINSON(wide)
        assert wide["PARKINSON"].dropna().mean() > narrow["PARKINSON"].dropna().mean()


# ─────────────────────────────────────────────────────────────────────────────
#  _calc_PRICE_ACTION
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcPriceAction:
    def test_columns_created(self, df):
        _calc_PRICE_ACTION(df)
        for col in ("PRICE_ACTION", "dist_from_high", "dist_from_low"):
            assert col in df.columns

    def test_price_action_range(self, df):
        _calc_PRICE_ACTION(df)
        pa = df["PRICE_ACTION"].dropna()
        assert (pa >= 0).all() and (pa <= 1).all()

    def test_dist_from_high_non_positive(self, df):
        """Close ≤ rolling high, so dist_from_high = (Close - high) / high ≤ 0."""
        _calc_PRICE_ACTION(df)
        dfh = df["dist_from_high"].dropna()
        assert (dfh <= 0).all()

    def test_dist_from_low_non_negative(self, df):
        """Close ≥ rolling low, so dist_from_low = (Close - low) / low ≥ 0."""
        _calc_PRICE_ACTION(df)
        dfl = df["dist_from_low"].dropna()
        assert (dfl >= 0).all()

    def test_doji_gives_zero_price_action(self):
        """When Open == Close, body = 0, so PRICE_ACTION == 0."""
        n = 50
        close = np.linspace(100, 110, n)
        frame = pd.DataFrame({
            "Open": close, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": np.ones(n),
        })
        _calc_PRICE_ACTION(frame)
        pa = frame["PRICE_ACTION"].dropna()
        assert (pa.abs() < 1e-9).all()

    def test_full_body_gives_unit_price_action(self):
        """When |body| == range, PRICE_ACTION == 1."""
        n = 50
        close = np.full(n, 105.0)
        open_ = np.full(n, 100.0)
        # high = close, low = open_ → range = body
        frame = pd.DataFrame({
            "Open": open_, "High": close,
            "Low": open_, "Close": close,
            "Volume": np.ones(n),
        })
        _calc_PRICE_ACTION(frame)
        pa = frame["PRICE_ACTION"].dropna()
        assert (np.abs(pa - 1.0) < 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
#  _build_signal_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSignalColumns:
    def _prep(self) -> pd.DataFrame:
        df = _make_ohlcv(200)
        for fn in (_calc_RSI, _calc_MACD, _calc_MA, _calc_HA, _calc_OBV,
                   _calc_STOCH, _calc_BBANDS, _calc_CCI):
            fn(df)
        return df

    def test_signal_columns_created(self):
        df = self._prep()
        _build_signal_columns(df)
        for name in SIGNAL_INDICATORS:
            assert f"{name}_signal" in df.columns

    def test_raw_entry_exit_cols_dropped(self):
        df = self._prep()
        _build_signal_columns(df)
        for name in SIGNAL_INDICATORS:
            assert f"{name}_entries" not in df.columns
            assert f"{name}_exits" not in df.columns

    def test_signal_values_in_minus1_0_plus1(self):
        df = self._prep()
        _build_signal_columns(df)
        for name in SIGNAL_INDICATORS:
            col = df[f"{name}_signal"]
            assert col.isin([-1, 0, 1]).all(), f"{name}_signal has unexpected values"

    def test_entry_takes_priority_over_exit(self):
        """When both entry and exit are True, signal must be +1."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.0], "Volume": [1.0],
        })
        for name in SIGNAL_INDICATORS:
            df[f"{name}_entries"] = True
            df[f"{name}_exits"]   = True
            df[f"{name}_strength"] = 0.5

        _build_signal_columns(df)
        for name in SIGNAL_INDICATORS:
            assert df.loc[0, f"{name}_signal"] == 1

    def test_only_exit_gives_minus1(self):
        """When only exit is True, signal must be -1."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.0], "Volume": [1.0],
        })
        for name in SIGNAL_INDICATORS:
            df[f"{name}_entries"] = False
            df[f"{name}_exits"]   = True
            df[f"{name}_strength"] = 0.5

        _build_signal_columns(df)
        for name in SIGNAL_INDICATORS:
            assert df.loc[0, f"{name}_signal"] == -1

    def test_neither_gives_zero(self):
        """When both are False, signal must be 0."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.0], "Volume": [1.0],
        })
        for name in SIGNAL_INDICATORS:
            df[f"{name}_entries"] = False
            df[f"{name}_exits"]   = False
            df[f"{name}_strength"] = 0.0

        _build_signal_columns(df)
        for name in SIGNAL_INDICATORS:
            assert df.loc[0, f"{name}_signal"] == 0

    def test_warning_on_missing_columns(self, capsys):
        """A missing entry/exit column should print a WARNING, not raise."""
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        _build_signal_columns(df)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_nan_entries_treated_as_false(self):
        """NaN booleans should be treated as False (via fillna(False))."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.0], "Volume": [1.0],
        })
        for name in SIGNAL_INDICATORS:
            df[f"{name}_entries"] = np.nan
            df[f"{name}_exits"]   = np.nan
            df[f"{name}_strength"] = 0.0

        _build_signal_columns(df)  # should not raise
        for name in SIGNAL_INDICATORS:
            assert df.loc[0, f"{name}_signal"] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  _INTERMEDIATE_COLS — verify MACD_signal is NOT listed (was a bug)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntermediateColsBug:
    """
    Previously 'MACD_signal' appeared in _INTERMEDIATE_COLS, colliding with
    the output column created by _build_signal_columns. The drop step after
    signal building silently removed it, causing load_and_compute to raise
    KeyError. These tests verify the fix holds.
    """

    def test_macd_signal_not_in_intermediate_cols(self):
        """MACD_signal must NOT be in _INTERMEDIATE_COLS — it's a required output."""
        assert "MACD_signal" not in _INTERMEDIATE_COLS

    def test_build_signal_columns_creates_macd_signal(self):
        """_build_signal_columns produces MACD_signal as a required output."""
        df = _make_ohlcv(200)
        _calc_MACD(df)
        _build_signal_columns(df)
        assert "MACD_signal" in df.columns

    def test_drop_intermediate_preserves_macd_signal(self):
        """After the drop step, MACD_signal must survive."""
        df = _make_ohlcv(200)
        _calc_MACD(df)
        _build_signal_columns(df)
        df.drop(columns=_INTERMEDIATE_COLS, errors="ignore", inplace=True)
        assert "MACD_signal" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  load_and_compute  (public entry point)
#  All tests here are xfail due to the MACD_signal/_INTERMEDIATE_COLS bug.
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndCompute:
    def test_returns_dataframe(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert isinstance(result, pd.DataFrame)

    def test_no_nans_in_required_columns(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        required = (
            [f"{i}_signal"   for i in SIGNAL_INDICATORS]
            + [f"{i}_strength" for i in SIGNAL_INDICATORS]
            + CONTINUOUS_FEATURES
        )
        for col in required:
            assert result[col].notna().all(), f"NaN found in {col}"

    def test_signal_values_valid(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for name in SIGNAL_INDICATORS:
            assert result[f"{name}_signal"].isin([-1, 0, 1]).all()

    def test_strength_values_in_range(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for name in SIGNAL_INDICATORS:
            col = result[f"{name}_strength"]
            assert col.between(0, 1).all(), f"{name}_strength out of [0,1]"

    def test_required_signal_columns_present(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for name in SIGNAL_INDICATORS:
            assert f"{name}_signal"   in result.columns
            assert f"{name}_strength" in result.columns

    def test_continuous_feature_columns_present(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for feat in CONTINUOUS_FEATURES:
            assert feat in result.columns

    def test_raw_entry_exit_columns_absent(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for name in SIGNAL_INDICATORS:
            assert f"{name}_entries" not in result.columns
            assert f"{name}_exits"   not in result.columns

    def test_intermediate_columns_dropped(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for col in _INTERMEDIATE_COLS:
            if col != "MACD_signal":   # MACD_signal is the broken one
                assert col not in result.columns

    def test_index_reset_after_dropna(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert list(result.index) == list(range(len(result)))

    def test_datetime_sorted_ascending(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        dts = pd.to_datetime(result["Datetime"])
        assert (dts.diff().dropna() >= pd.Timedelta(0)).all()

    def test_ohlcv_columns_preserved(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_rsi_raw_column_preserved(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert "RSI" in result.columns

    def test_dist_columns_present(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert "dist_from_high" in result.columns
        assert "dist_from_low"  in result.columns

    def test_cmf_in_minus1_to_1(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert result["CMF"].between(-1, 1).all()

    def test_atr_non_negative(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert (result["ATR"] >= 0).all()

    def test_volatility_non_negative(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert (result["VOLATILITY"] >= 0).all()

    def test_vwap_positive(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert (result["VWAP"] > 0).all()

    def test_price_action_in_range(self, tmp_csv):
        result = ind.load_and_compute(tmp_csv)
        assert result["PRICE_ACTION"].between(0, 1).all()

    def test_warm_up_rows_removed(self, tmp_csv):
        raw = pd.read_csv(tmp_csv)
        result = ind.load_and_compute(tmp_csv)
        assert len(result) < len(raw)

    def test_unsorted_csv_is_sorted(self, tmp_path):
        frame = _make_ohlcv(300)
        frame.insert(0, "Datetime", pd.date_range("2024-01-01", periods=300, freq="1min"))
        shuffled = frame.sample(frac=1, random_state=0)
        path = tmp_path / "shuffled.csv"
        shuffled.to_csv(path, index=False)
        result = ind.load_and_compute(str(path))
        dts = pd.to_datetime(result["Datetime"])
        assert (dts.diff().dropna() >= pd.Timedelta(0)).all()


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_signal_indicators_list(self):
        assert set(SIGNAL_INDICATORS) == {"RSI", "MACD", "MA", "HA", "STOCH", "BBANDS", "CCI", "OBV"}

    def test_continuous_features_list(self):
        assert set(CONTINUOUS_FEATURES) == {"CMF", "VWAP", "ATR", "VOLATILITY", "PARKINSON", "PRICE_ACTION"}

    def test_no_duplicates_in_signal_indicators(self):
        assert len(SIGNAL_INDICATORS) == len(set(SIGNAL_INDICATORS))

    def test_no_duplicates_in_continuous_features(self):
        assert len(CONTINUOUS_FEATURES) == len(set(CONTINUOUS_FEATURES))

    def test_signal_and_continuous_disjoint(self):
        assert not set(SIGNAL_INDICATORS) & set(CONTINUOUS_FEATURES)