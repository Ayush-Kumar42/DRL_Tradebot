import numpy as np
from scipy.stats import entropy
import ruptures as rpt
from hurst import compute_Hc
import gymnasium as gym
from gymnasium import spaces

from businesslogic import BusinessLogic, Position, CloseStrategy

# Numerical floor used whenever we divide or take a log of a value that could
# legitimately be zero (e.g. cost basis, portfolio value during a guard).
_EPS = 1e-8


class TradingEnv(gym.Env):
    """
    Gymnasium trading environment.

    All trade execution (open, close, trailing stop) is delegated to
    BusinessLogic.  The environment is responsible only for:
      - Building observations (normalized — see `Observation normalization`)
      - Computing the Hurst exponent / regime
      - Advancing `current_step`
      - Computing reward (normalized log-return of portfolio valuation,
        plus a per-position "holding reward" closing bonus)
      - Formatting the `info` dict for the caller

    Reward design
    --------------
    1. Global step reward:
           ln(portfolio_value_after_step / portfolio_value_before_step)
       computed once per step, guarded against non-positive values.

    2. Per-position holding reward (logged every step a position stays open,
       NOT added to reward directly):
           ln(current_value / previous_value)
       where both current_value and previous_value are anchored to the
       position's cost basis at *current* volume (p.volume * p.price), and
       the incremental step PnL uses the change in mark price since the
       previous step (not the cumulative PnL since entry). This makes the
       per-position log a true per-step series, suitable for averaging.

    3. Closing bonus (manual/strategy closes only - i.e. execute_trade,
       NOT trailing stops or force-close):
           mean(holding_log[order]) * (closed_qty / volume_before_close)
       added once per close event into that step's total reward. On a
       partial close the log is left untouched for the remaining open
       volume (no reset) since the weighting by closed_qty already accounts
       for how much of the position is being "cashed in".

    Observation normalization
    --------------------------
    Raw OHLCV/price-derived features are non-stationary (price drifts over
    the life of a dataset), so they cannot be safely normalized with a
    single global statistic fit once at `reset()` — that would both leak
    future information into early steps and go stale by the end of a long
    episode. Instead every numeric observation field is normalized
    *causally*: any statistic used (rolling mean/std, rolling min/max) is
    computed only from data up to and including `current_step`, using a
    trailing window (`norm_window`, default 200 bars). Three transform
    families are used, chosen per-feature by its statistical shape:

      - "ratio_log"  : log of the ratio to a trailing reference, e.g.
                       ln(Close / Close.rolling.mean()). Removes the price
                       level entirely so the network sees stationary,
                       roughly-zero-centered relative moves regardless of
                       whether BTC is at 20k or 90k. Used for Open/High/
                       Low/Close/VWAP.
      - "log_zscore" : log1p (to tame heavy right tails) then a rolling
                       z-score. Used for strictly-positive, skewed
                       magnitude features: Volume, ATR, PARKINSON.
      - "zscore"     : plain rolling z-score, no log. Used for features
                       that are already roughly symmetric but not
                       naturally bounded: VOLATILITY.
      - "passthrough": already on a clean, bounded, semantically-meaningful
                       scale, so rescaling would only relabel it without
                       adding information. Used for CMF ([-1, 1]),
                       PRICE_ACTION ([0, 1]), and every `*_signal` column
                       ({-1, 0, +1}).

    All rolling statistics use only `df` rows `[0, current_step]` (inclusive)
    so normalization at any step is reproducible from data the agent could
    actually have seen up to that point.

    Random-window evaluation
    -------------------------
    When `random_start=True`, each call to `reset()` selects a contiguous
    slice of `window_size` rows starting at a uniformly-random offset within
    the full dataset.  The full dataset is stored in `_full_df`; `self.df`
    is replaced with the slice on every reset so all downstream code
    (BusinessLogic, normalization, step indexing) operates identically to
    a fixed-window run.  `random_start=False` (default) preserves the
    original behaviour exactly.

    Parameters
    ----------
    random_start : bool
        If True, pick a random window each episode.  Default False.
    window_size : int
        Number of rows per episode when `random_start=True`.
        Ignored when `random_start=False`.  Must be ≤ len(df).
    """

    # Per-feature normalization strategy. Anything in `continuous_features`
    # or the OHLC/Volume block not listed here defaults to "zscore" as a
    # safe fallback (see `_normalize_value`).
    _PRICE_RATIO_FEATURES = {"Open", "High", "Low", "Close", "VWAP"}
    _LOG_ZSCORE_FEATURES = {"Volume", "ATR", "PARKINSON"}
    _ZSCORE_FEATURES = {"VOLATILITY"}
    _PASSTHROUGH_FEATURES = {"CMF", "PRICE_ACTION"}

    def __init__(
        self,
        df,
        T_indicators: list[str],
        MR_indicators: list[str],
        continuous_features: list[str],
        initial_balance: float,
        close_strategy: CloseStrategy = "fifo",
        trail_pct: float = 0.05,
        norm_window: int = 200,
        norm_eps: float = 1e-8,
        random_start: bool = False,
        window_size: int = 5_000,
    ):
        super().__init__()

        # --- Data & feature config ---
        # _full_df always holds the complete dataset passed in at construction.
        # self.df is the active slice used by this episode; when random_start
        # is False it stays identical to _full_df across all episodes.
        self._full_df = df.reset_index(drop=True)
        self.df = self._full_df          # will be replaced in reset() if random_start
        self.T_indicators = T_indicators
        self.MR_indicators = MR_indicators
        self.continuous_features = continuous_features
        self.initial_balance = initial_balance
        self.close_strategy = close_strategy
        self.trail_pct = trail_pct

        # --- Random-window config ---
        self.random_start = random_start
        self.window_size  = window_size
        if random_start and window_size > len(self._full_df):
            raise ValueError(
                f"window_size={window_size} exceeds dataset length {len(self._full_df)}. "
                f"Reduce window_size or pass a larger DataFrame."
            )

        # Internal RNG for window selection; seeded properly in reset().
        self._window_rng = np.random.default_rng()

        # --- Normalization config ---
        # `norm_window` bars of trailing history are used to compute rolling
        # mean/std (or rolling mean reference, for the ratio transform).
        # Must be causal: only ever indexes df[max(0, t - window + 1) : t+1].
        self.norm_window = norm_window
        self.norm_eps = norm_eps

        # --- Action space: one weight per indicator + volume fraction ---
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(T_indicators) + 1,),
            dtype=np.float32,
        )

        # --- Observation space ---
        # continuous_features  +  OHLCV (5)  +  regime indicator signals
        # Bounds reflect the *normalized* ranges, not raw data:
        #   - ratio_log / log_zscore / zscore features are unbounded in
        #     principle (clipped in practice, see `_normalize_value`) ->
        #     [-inf, inf] is the honest declared bound.
        #   - passthrough features (CMF, PRICE_ACTION, *_signal) keep their
        #     natural bounds.
        obs_dim = len(continuous_features) + 5 + len(T_indicators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Populated in reset()
        self.bl: BusinessLogic | None = None
        self.positions: list[Position] = []
        self.current_step: int = 0
        self.Hurst: float = 0.5
        self._realised_pnl_acc: float = 0.0   # accumulates all closed-trade PnL

        # --- Per-position holding-reward bookkeeping ---
        # order -> list of per-step incremental log returns logged while open
        self._holding_log: dict[int, list[float]] = {}
        # order -> mark price used as the "previous" reference for the next
        # incremental log-return calculation
        self._prev_mark_price: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    #  Utility helpers                                                     #
    # ------------------------------------------------------------------ #

    def calculate_entropy(self, series):
        value_counts = series.value_counts(normalize=True)
        return entropy(value_counts, base=2) if len(value_counts) > 1 else 0

    def calculate_hurst(self, time_series) -> float:
        H, _c, _data = compute_Hc(time_series, kind="price", simplified=True)
        return H

    def detect_change_points(self, time_series, penalty: int = 10):
        time_series_2d = time_series.values.reshape(-1, 1)
        algo = rpt.Pelt(model="l2").fit(time_series_2d)
        return algo.predict(pen=penalty)

    def _regime(self) -> str:
        return "trending" if self.Hurst > 0.5 else "mean_reverting"

    def _update_hurst(self):
        """Recompute Hurst using all data up to the current step (min 100 bars)."""
        min_window = 100
        if self.current_step >= min_window:
            self.Hurst = self.calculate_hurst(
                self.df["Close"].iloc[: self.current_step]
            )

    @staticmethod
    def _safe_log_return(current_value: float, previous_value: float) -> float:
        """
        ln(current_value / previous_value), guarded against non-positive
        values on either side. Returns 0.0 in degenerate cases rather than
        propagating NaN/-inf into training.
        """
        denom = previous_value if previous_value > _EPS else _EPS
        numer = current_value if current_value > _EPS else _EPS
        return float(np.log(numer / denom))

    # ------------------------------------------------------------------ #
    #  Observation normalization                                          #
    # ------------------------------------------------------------------ #

    def _causal_window(self, column: str) -> np.ndarray:
        """
        Trailing window of `column` ending at (and including) current_step,
        length `min(norm_window, current_step + 1)`. Always causal: never
        touches rows beyond current_step.
        """
        start = max(0, self.current_step - self.norm_window + 1)
        end = self.current_step + 1  # inclusive of current_step
        return self.df[column].iloc[start:end].to_numpy(dtype=np.float64)

    def _normalize_value(self, column: str, raw_value: float) -> float:
        """
        Normalize a single scalar observation field using only causal
        (trailing, up-to-current_step) statistics. Dispatches by feature
        name to one of four transforms; see class docstring for rationale.
        """
        # Signal columns (already {-1, 0, +1}) and other declared
        # passthrough features need no rescaling.
        if column in self._PASSTHROUGH_FEATURES or column.endswith("_signal"):
            return float(raw_value)

        window = self._causal_window(column)

        if column in self._PRICE_RATIO_FEATURES:
            return self._ratio_log(raw_value, window)

        if column in self._LOG_ZSCORE_FEATURES:
            return self._log_zscore(raw_value, window)

        # _ZSCORE_FEATURES and anything else unrecognised -> plain z-score,
        # the safest general-purpose fallback for an unbounded continuous
        # feature we don't have special-case knowledge about.
        return self._zscore(raw_value, window)

    def _ratio_log(self, raw_value: float, window: np.ndarray) -> float:
        """
        ln(raw_value / trailing_mean(window)). Removes the absolute price
        level (non-stationary) and leaves a stationary, roughly-zero-
        centered relative measure. Clipped to +/-5 (~e^5 ~ 148x) to bound
        the effect of any single bad tick or near-zero reference.
        """
        ref = float(np.mean(window))
        ref = ref if abs(ref) > self.norm_eps else self.norm_eps
        val = raw_value if raw_value > self.norm_eps else self.norm_eps
        result = float(np.log(val / ref))
        return float(np.clip(result, -5.0, 5.0))

    def _log_zscore(self, raw_value: float, window: np.ndarray) -> float:
        """
        log1p(raw_value) z-scored against the log1p'd trailing window.
        log1p tames heavy right tails (Volume, ATR, PARKINSON are all
        strictly non-negative and can spike by orders of magnitude), and
        the subsequent z-score gives the network a comparable, roughly
        unit-scale signal regardless of the asset's native volume/range.
        Clipped to +/-5 standard deviations.
        """
        log_window = np.log1p(np.clip(window, 0.0, None))
        mean = float(np.mean(log_window))
        std = float(np.std(log_window))
        std = std if std > self.norm_eps else self.norm_eps
        log_val = float(np.log1p(max(raw_value, 0.0)))
        result = (log_val - mean) / std
        return float(np.clip(result, -5.0, 5.0))

    def _zscore(self, raw_value: float, window: np.ndarray) -> float:
        """
        Plain rolling z-score: (raw_value - trailing_mean) / trailing_std.
        Used for features that are already roughly symmetric (no heavy
        one-sided tail) but not naturally bounded, e.g. VOLATILITY.
        Clipped to +/-5 standard deviations.
        """
        mean = float(np.mean(window))
        std = float(np.std(window))
        std = std if std > self.norm_eps else self.norm_eps
        result = (raw_value - mean) / std
        return float(np.clip(result, -5.0, 5.0))

    # ------------------------------------------------------------------ #
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def get_observation(self, regime: str) -> np.ndarray:
        row = self.df.iloc[self.current_step]

        obs: list[float] = []

        # continuous_features: each gets its dedicated normalization.
        for col in self.continuous_features:
            obs.append(self._normalize_value(col, float(row[col])))

        # OHLCV: O/H/L/C normalized via price-ratio, Volume via log-zscore.
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            obs.append(self._normalize_value(col, float(row[col])))

        # Regime-specific indicator signals: already {-1, 0, +1}, passthrough.
        indicators = self.T_indicators if regime == "trending" else self.MR_indicators
        signal_cols = [f"{ind}_signal" for ind in indicators]
        for col in signal_cols:
            obs.append(self._normalize_value(col, float(row[col])))

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Position-change diff helper                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _diff_positions(
        before: list[Position],
        after: list[Position],
        price: float,
        step: int,
    ) -> dict | None:
        """
        Compare position lists to produce an opened/closed summary dict
        compatible with the original info["position_changes"] schema.
        """
        # Opened: present in after but not before (new object -> new id)
        before_orders = {p.order for p in before}
        opened = []
        for p in after:
            if p.order not in before_orders:
                opened.append({
                    "id":          p.order,
                    "entry_price": p.price,
                    "quantity":    p.volume,
                    "entry_step":  step,
                })

        # Closed / partially reduced: order existed before, volume shrank or gone
        after_by_order = {p.order: p for p in after}
        closed = []
        for p in before:
            after_p = after_by_order.get(p.order)
            closed_qty = p.volume if after_p is None else p.volume - after_p.volume
            if closed_qty > 0:
                trade_pnl = closed_qty * (price - p.price) * p.signal
                closed.append({
                    "id":             p.order,
                    "entry_price":    p.price,
                    "exit_price":     price,
                    "quantity":       closed_qty,
                    "pnl":            trade_pnl,
                    "pnl_percent":    (price - p.price) / p.price * 100 if p.price else 0,
                    "holding_period": step - getattr(p, "entry_step", 0),
                    "close_reason":   "trade",
                    "volume_before_close": p.volume,
                })

        if not opened and not closed:
            return None
        return {"opened": opened, "closed": closed}

    # ------------------------------------------------------------------ #
    #  Holding-reward bookkeeping                                          #
    # ------------------------------------------------------------------ #

    def _update_holding_log(self, price: float) -> None:
        """
        For every currently open position, append this step's incremental
        log return to its holding log, and roll forward its previous mark
        price. Must be called once per step, on the position list as it
        stands *before* any closes are applied this step, using the same
        `price` that step() uses for PnL (current bar's Open).

        Newly-opened positions (no prior mark price recorded) are seeded
        with their entry price as the initial reference and do not log a
        return on the step they were opened (no prior step to compare to).
        """
        for p in self.positions:
            cost_basis = p.volume * p.price
            prev_mark = self._prev_mark_price.get(p.order)

            if prev_mark is None:
                # First time we see this position: seed reference, no log entry yet.
                self._prev_mark_price[p.order] = p.price
                self._holding_log.setdefault(p.order, [])
                continue

            step_pnl_increment = p.volume * (price - prev_mark) * p.signal
            current_value = cost_basis + step_pnl_increment
            log_return = self._safe_log_return(current_value, cost_basis)

            self._holding_log.setdefault(p.order, []).append(log_return)
            self._prev_mark_price[p.order] = price

    def _closing_bonus(self, closed_entries: list[dict]) -> float:
        """
        For a list of closed-position dicts (as produced by _diff_positions),
        compute the sum of closing bonuses:
            mean(holding_log[order]) * (closed_qty / volume_before_close)
        Only applies to entries whose order has a non-empty holding log;
        positions closed on the same step they were opened contribute 0.
        """
        bonus_total = 0.0
        for entry in closed_entries:
            order = entry["id"]
            log = self._holding_log.get(order)
            if not log:
                continue
            volume_before = entry.get("volume_before_close", 0.0)
            if volume_before <= 0:
                continue
            weight = entry["quantity"] / volume_before
            bonus_total += float(np.mean(log)) * weight
        return bonus_total

    def _cleanup_closed_orders(self) -> None:
        """Drop bookkeeping for positions that are fully closed (no longer open)."""
        open_orders = {p.order for p in self.positions}
        for order in list(self._holding_log.keys()):
            if order not in open_orders:
                del self._holding_log[order]
                self._prev_mark_price.pop(order, None)

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # ── Random-window selection ───────────────────────────────────────
        # When random_start is enabled, pick a contiguous slice of
        # window_size rows at a uniformly-random offset within _full_df.
        # Re-seed the internal RNG from the Gymnasium seed when one is
        # supplied so that eval runs remain reproducible across calls.
        if self.random_start:
            if seed is not None:
                self._window_rng = np.random.default_rng(seed)
            max_start = len(self._full_df) - self.window_size
            start     = int(self._window_rng.integers(0, max_start + 1))
            self.df   = self._full_df.iloc[start : start + self.window_size].reset_index(drop=True)
        else:
            # Fixed mode: always use the full dataset, same as before.
            self.df = self._full_df

        # ── Standard reset logic (unchanged) ─────────────────────────────
        self.current_step = 0
        self.positions = []
        self._realised_pnl_acc = 0.0   # reset accumulator each episode
        self._holding_log = {}
        self._prev_mark_price = {}

        # Bootstrap BusinessLogic; balance is the portfolio valuation tracked here.
        # trail_pct is passed in so BusinessLogic._register_stop uses the same
        # value as the environment was configured with.
        self.bl = BusinessLogic(
            df=self.df,
            balance=self.initial_balance,
            current_step=self.current_step,
            T_indicators=self.T_indicators,
            MR_indicators=self.MR_indicators,
        )
        self.bl.trail_pct = self.trail_pct

        # Hurst bootstrap
        min_window = 100
        if len(self.df) < min_window:
            raise ValueError(
                f"DataFrame must have at least {min_window} rows for Hurst calculation."
            )
        self.Hurst = self.calculate_hurst(self.df["Close"].iloc[:min_window])

        regime = self._regime()
        obs = self.get_observation(regime)
        info = {
            "portfolio_value":     self.initial_balance,
            "position_changes":    None,
            "current_price":       self.df.iloc[0]["Open"],
            "global_log_return":   0.0,
            "closing_bonus_total": 0.0,
        }
        return obs, info

    def step(self, action):
        assert self.bl is not None, "Call reset() before step()."

        self.bl.current_step = self.current_step
        self.bl.balance      = self._portfolio_valuation()

        price = self.df.iloc[self.current_step]["Open"]

        # Snapshot portfolio value BEFORE any trades/closes this step.
        prev_portfolio_value = self._portfolio_valuation()

        # Log this step's incremental holding return for every currently
        # open position, before any closes are applied below.
        self._update_holding_log(price)

        closing_bonus_total = 0.0
        all_opened: list[dict] = []
        all_closed: list[dict] = []

        # 1. Trailing stops (no closing bonus - see class docstring)
        before_trailing = list(self.positions)
        trailing_pnl, self.positions = self.bl.check_trailing_stops(
            self.positions, price
        )
        if trailing_pnl != 0.0:
            diff = self._diff_positions(before_trailing, self.positions, price, self.current_step)
            if diff:
                for c in diff["closed"]:
                    c["close_reason"] = "trailing_stop"
                all_closed.extend(diff["closed"])
            self._cleanup_closed_orders()

        # 2. Hurst / regime
        self._update_hurst()
        regime = self._regime()

        # 3. Execute trade (manual/strategy close -> eligible for closing bonus).
        # Skipped when trailing stops already fired this step to avoid acting
        # on a position list that has just been materially changed.
        if trailing_pnl == 0.0:
            indicator_vector = action[:-1]
            indicator_list   = self.T_indicators if regime == "trending" else self.MR_indicators
            strong_inds      = [i for i, w in enumerate(indicator_vector) if w > 0.5]
            all_zero = all(
                self.df.iloc[self.current_step].get(indicator_list[i], 0) == 0
                for i in strong_inds
            )

            if strong_inds and not all_zero:
                before_trade = list(self.positions)
                self.bl.balance      = self._portfolio_valuation()
                self.bl.current_step = self.current_step

                self.positions, trade_pnl = self.bl.execute_trade(
                    positions=self.positions,
                    action=action,
                    regime=regime,
                    strategy=self.close_strategy,
                )

                diff = self._diff_positions(before_trade, self.positions, price, self.current_step)
                if diff:
                    for c in diff["closed"]:
                        c["close_reason"] = "manual_sell"
                    all_opened.extend(diff["opened"])
                    all_closed.extend(diff["closed"])
                    closing_bonus_total += self._closing_bonus(diff["closed"])
                self._cleanup_closed_orders()

        # 4. Accumulate realised PnL (for portfolio valuation bookkeeping only;
        #    no longer drives reward directly - reward comes from the log
        #    return of portfolio valuation, computed below).
        realised_this_step = sum(c["pnl"] for c in all_closed)
        self._realised_pnl_acc += realised_this_step

        # 5. Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated  = False

        # 6. Force-close at episode end (no closing bonus - see class docstring).
        # force_close_all also clears price_peak / price_trough / heaps in bl.
        if terminated and self.positions:
            self.bl.current_step = self.current_step - 1
            before_fc = list(self.positions)
            fc_pnl, self.positions = self.bl.force_close_all(self.positions)
            self._realised_pnl_acc += fc_pnl
            diff = self._diff_positions(
                before_fc, self.positions,
                self.df.iloc[self.current_step - 1]["Close"],
                self.current_step,
            )
            if diff:
                for c in diff["closed"]:
                    c["close_reason"] = "force_close_end"
                all_closed.extend(diff["closed"])
            self._cleanup_closed_orders()

        # 7. Global reward: normalized log return of portfolio valuation,
        #    plus the closing bonus(es) earned this step.
        current_portfolio_value = self._portfolio_valuation()
        global_log_return = self._safe_log_return(current_portfolio_value, prev_portfolio_value)
        reward = global_log_return + closing_bonus_total

        # 8. Build info
        obs = self.get_observation(regime)
        position_changes = None
        if all_opened or all_closed:
            position_changes = {"opened": all_opened, "closed": all_closed}

        info = {
            "portfolio_value":     current_portfolio_value,
            "position_changes":    position_changes,
            "current_price":       price,
            "global_log_return":   global_log_return,
            "closing_bonus_total": closing_bonus_total,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _portfolio_valuation(self) -> float:
        """
        Mark-to-market portfolio value:
          initial_balance
          + all realised PnL from closed trades this episode
          + unrealised MTM of currently open positions

        Note: this can go negative in pathological cases (e.g. a large
        adverse move on a short before a trailing stop triggers); callers
        that need a strictly positive value (e.g. log-return guards) clamp
        separately via `_safe_log_return`.
        """
        price = self.df.iloc[min(self.current_step, len(self.df) - 1)]["Open"]
        unrealised = sum(p.volume * (price - p.price) * p.signal for p in self.positions)
        return self.initial_balance + self._realised_pnl_acc + unrealised

    def _cost_basis(self) -> float:
        return sum(p.price * p.volume for p in self.positions)

    def _realised_pnl(self) -> float:
        """Accumulated realised PnL for all closed trades this episode."""
        return self._realised_pnl_acc