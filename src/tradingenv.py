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
      - Building observations
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
    """

    def __init__(
        self,
        df,
        T_indicators: list[str],
        MR_indicators: list[str],
        continuous_features: list[str],
        initial_balance: float,
        close_strategy: CloseStrategy = "fifo",
        trail_pct: float = 0.05,
    ):
        super().__init__()

        # --- Data & feature config ---
        self.df = df.reset_index(drop=True)
        self.T_indicators = T_indicators
        self.MR_indicators = MR_indicators
        self.continuous_features = continuous_features
        self.initial_balance = initial_balance
        self.close_strategy = close_strategy
        self.trail_pct = trail_pct

        # --- Action space: one weight per indicator + volume fraction ---
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(T_indicators) + 1,),
            dtype=np.float32,
        )

        # --- Observation space ---
        # continuous_features  +  OHLCV (5)  +  regime indicator signals
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
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def get_observation(self, regime: str) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        obs = list(row[self.continuous_features].values)
        obs.extend(row[["Open", "High", "Low", "Close", "Volume"]].values)
        indicators = self.T_indicators if regime == "trending" else self.MR_indicators
        signal_cols = [f"{ind}_signal" for ind in indicators]  # add _signal suffix
        obs.extend(row[signal_cols].values)
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
        before_ids = {id(p): p for p in before}
        after_ids  = {id(p): p for p in after}

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

        self.current_step = 0
        self.positions = []
        self._realised_pnl_acc = 0.0   # reset accumulator each episode
        self._holding_log = {}
        self._prev_mark_price = {}

        # Bootstrap BusinessLogic; balance is the portfolio valuation tracked here.
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
            "portfolio_value":  self.initial_balance,
            "position_changes": None,
            "current_price":    self.df.iloc[0]["Open"],
            "global_log_return": 0.0,
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

        # 3. Execute trade (manual/strategy close -> eligible for closing bonus)
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
                    symbol=self._active_symbol(),
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

        # 6. Force-close at episode end (no closing bonus - see class docstring)
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

    def _active_symbol(self) -> str:
        """
        Returns the symbol for the current episode.
        Single-asset envs store it on `self.symbol`; fall back to "DEFAULT".
        Override this method (or set `self.symbol`) for multi-asset use.
        """
        return getattr(self, "symbol", "DEFAULT")