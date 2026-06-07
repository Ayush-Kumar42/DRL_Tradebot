import numpy as np
from scipy.stats import entropy
import ruptures as rpt
from hurst import compute_Hc
import gymnasium as gym
from gymnasium import spaces

from businesslogic import BusinessLogic, Position, CloseStrategy


class TradingEnv(gym.Env):
    """
    Gymnasium trading environment.

    All trade execution (open, close, trailing stop) is delegated to
    BusinessLogic.  The environment is responsible only for:
      - Building observations
      - Computing the Hurst exponent / regime
      - Advancing `current_step`
      - Formatting the `info` dict for the caller
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

    # ------------------------------------------------------------------ #
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def get_observation(self, regime: str) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        obs = list(row[self.continuous_features].values)
        obs.extend(row[["Open", "High", "Low", "Close", "Volume"]].values)
        if regime == "trending":
            obs.extend(row[self.T_indicators].values)
        else:
            obs.extend(row[self.MR_indicators].values)
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

        # Opened: present in after but not before (new object → new id)
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
                })

        if not opened and not closed:
            return None
        return {"opened": opened, "closed": closed}

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.positions = []

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
        }
        return obs, info

    def step(self, action):
        assert self.bl is not None, "Call reset() before step()."

        # Sync BusinessLogic's view of the world
        self.bl.current_step = self.current_step
        self.bl.balance      = self._portfolio_valuation()

        price  = self.df.iloc[self.current_step]["Open"]
        reward = 0.0
        all_opened: list[dict] = []
        all_closed: list[dict] = []

        # ── 1. Trailing stops ──────────────────────────────────────────
        before_trailing = list(self.positions)  # shallow copy for diff
        trailing_pnl, self.positions = self.bl.check_trailing_stops(
            self.positions, price
        )
        if trailing_pnl != 0.0:
            reward += trailing_pnl
            diff = self._diff_positions(before_trailing, self.positions, price, self.current_step)
            if diff:
                for c in diff["closed"]:
                    c["close_reason"] = "trailing_stop"
                all_closed.extend(diff["closed"])

        # ── 2. Hurst / regime ─────────────────────────────────────────
        self._update_hurst()
        regime = self._regime()

        # ── 3. Execute trade (skip if trailing stop already fired) ────
        if trailing_pnl == 0.0:
            # Mirror the original "skip if all active indicator values are zero"
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
                reward += trade_pnl

                diff = self._diff_positions(before_trade, self.positions, price, self.current_step)
                if diff:
                    for c in diff["closed"]:
                        c["close_reason"] = "manual_sell"
                    all_opened.extend(diff["opened"])
                    all_closed.extend(diff["closed"])

        # ── 4. Advance step ───────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated  = False

        # ── 5. Force-close at episode end ────────────────────────────
        if terminated and self.positions:
            self.bl.current_step = self.current_step - 1   # last valid index
            before_fc = list(self.positions)
            fc_pnl, self.positions = self.bl.force_close_all(self.positions)
            reward += fc_pnl
            diff = self._diff_positions(
                before_fc, self.positions,
                self.df.iloc[self.current_step - 1]["Close"],
                self.current_step,
            )
            if diff:
                for c in diff["closed"]:
                    c["close_reason"] = "force_close_end"
                all_closed.extend(diff["closed"])

        # ── 6. Build info ─────────────────────────────────────────────
        obs = self.get_observation(regime)
        position_changes = None
        if all_opened or all_closed:
            position_changes = {"opened": all_opened, "closed": all_closed}

        info = {
            "portfolio_value":  self._portfolio_valuation(),
            "position_changes": position_changes,
            "current_price":    price,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _portfolio_valuation(self) -> float:
        """
        Mark-to-market portfolio value:
        cash-equivalent balance + unrealised value of open positions.
        """
        price = self.df.iloc[min(self.current_step, len(self.df) - 1)]["Open"]
        position_value = sum(p.volume * price for p in self.positions)
        return self.initial_balance + self._realised_pnl() + position_value - self._cost_basis()

    def _cost_basis(self) -> float:
        return sum(p.price * p.volume for p in self.positions)

    def _realised_pnl(self) -> float:
        """Accumulated realised PnL is not tracked here; rely on reward signals."""
        # Portfolio valuation is approximated as initial_balance + open position MTM.
        # For a precise running balance, callers should accumulate `reward` externally.
        return 0.0

    def _active_symbol(self) -> str:
        """
        Returns the symbol for the current episode.
        Single-asset envs store it on `self.symbol`; fall back to "DEFAULT".
        Override this method (or set `self.symbol`) for multi-asset use.
        """
        return getattr(self, "symbol", "DEFAULT")