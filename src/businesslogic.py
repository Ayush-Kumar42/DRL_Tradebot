import pandas as pd
import numpy as np
from hurst import compute_Hc
from dataclasses import dataclass
from typing import Literal


@dataclass
class Position:
    order: int       # monotonically increasing order index (older = lower)
    price: float     # entry price
    volume: float      # units held
    signal: int      # 1 = long, -1 = short
    symbol: str      # ticker/instrument identifier e.g. "AAPL", "BTCUSDT"


CloseStrategy = Literal[
    "max_profit",
    "least_loss",
    "fifo",
    "age_weighted",
    "most_risk",
    "pnl_balanced",
]


class BusinessLogic:
    def __init__(self, df, balance, current_step, T_indicators, MR_indicators):
        self.df = df
        self.balance = balance
        self.current_step = current_step
        self.T_indicators = T_indicators
        self.MR_indicators = MR_indicators
        self.trading_threshold = 0.3
        self.trail_pct = 0.05
        self.order_counter = 0   # monotonically increasing; incremented on every new position

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def calculate_volume(self):
        price = self.df.iloc[self.current_step]["Open"]
        return self.balance / price  # float, no int()

    def calculate_hurst(self, time_series):
        H, c, data = compute_Hc(time_series, kind="price", simplified=True)
        return H

    def determine_action(self, action, regime):
        indicator_vector = action[:-1]
        volume_fraction = action[-1]

        active_indicators = self.T_indicators if regime == "trending" else self.MR_indicators

        active_signals = {}
        for idx, indicator in enumerate(active_indicators):
            weight = indicator_vector[idx]
            if weight > 0.5:
                try:
                    signal_col = next(
                        col for col in self.df.columns
                        if col.lower() == f"{indicator.lower()}_signal"
                    )
                except StopIteration:
                    raise KeyError(f"No signal column found for indicator '{indicator}'")
                signal_value = self.df.iloc[self.current_step][signal_col]
                active_signals[indicator] = weight * signal_value

        positive_total = sum(v for v in active_signals.values() if v > 0)
        negative_total = sum(v for v in active_signals.values() if v < 0)
        net_signal = positive_total + negative_total

        if net_signal > self.trading_threshold:
            final_action = 1
        elif net_signal < -self.trading_threshold:
            final_action = -1
        else:
            final_action = 0

        return final_action, volume_fraction

    # ------------------------------------------------------------------ #
    #  Shared infrastructure                                               #
    # ------------------------------------------------------------------ #

    def _prepare_close(
        self,
        positions: list[Position],
        symbol: str,
        signal: int,
    ) -> tuple[list[Position], int]:
        """
        Filter positions matching both `symbol` and `signal`.
        Returns (matching_positions, total_units_held).
        Symbol is checked first so strategies never touch positions in a
        different instrument even if they share the same direction.
        """
        matching = [p for p in positions if p.symbol == symbol and p.signal == signal]
        total_units = sum(p.volume for p in matching)
        return matching, total_units

    def _apply_close(
        self,
        positions: list[Position],
        sorted_positions: list[Position],
        price: float,
        units_to_close: int,
        symbol: str = "",
    ) -> tuple[list[Position], float]:
        """
        Walk through `sorted_positions` in priority order, draining
        `units_to_close` from each position until we've closed enough.

        PnL formula:
            long  (signal=+1): profit when price > entry  → (price - entry) * +1
            short (signal=-1): profit when price < entry  → (price - entry) * -1

        Returns:
            remaining_positions  – updated list; zero-volume positions removed
            total_profit         – realised PnL for this close operation
        """
        total_profit = 0.0
        pos_map = {id(p): p for p in positions}   # mutate originals in-place

        for pos in sorted_positions:
            if units_to_close <= 0:
                break

            p = pos_map[id(pos)]
            units_from_this = min(p.volume, units_to_close)

            total_profit += units_from_this * (price - p.price) * p.signal
            p.volume -= units_from_this
            units_to_close -= units_from_this

        remaining = [p for p in positions if p.volume > 0]
        return remaining, total_profit

    # ------------------------------------------------------------------ #
    #  Trailing stops                                                      #
    # ------------------------------------------------------------------ #

    def check_trailing_stops(
        self,
        positions: list[Position],
        price: float,
    ) -> tuple[float, list[Position]]:
        """
        Evaluate trailing stops for every open position and close those
        whose stop has been breached.

        Stop prices
        -----------
        Long  (signal=+1): stop = peak * (1 - trail_pct)
            where peak = entry price (conservative: we don't track intra-bar
            highs inside BusinessLogic, so `entry` serves as the reference).
            The environment should pass the current bar's Open as `price`.
        Short (signal=-1): stop = trough * (1 + trail_pct)
            where trough = entry price.

        The original TradingEnv tracked `highest_price` per position to
        produce a true trailing peak.  BusinessLogic uses the Position
        dataclass which has no such field, so the stop is anchored to
        `entry` rather than the running high/low.  If you need a true
        trailing high you can subclass Position to add `highest_price` /
        `lowest_price` and update this method accordingly.

        Returns
        -------
        (total_pnl, remaining_positions)
            total_pnl           – sum of all realised PnL from stopped-out positions
            remaining_positions – positions list with stopped-out entries removed
        """
        total_pnl = 0.0
        remaining: list[Position] = []

        for p in positions:
            if p.signal == 1:
                # Long trailing stop
                peak = p.price          # conservative anchor; see docstring
                stop = peak * (1 - self.trail_pct)
                triggered = price <= stop
            else:
                # Short trailing stop
                trough = p.price        # conservative anchor
                stop = trough * (1 + self.trail_pct)
                triggered = price >= stop

            if triggered:
                total_pnl += p.volume * (price - p.price) * p.signal
            else:
                remaining.append(p)

        return total_pnl, remaining

    # ------------------------------------------------------------------ #
    #  Force-close all positions (end of episode)                          #
    # ------------------------------------------------------------------ #

    def force_close_all(
        self,
        positions: list[Position],
    ) -> tuple[float, list[Position]]:
        """
        Immediately close every open position at the current bar's Close price.
        Used at episode termination so the agent is never left holding positions
        across episodes.

        Returns
        -------
        (total_pnl, [])
            total_pnl – sum of all realised PnL
            []        – empty position list (all positions are gone)
        """
        price = self.df.iloc[self.current_step]["Close"]
        total_pnl = sum(
            p.volume * (price - p.price) * p.signal for p in positions
        )
        return total_pnl, []

    # ------------------------------------------------------------------ #
    #  Strategy 1 – Max-Profit First                                       #
    # ------------------------------------------------------------------ #

    def close_max_profit_first(
        self,
        positions: list[Position],
        price: float,
        signal: int,
        units_to_close: int,
        symbol: str = "",
    ) -> tuple[list[Position], float]:
        """
        Close positions with the highest per-unit profit first.

        PnL per unit = (price - entry) * signal
            long  → positive when price > entry
            short → positive when price < entry  (signal=-1 flips the sign)

        Sorted descending so the most profitable position is closed first.
        Works symmetrically for longs and shorts because `* signal` already
        handles the direction.
        """
        matching, _ = self._prepare_close(positions, symbol, signal)
        sorted_pos = sorted(
            matching,
            key=lambda p: (price - p.price) * signal,
            reverse=True,
        )
        return self._apply_close(positions, sorted_pos, price, units_to_close)

    # ------------------------------------------------------------------ #
    #  Strategy 2 – Least-Loss First                                       #
    # ------------------------------------------------------------------ #

    def close_least_loss_first(
        self,
        positions: list[Position],
        price: float,
        signal: int,
        units_to_close: int,
        symbol: str = "",
    ) -> tuple[list[Position], float]:
        """
        Among losing positions, close those with the smallest absolute loss
        first (closest to break-even). Profitable positions are deprioritised
        so runners are left open as long as possible.

        PnL per unit = (price - entry) * signal
            < 0  → losing:  bucket 0, sorted by -pnl ascending  (least loss = smallest -pnl)
            ≥ 0  → winning: bucket 1, sorted by -pnl descending (keep best winners longest)

        Works for both longs and shorts because `* signal` normalises direction.
        """
        matching, _ = self._prepare_close(positions, symbol, signal)

        def least_loss_key(p):
            pnl = (price - p.price) * signal
            if pnl < 0:
                return (0, -pnl)
            return (1, -pnl)

        sorted_pos = sorted(matching, key=least_loss_key)
        return self._apply_close(positions, sorted_pos, price, units_to_close)

    # ------------------------------------------------------------------ #
    #  Strategy 3 – FIFO (Oldest First)                                    #
    # ------------------------------------------------------------------ #

    def close_fifo(
        self,
        positions: list[Position],
        price: float,
        signal: int,
        units_to_close: int,
        symbol: str = "",
    ) -> tuple[list[Position], float]:
        """
        Close the oldest positions first (lowest order index = entered earliest).
        Direction-agnostic: `order` is purely chronological, so this works
        identically for longs and shorts.
        """
        matching, _ = self._prepare_close(positions, symbol, signal)
        sorted_pos = sorted(matching, key=lambda p: p.order)
        return self._apply_close(positions, sorted_pos, price, units_to_close)

    # ------------------------------------------------------------------ #
    #  Strategy 4 – Age-Weighted (profit × age interaction)               #
    # ------------------------------------------------------------------ #

    def close_age_weighted(
        self,
        positions: list[Position],
        price: float,
        signal: int,
        units_to_close: int,
        symbol: str = "",
        age_weight: float = 0.3,
        profit_weight: float = 0.7,
        age_dominance_threshold: float = 0.85,
    ) -> tuple[list[Position], float]:
        """
        Blended score that prioritises profit but forces closure of positions
        that have grown extremely old relative to the rest of the portfolio.

        norm_age    : 0 = newest position, 1 = oldest position
        norm_profit : 0 = worst PnL in portfolio, 1 = best PnL

        Normal score  = profit_weight * norm_profit + age_weight * norm_age
        Dominance     : when norm_age ≥ age_dominance_threshold the position
                        jumps to bucket 0 and is sorted by age descending,
                        overriding profit entirely.
        """
        matching, _ = self._prepare_close(positions, symbol, signal)
        if not matching:
            return positions, 0.0

        orders = [p.order for p in matching]
        pnls   = [(price - p.price) * signal for p in matching]

        min_o, max_o = min(orders), max(orders)
        min_p, max_p = min(pnls),   max(pnls)

        o_range = max_o - min_o or 1
        p_range = max_p - min_p or 1

        def score(p):
            norm_age    = (p.order - min_o) / o_range
            norm_profit = ((price - p.price) * signal - min_p) / p_range

            if norm_age >= age_dominance_threshold:
                return (0, norm_age)

            combined = profit_weight * norm_profit + age_weight * norm_age
            return (1, combined)

        sorted_pos = sorted(matching, key=score, reverse=True)
        return self._apply_close(positions, sorted_pos, price, units_to_close)

    # ------------------------------------------------------------------ #
    #  Strategy 5 – Most-Risk First                                        #
    # ------------------------------------------------------------------ #

    def close_most_risk_first(
        self,
        positions: list[Position],
        price: float,
        signal: int,
        units_to_close: int,
        symbol: str = "",
        risk_free_rate: float = 0.0,
    ) -> tuple[list[Position], float]:
        """
        Close positions with the largest unrealised loss first to cap downside.
        Profitable positions are deprioritised (risk_score = 0 for winners).

        risk_score = max(0, -(pnl_per_unit - risk_free_rate))
            pnl < risk_free_rate → positive risk score (in the red)
            pnl ≥ risk_free_rate → score clamped to 0   (not at risk)

        Tiebreak: among equally safe positions, the least profitable closes
        first so the best winners keep running.
        """
        matching, _ = self._prepare_close(positions, symbol, signal)

        def risk_key(p):
            pnl = (price - p.price) * signal
            risk_score = max(0.0, -(pnl - risk_free_rate))
            return (risk_score, -pnl)

        sorted_pos = sorted(matching, key=risk_key, reverse=True)
        return self._apply_close(positions, sorted_pos, price, units_to_close)

    # ------------------------------------------------------------------ #
    #  Strategy 6 – PnL-Balanced (trailing-stop proximity vs profit)      #
    # ------------------------------------------------------------------ #

    def close_pnl_balanced(
        self,
        positions: list[Position],
        price: float,
        signal: int,
        units_to_close: int,
        symbol: str = "",
    ) -> tuple[list[Position], float]:
        """
        Compares each position's proximity to its trailing stop against its
        distance from break-even. If a loss is closer to being stopped out
        than a profit is away from zero, the loss is cut first.

        Trailing stop per position
        --------------------------
        Long  (signal=+1): stop = max(entry, price) * (1 - trail_pct)
        Short (signal=-1): stop = min(entry, price) * (1 + trail_pct)

        Urgency score
        -------------
        norm_sl_proximity = 1 - (sl_dist / max_sl_dist)   [1 = stop right here]
        norm_pnl_distance = pnl_dist / max_pnl_dist        [1 = deepest profit]
        urgency = norm_sl_proximity - norm_pnl_distance

        Sorted descending by urgency → highest urgency closed first.
        """
        matching, _ = self._prepare_close(positions, symbol, signal)
        if not matching:
            return positions, 0.0

        def stop_price(p: Position) -> float:
            if p.signal == 1:
                peak = max(p.price, price)
                return peak * (1 - self.trail_pct)
            else:
                trough = min(p.price, price)
                return trough * (1 + self.trail_pct)

        def metrics(p: Position) -> tuple[float, float]:
            stop     = stop_price(p)
            sl_dist  = abs(price - stop)
            pnl_dist = abs(price - p.price)
            return sl_dist, pnl_dist

        all_metrics  = [metrics(p) for p in matching]
        sl_dists     = [m[0] for m in all_metrics]
        pnl_dists    = [m[1] for m in all_metrics]

        max_sl_dist  = max(sl_dists)  or 1.0
        max_pnl_dist = max(pnl_dists) or 1.0

        def urgency(sl_dist: float, pnl_dist: float) -> tuple[float, float]:
            norm_sl_proximity = 1 - (sl_dist  / max_sl_dist)
            norm_pnl_distance = pnl_dist / max_pnl_dist
            score = norm_sl_proximity - norm_pnl_distance
            return (score, -sl_dist)

        scored = sorted(
            zip(matching, sl_dists, pnl_dists),
            key=lambda t: urgency(t[1], t[2]),
            reverse=True,
        )
        sorted_pos = [p for p, *_ in scored]
        return self._apply_close(positions, sorted_pos, price, units_to_close)

    # ------------------------------------------------------------------ #
    #  Main dispatcher                                                     #
    # ------------------------------------------------------------------ #

    def execute_trade(
        self,
        positions: list[Position],
        action,
        regime: str,
        symbol: str,
        strategy: CloseStrategy = "fifo",
        **strategy_kwargs,
    ) -> tuple[list[Position], float]:
        """
        One method handles both opening and closing for `symbol`.

        Decision tree
        -------------
        1. Run determine_action → (final_action, volume_fraction).
           final_action  0  → hold, do nothing.

        2. Check existing positions for this symbol.

        3. OPEN  – triggered when:
             a) No position exists for this symbol at all, OR
             b) All existing positions for this symbol share the SAME direction
                as final_action (pyramiding / adding to a winner).

        4. CLOSE – triggered when the last position for this symbol has the
           OPPOSITE direction to final_action (signal reversal).

        Returns
        -------
        (updated_positions, realised_pnl)
        """
        price = self.df.iloc[self.current_step]["Open"]
        final_action, volume_fraction = self.determine_action(action, regime)

        if final_action == 0:
            return positions, 0.0

        symbol_positions = [p for p in positions if p.symbol == symbol]

        if not symbol_positions:
            should_open  = True
            should_close = False
        else:
            last_signal     = symbol_positions[-1].signal
            same_direction  = all(p.signal == final_action for p in symbol_positions)
            opposing        = last_signal != final_action
            should_open     = same_direction
            should_close    = opposing

        # ── OPEN ──────────────────────────────────────────────────────
        if should_open:
            max_units  = self.calculate_volume()
            new_volume = volume_fraction * max_units

            if new_volume <= 0:
                return positions, 0.0

            self.order_counter += 1
            new_position = Position(
                order=self.order_counter,
                price=price,
                volume=new_volume,
                signal=final_action,
                symbol=symbol,
            )
            return positions + [new_position], 0.0

        # ── CLOSE ─────────────────────────────────────────────────────
        if should_close:
            close_signal = symbol_positions[-1].signal
            total_units  = sum(
                p.volume for p in symbol_positions if p.signal == close_signal
            )
            units_to_close = volume_fraction * total_units

            if units_to_close == 0:
                return positions, 0.0

            strategy_map: dict[str, callable] = {
                "max_profit":   self.close_max_profit_first,
                "least_loss":   self.close_least_loss_first,
                "fifo":         self.close_fifo,
                "age_weighted": self.close_age_weighted,
                "most_risk":    self.close_most_risk_first,
                "pnl_balanced": self.close_pnl_balanced,
            }

            close_fn = strategy_map.get(strategy)
            if close_fn is None:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Choose from: {list(strategy_map)}"
                )

            return close_fn(
                positions,
                price,
                close_signal,
                units_to_close,
                symbol=symbol,
                **strategy_kwargs,
            )

        return positions, 0.0