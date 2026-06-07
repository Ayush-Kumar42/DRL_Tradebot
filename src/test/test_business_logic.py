"""
pytest test suite for BusinessLogic closing strategies and position opening.

Conventions
-----------
- All tests bypass determine_action / execute_trade and call the strategy
  methods directly, so there is no dependency on indicator columns or the
  RL action vector.
- `make_bl()` returns a BusinessLogic instance with a minimal single-row
  DataFrame whose "Open" price is controlled per test.
- `pos()` is a shorthand Position factory.
- Each test group covers both long (signal=+1) and short (signal=-1) trades.
- PnL assertions use pytest.approx(abs=1e-6) to avoid floating-point fragility.
"""

import pytest
import pandas as pd
import numpy as np
from src.businesslogic import BusinessLogic, Position
import copy


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

SYM = "TEST"


def make_bl(open_price: float = 100.0, balance: float = 100_000.0) -> BusinessLogic:
    """Minimal BusinessLogic with a one-row DataFrame."""
    df = pd.DataFrame({"Open": [open_price]})
    return BusinessLogic(
        df=df,
        balance=balance,
        current_step=0,
        T_indicators=[],
        MR_indicators=[],
    )


def pos(order: int, price: float, volume: int, signal: int, symbol: str = SYM) -> Position:
    return Position(order=order, price=price, volume=volume, signal=signal, symbol=symbol)


def total_volume(positions: list[Position]) -> int:
    return sum(p.volume for p in positions)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: assert no positions in another symbol were touched
# ─────────────────────────────────────────────────────────────────────────────

def test_close_fifo_does_not_touch_other_symbols():
    bl = make_bl()

    positions = [
        pos(1, 100, 10, 1, "AAPL"),   # target symbol
        pos(2, 105, 20, 1, "AAPL"),

        pos(3, 200, 15, 1, "MSFT"),   # should remain unchanged
        pos(4, 250, 25, 1, "GOOG"),   # should remain unchanged
    ]

    before = copy.deepcopy(positions)

    after, pnl = bl.close_fifo(
        positions=positions,
        price=110,
        signal=1,
        units_to_close=15,
        symbol="AAPL",
    )

    # AAPL volume should decrease from 30 -> 15
    aapl_before = sum(p.volume for p in before if p.symbol == "AAPL")
    aapl_after = sum(p.volume for p in after if p.symbol == "AAPL")

    assert aapl_before == 30
    assert aapl_after == 15

    # All non-AAPL positions must be identical
    before_other = sorted(
        (p.order, p.symbol, p.volume, p.price, p.signal)
        for p in before
        if p.symbol != "AAPL"
    )

    after_other = sorted(
        (p.order, p.symbol, p.volume, p.price, p.signal)
        for p in after
        if p.symbol != "AAPL"
    )

    assert before_other == after_other


# ─────────────────────────────────────────────────────────────────────────────
#  1. OPEN POSITION  (via execute_trade with a mocked determine_action)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenPosition:
    """
    We patch determine_action to return a fixed (final_action, volume_fraction)
    so execute_trade's open branch is exercised deterministically.
    """

    def _open(self, bl: BusinessLogic, positions, signal: int, volume_fraction: float = 0.5):
        """Monkey-patch determine_action and call execute_trade."""
        bl.determine_action = lambda action, regime: (signal, volume_fraction)
        return bl.execute_trade(positions, action=None, regime="trending", symbol=SYM)

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_open_long_fresh(self):
        """No prior positions → open a new long."""
        bl = make_bl(open_price=100.0, balance=10_000.0)
        new_positions, pnl = self._open(bl, [], signal=1, volume_fraction=0.5)

        assert pnl == 0.0
        assert len(new_positions) == 1
        p = new_positions[0]
        assert p.signal == 1
        assert p.symbol == SYM
        assert p.price  == 100.0
        assert p.volume == 50          # floor(0.5 * (10_000 // 100)) = floor(0.5 * 100) = 50
        assert p.order  == 1

    def test_open_long_pyramid(self):
        """Existing long → same direction → pyramid (add another long)."""
        bl = make_bl(open_price=110.0, balance=10_000.0)
        existing = [pos(1, 100.0, 30, signal=1)]
        new_positions, pnl = self._open(bl, existing, signal=1, volume_fraction=0.5)

        assert pnl == 0.0
        assert len(new_positions) == 2
        new_p = new_positions[-1]
        assert new_p.signal == 1
        assert new_p.price  == 110.0
        assert new_p.order  == 1      # order_counter starts at 0, increments to 1

    def test_open_long_increments_order_counter(self):
        """Each open increments order_counter."""
        bl = make_bl(open_price=100.0, balance=10_000.0)
        positions, _ = self._open(bl, [], signal=1)
        positions, _ = self._open(bl, positions, signal=1)
        orders = [p.order for p in positions]
        assert orders == sorted(orders)          # monotonically increasing
        assert len(set(orders)) == len(orders)   # all unique

    def test_open_long_zero_volume_fraction(self):
        """volume_fraction=0 → no position created."""
        bl = make_bl(open_price=100.0, balance=10_000.0)
        new_positions, pnl = self._open(bl, [], signal=1, volume_fraction=0.0)
        assert new_positions == []
        assert pnl == 0.0

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_open_short_fresh(self):
        """No prior positions → open a new short."""
        bl = make_bl(open_price=100.0, balance=10_000.0)
        new_positions, pnl = self._open(bl, [], signal=-1, volume_fraction=0.5)

        assert pnl == 0.0
        assert len(new_positions) == 1
        p = new_positions[0]
        assert p.signal == -1
        assert p.symbol == SYM
        assert p.price  == 100.0
        assert p.volume == 50

    def test_open_short_pyramid(self):
        """Existing short → same direction → pyramid."""
        bl = make_bl(open_price=90.0, balance=10_000.0)
        existing = [pos(1, 100.0, 30, signal=-1)]
        new_positions, pnl = self._open(bl, existing, signal=-1, volume_fraction=0.5)

        assert len(new_positions) == 2
        assert new_positions[-1].signal == -1
        assert new_positions[-1].price  == 90.0

    def test_open_does_not_touch_other_symbols(self):
        """Opening on SYM must not alter positions in OTHER."""
        bl = make_bl(open_price=100.0, balance=10_000.0)
        other = pos(1, 50.0, 20, signal=1, symbol="OTHER")
        new_positions, _ = self._open(bl, [other], signal=1)

        other_after = [p for p in new_positions if p.symbol == "OTHER"]
        assert len(other_after) == 1
        assert other_after[0].volume == 20

    def test_hold_signal_opens_nothing(self):
        """final_action=0 → positions unchanged."""
        bl = make_bl(open_price=100.0, balance=10_000.0)
        bl.determine_action = lambda action, regime: (0, 0.5)
        new_positions, pnl = bl.execute_trade([], action=None, regime="trending", symbol=SYM)
        assert new_positions == []
        assert pnl == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  2. CLOSE: max_profit_first
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseMaxProfitFirst:

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_long_closes_most_profitable_first(self):
        """
        Three long positions at different entry prices.
        Current price=120. Most profit: entry=80 (+40), then 100 (+20), then 110 (+10).
        Close 1 unit total → should drain from entry=80 position first.
        """
        bl = make_bl(open_price=120.0)
        positions = [
            pos(1, 110.0, 10, signal=1),   # pnl/unit = +10
            pos(2, 100.0, 10, signal=1),   # pnl/unit = +20
            pos(3,  80.0, 10, signal=1),   # pnl/unit = +40  ← closes first
        ]
        remaining, pnl = bl.close_max_profit_first(positions, price=120.0, signal=1,
                                                    units_to_close=10, symbol=SYM)
        assert pnl == pytest.approx(10 * (120.0 - 80.0), abs=1e-6)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 3 in closed_orders   # entry=80 (order 3) was closed

    def test_long_partial_fill(self):
        """Close fewer units than a single position holds → partial fill."""
        bl = make_bl()
        positions = [pos(1, 80.0, 10, signal=1)]
        remaining, pnl = bl.close_max_profit_first(positions, price=120.0, signal=1,
                                                    units_to_close=4, symbol=SYM)
        assert len(remaining) == 1
        assert remaining[0].volume == 6
        assert pnl == pytest.approx(4 * 40.0, abs=1e-6)

    def test_long_zero_volume_positions_removed(self):
        """Fully drained position must be absent from remaining."""
        bl = make_bl()
        positions = [pos(1, 80.0, 5, signal=1)]
        remaining, _ = bl.close_max_profit_first(positions, price=120.0, signal=1,
                                                  units_to_close=5, symbol=SYM)
        assert remaining == []

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_short_closes_most_profitable_first(self):
        """
        Three short positions. Current price=80.
        Profit for short = (entry - price): entry=120 → +40, entry=100 → +20, entry=90 → +10.
        Close 10 units → should drain entry=120 first.
        """
        bl = make_bl(open_price=80.0)
        positions = [
            pos(1,  90.0, 10, signal=-1),  # pnl/unit = +10
            pos(2, 100.0, 10, signal=-1),  # pnl/unit = +20
            pos(3, 120.0, 10, signal=-1),  # pnl/unit = +40  ← closes first
        ]
        remaining, pnl = bl.close_max_profit_first(positions, price=80.0, signal=-1,
                                                    units_to_close=10, symbol=SYM)
        assert pnl == pytest.approx(10 * (80.0 - 120.0) * -1, abs=1e-6)  # = +400
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 3 in closed_orders

    def test_short_losing_position_closed_last(self):
        """A short losing position (price risen above entry) should be last."""
        bl = make_bl(open_price=110.0)
        positions = [
            pos(1, 120.0, 10, signal=-1),  # profit +10
            pos(2, 105.0, 10, signal=-1),  # loss   -5  ← should stay open longer
        ]
        remaining, _ = bl.close_max_profit_first(positions, price=110.0, signal=-1,
                                                  units_to_close=10, symbol=SYM)
        surviving_orders = {p.order for p in remaining}
        assert 2 in surviving_orders   # losing position stays


# ─────────────────────────────────────────────────────────────────────────────
#  3. CLOSE: least_loss_first
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseLeastLossFirst:

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_long_closes_smallest_loss_first(self):
        """
        Two losing longs. Loss of -5 (entry=105) and -20 (entry=120).
        Least loss first → entry=105 closes first.
        """
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1, 120.0, 10, signal=1),   # loss = -20
            pos(2, 105.0, 10, signal=1),   # loss =  -5  ← least loss, closes first
        ]
        remaining, pnl = bl.close_least_loss_first(positions, price=100.0, signal=1,
                                                    units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 2 in closed_orders
        assert pnl == pytest.approx(10 * (100.0 - 105.0), abs=1e-6)   # = -50

    def test_long_profitable_positions_deprioritised(self):
        """Profitable longs stay open when there are losses to cut first."""
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1,  80.0, 10, signal=1),   # profit +20
            pos(2, 110.0, 10, signal=1),   # loss   -10  ← closes first
        ]
        remaining, _ = bl.close_least_loss_first(positions, price=100.0, signal=1,
                                                  units_to_close=10, symbol=SYM)
        surviving_orders = {p.order for p in remaining}
        assert 1 in surviving_orders    # profitable position stays

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_short_closes_smallest_loss_first(self):
        """
        Two losing shorts (price rose above entry). 
        entry=95 → loss=-5, entry=90 → loss=-10. Least loss (entry=95) closes first.
        """
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1,  90.0, 10, signal=-1),   # loss = -10
            pos(2,  95.0, 10, signal=-1),   # loss =  -5  ← least loss, closes first
        ]
        remaining, pnl = bl.close_least_loss_first(positions, price=100.0, signal=-1,
                                                    units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 2 in closed_orders
        assert pnl == pytest.approx(10 * (100.0 - 95.0) * -1, abs=1e-6)  # = -50

    def test_short_profitable_positions_deprioritised(self):
        """Profitable shorts (price fell) stay open when losses exist."""
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1, 120.0, 10, signal=-1),   # profit +20
            pos(2,  95.0, 10, signal=-1),   # loss    -5  ← closes first
        ]
        remaining, _ = bl.close_least_loss_first(positions, price=100.0, signal=-1,
                                                  units_to_close=10, symbol=SYM)
        surviving_orders = {p.order for p in remaining}
        assert 1 in surviving_orders


# ─────────────────────────────────────────────────────────────────────────────
#  4. CLOSE: fifo
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseFifo:

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_long_closes_oldest_first(self):
        """order=1 is oldest → closes before order=2 and order=3."""
        bl = make_bl(open_price=120.0)
        positions = [
            pos(3, 100.0, 10, signal=1),
            pos(1,  90.0, 10, signal=1),   # oldest ← closes first
            pos(2, 110.0, 10, signal=1),
        ]
        remaining, pnl = bl.close_fifo(positions, price=120.0, signal=1,
                                        units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert closed_orders == {1}
        assert pnl == pytest.approx(10 * (120.0 - 90.0), abs=1e-6)

    def test_long_fifo_spans_multiple_positions(self):
        """Close 15 units across two positions of 10 each (FIFO order)."""
        bl = make_bl(open_price=120.0)
        positions = [
            pos(1, 90.0, 10, signal=1),   # oldest
            pos(2, 95.0, 10, signal=1),
        ]
        remaining, pnl = bl.close_fifo(positions, price=120.0, signal=1,
                                        units_to_close=15, symbol=SYM)
        assert len(remaining) == 1
        assert remaining[0].order == 2
        assert remaining[0].volume == 5
        expected = 10 * (120 - 90) + 5 * (120 - 95)
        assert pnl == pytest.approx(expected, abs=1e-6)

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_short_closes_oldest_first(self):
        """FIFO for shorts: oldest order regardless of PnL."""
        bl = make_bl(open_price=80.0)
        positions = [
            pos(3, 100.0, 10, signal=-1),
            pos(1, 110.0, 10, signal=-1),   # oldest ← closes first
            pos(2,  90.0, 10, signal=-1),
        ]
        remaining, pnl = bl.close_fifo(positions, price=80.0, signal=-1,
                                        units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert closed_orders == {1}
        assert pnl == pytest.approx(10 * (80.0 - 110.0) * -1, abs=1e-6)   # = +300


# ─────────────────────────────────────────────────────────────────────────────
#  5. CLOSE: age_weighted
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseAgeWeighted:

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_long_age_dominates_when_very_old(self):
        """
        Position with order=1 has norm_age=1.0 ≥ age_dominance_threshold=0.85
        → it must be closed first regardless of profit.
        """
        bl = make_bl(open_price=120.0)
        positions = [
            pos(1,  50.0, 10, signal=1),   # extremely old, huge profit → age dominates
            pos(100, 115.0, 10, signal=1), # newest, small profit
        ]
        remaining, _ = bl.close_age_weighted(
            positions, price=120.0, signal=1, units_to_close=10, symbol=SYM,
            age_dominance_threshold=0.85,
        )
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 1 in closed_orders

    def test_long_profit_dominates_when_ages_similar(self):
        """
        All positions have similar ages (no dominance triggered).
        Profit weight=1.0, age_weight=0.0 → most profitable closes first.
        """
        bl = make_bl(open_price=120.0)
        positions = [
            pos(1,  80.0, 10, signal=1),   # profit=40  ← highest, closes first
            pos(2, 100.0, 10, signal=1),   # profit=20
            pos(3, 110.0, 10, signal=1),   # profit=10
        ]
        remaining, _ = bl.close_age_weighted(
            positions, price=120.0, signal=1, units_to_close=10, symbol=SYM,
            profit_weight=1.0, age_weight=0.0, age_dominance_threshold=0.99,
        )
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 1 in closed_orders

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_short_age_dominates_when_very_old(self):
        """Oldest short position closed first regardless of PnL."""
        bl = make_bl(open_price=80.0)
        positions = [
            pos(1,  120.0, 10, signal=-1),   # oldest, most profitable
            pos(100, 85.0, 10, signal=-1),   # newest
        ]
        remaining, _ = bl.close_age_weighted(
            positions, price=80.0, signal=-1, units_to_close=10, symbol=SYM,
            age_dominance_threshold=0.85,
        )
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 1 in closed_orders

    def test_short_profit_weight_only(self):
        """profit_weight=1, age_weight=0 → most profitable short closed first."""
        bl = make_bl(open_price=80.0)
        positions = [
            pos(1,  70.0, 10, signal=-1),   # loss  (price rose above entry)
            pos(2, 120.0, 10, signal=-1),   # profit=40  ← closes first
            pos(3, 100.0, 10, signal=-1),   # profit=20
        ]
        remaining, _ = bl.close_age_weighted(
            positions, price=80.0, signal=-1, units_to_close=10, symbol=SYM,
            profit_weight=1.0, age_weight=0.0, age_dominance_threshold=0.99,
        )
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 2 in closed_orders


# ─────────────────────────────────────────────────────────────────────────────
#  6. CLOSE: most_risk_first
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseMostRiskFirst:

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_long_closes_biggest_loss_first(self):
        """
        Two losing longs. entry=120 → loss=-20, entry=110 → loss=-10.
        Most risk first → entry=120 (bigger loss) closes first.
        """
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1, 120.0, 10, signal=1),   # loss=-20  ← closes first
            pos(2, 110.0, 10, signal=1),   # loss=-10
        ]
        remaining, pnl = bl.close_most_risk_first(positions, price=100.0, signal=1,
                                                   units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 1 in closed_orders
        assert pnl == pytest.approx(10 * (100.0 - 120.0), abs=1e-6)   # = -200

    def test_long_profitable_positions_have_zero_risk(self):
        """Profitable longs are not at risk → kept open when losses exist."""
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1,  80.0, 10, signal=1),   # profit +20  ← risk=0, stays
            pos(2, 120.0, 10, signal=1),   # loss   -20  ← risk=20, closes
        ]
        remaining, _ = bl.close_most_risk_first(positions, price=100.0, signal=1,
                                                 units_to_close=10, symbol=SYM)
        surviving_orders = {p.order for p in remaining}
        assert 1 in surviving_orders

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_short_closes_biggest_loss_first(self):
        """
        Losing shorts: price rose above entry.
        entry=90 → loss=-10, entry=85 → loss=-15. Bigger loss (entry=85) closes first.
        """
        bl = make_bl(open_price=100.0)
        positions = [
            pos(1,  90.0, 10, signal=-1),   # loss=-10
            pos(2,  85.0, 10, signal=-1),   # loss=-15  ← closes first
        ]
        remaining, pnl = bl.close_most_risk_first(positions, price=100.0, signal=-1,
                                                   units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 2 in closed_orders
        assert pnl == pytest.approx(10 * (100.0 - 85.0) * -1, abs=1e-6)   # = -150

    def test_short_profitable_positions_have_zero_risk(self):
        """Profitable shorts (price below entry) are not at risk."""
        bl = make_bl(open_price=80.0)
        positions = [
            pos(1, 100.0, 10, signal=-1),   # profit +20 → stays
            pos(2,  85.0, 10, signal=-1),   # loss    -5 → closes
        ]
        remaining, _ = bl.close_most_risk_first(positions, price=80.0, signal=-1,
                                                 units_to_close=10, symbol=SYM)
        surviving_orders = {p.order for p in remaining}
        assert 1 in surviving_orders


# ─────────────────────────────────────────────────────────────────────────────
#  7. CLOSE: pnl_balanced (trailing stop proximity vs profit distance)
# ─────────────────────────────────────────────────────────────────────────────

class TestClosePnlBalanced:
    """
    trail_pct = 0.05 (default in make_bl).

    For a long at entry=100, current=95:
        stop = max(100, 95) * 0.95 = 100 * 0.95 = 95.0
        sl_dist  = |95 - 95| = 0.0   → stop already hit, maximum urgency

    For a long at entry=100, current=120:
        stop = max(100, 120) * 0.95 = 120 * 0.95 = 114.0
        sl_dist  = |120 - 114| = 6.0  → stop is 6 away
        pnl_dist = |120 - 100| = 20
    """

    # ── Long ──────────────────────────────────────────────────────────────────

    def test_long_loss_near_stop_closes_before_profit(self):
        """
        Position A: entry=100, current=95  → stop=95, sl_dist=0  (stop hit)
        Position B: entry=80,  current=95  → stop=90.25, sl_dist=4.75, pnl_dist=15

        A has higher urgency → closes first.
        """
        bl = make_bl(open_price=95.0)
        positions = [
            pos(1, 100.0, 10, signal=1),   # losing, stop hit → closes first
            pos(2,  80.0, 10, signal=1),   # profitable, stop far away
        ]
        remaining, pnl = bl.close_pnl_balanced(positions, price=95.0, signal=1,
                                                units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 1 in closed_orders
        assert pnl == pytest.approx(10 * (95.0 - 100.0), abs=1e-6)   # = -50

    def test_long_deep_profit_far_from_stop_stays_open(self):
        """
        Position A: deep profit (entry=50, current=120) → sl_dist large, pnl_dist large
        Position B: small loss  (entry=110, current=120 — wait, that's profitable too)

        Use a clearer setup: A is losing near stop, B is profitable far from stop.
        A closes first.
        """
        bl = make_bl(open_price=95.0)
        positions = [
            pos(1, 100.0, 10, signal=1),   # losing, stop=95 (hit), high urgency
            pos(2,  60.0, 10, signal=1),   # profit=35, stop=90.25, low urgency
        ]
        remaining, _ = bl.close_pnl_balanced(positions, price=95.0, signal=1,
                                              units_to_close=10, symbol=SYM)
        surviving = {p.order for p in remaining}
        assert 2 in surviving

    # ── Short ─────────────────────────────────────────────────────────────────

    def test_short_loss_near_stop_closes_before_profit(self):
        """
        Short position A: entry=100, current=106 → stop=105, sl_dist=1 (near stop, losing)
        Short position B: entry=120, current=106 → stop=106*1.05=111.3... wait, 
            trough=min(120,106)=106, stop=106*1.05=111.3, sl_dist=|106-111.3|=5.3

        A is more urgent (sl_dist smaller AND losing).
        """
        bl = make_bl(open_price=106.0)
        positions = [
            pos(1, 100.0, 10, signal=-1),   # losing (price rose), stop near → closes first
            pos(2, 120.0, 10, signal=-1),   # profitable, stop further away
        ]
        remaining, pnl = bl.close_pnl_balanced(positions, price=106.0, signal=-1,
                                                units_to_close=10, symbol=SYM)
        closed_orders = {p.order for p in positions} - {p.order for p in remaining}
        assert 1 in closed_orders
        assert pnl == pytest.approx(10 * (106.0 - 100.0) * -1, abs=1e-6)   # = -60

    def test_short_profitable_far_from_stop_stays_open(self):
        """Profitable short with stop well away should be kept open."""
        bl = make_bl(open_price=106.0)
        positions = [
            pos(1, 100.0, 10, signal=-1),   # losing, high urgency
            pos(2, 120.0, 10, signal=-1),   # profitable, low urgency → stays
        ]
        remaining, _ = bl.close_pnl_balanced(positions, price=106.0, signal=-1,
                                              units_to_close=10, symbol=SYM)
        surviving = {p.order for p in remaining}
        assert 2 in surviving


# ─────────────────────────────────────────────────────────────────────────────
#  8. Cross-symbol isolation (all strategies)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossSymbolIsolation:
    """
    Positions in OTHER must never be touched when closing SYM.
    Tested for each strategy.
    """

    STRATEGIES = [
        "max_profit", "least_loss", "fifo",
        "age_weighted", "most_risk", "pnl_balanced",
    ]

    def _close_via_strategy(self, bl, positions, strategy: str):
        fn_map = {
            "max_profit":   bl.close_max_profit_first,
            "least_loss":   bl.close_least_loss_first,
            "fifo":         bl.close_fifo,
            "age_weighted": bl.close_age_weighted,
            "most_risk":    bl.close_most_risk_first,
            "pnl_balanced": bl.close_pnl_balanced,
        }
        return fn_map[strategy](positions, price=120.0, signal=1,
                                units_to_close=10, symbol=SYM)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_other_symbol_untouched(self, strategy):
        bl = make_bl(open_price=120.0)
        positions = [
            pos(1, 100.0, 10, signal=1, symbol=SYM),
            pos(2,  90.0, 20, signal=1, symbol="OTHER"),
        ]
        remaining, _ = self._close_via_strategy(bl, positions, strategy)
        other = [p for p in remaining if p.symbol == "OTHER"]
        assert len(other) == 1
        assert other[0].volume == 20   # untouched


# ─────────────────────────────────────────────────────────────────────────────
#  9. _apply_close: zero-volume cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyClose:

    def test_fully_closed_position_removed(self):
        bl = make_bl(open_price=120.0)
        p = pos(1, 100.0, 10, signal=1)
        remaining, pnl = bl._apply_close([p], [p], price=120.0, units_to_close=10)
        assert remaining == []
        assert pnl == pytest.approx(10 * 20.0, abs=1e-6)

    def test_partial_close_preserves_position(self):
        bl = make_bl(open_price=120.0)
        p = pos(1, 100.0, 10, signal=1)
        remaining, pnl = bl._apply_close([p], [p], price=120.0, units_to_close=4)
        assert len(remaining) == 1
        assert remaining[0].volume == 6
        assert pnl == pytest.approx(4 * 20.0, abs=1e-6)

    def test_units_to_close_exceeds_available_is_capped(self):
        """Close more units than exist → drains everything, doesn't go negative."""
        bl = make_bl(open_price=120.0)
        p = pos(1, 100.0, 5, signal=1)
        remaining, pnl = bl._apply_close([p], [p], price=120.0, units_to_close=999)
        assert remaining == []
        assert pnl == pytest.approx(5 * 20.0, abs=1e-6)

    def test_short_pnl_correct_in_apply_close(self):
        """_apply_close uses p.signal for PnL direction."""
        bl = make_bl(open_price=80.0)
        p = pos(1, 100.0, 10, signal=-1)   # short, profit when price fell
        remaining, pnl = bl._apply_close([p], [p], price=80.0, units_to_close=10)
        assert remaining == []
        assert pnl == pytest.approx(10 * (80.0 - 100.0) * -1, abs=1e-6)   # = +200


# ─────────────────────────────────────────────────────────────────────────────
#  10. execute_trade open vs close routing
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteTradeRouting:
    """
    Patch determine_action so we control routing without touching indicator logic.
    """

    def _patch(self, bl, final_action: int, volume_fraction: float = 0.5):
        bl.determine_action = lambda action, regime: (final_action, volume_fraction)

    def test_close_long_via_execute_trade(self):
        """Existing long + sell signal → positions closed, PnL returned."""
        bl = make_bl(open_price=120.0, balance=100_000.0)
        positions = [pos(1, 100.0, 10, signal=1)]
        self._patch(bl, final_action=-1, volume_fraction=1.0)
        remaining, pnl = bl.execute_trade(positions, action=None, regime="trending",
                                           symbol=SYM, strategy="fifo")
        assert remaining == []
        assert pnl == pytest.approx(10 * 20.0, abs=1e-6)

    def test_close_short_via_execute_trade(self):
        """Existing short + buy signal → positions closed, PnL returned."""
        bl = make_bl(open_price=80.0, balance=100_000.0)
        positions = [pos(1, 100.0, 10, signal=-1)]
        self._patch(bl, final_action=1, volume_fraction=1.0)
        remaining, pnl = bl.execute_trade(positions, action=None, regime="trending",
                                           symbol=SYM, strategy="fifo")
        assert remaining == []
        assert pnl == pytest.approx(10 * (80.0 - 100.0) * -1, abs=1e-6)   # = +200

    def test_same_direction_triggers_open_not_close(self):
        """Existing long + buy signal → pyramid, not close."""
        bl = make_bl(open_price=110.0, balance=10_000.0)
        positions = [pos(1, 100.0, 10, signal=1)]
        self._patch(bl, final_action=1, volume_fraction=0.5)
        remaining, pnl = bl.execute_trade(positions, action=None, regime="trending",
                                           symbol=SYM)
        assert pnl == 0.0
        assert len(remaining) == 2
        assert remaining[-1].signal == 1
        assert remaining[-1].price  == 110.0

    def test_hold_signal_no_change(self):
        """final_action=0 → no change to positions."""
        bl = make_bl(open_price=100.0)
        positions = [pos(1, 100.0, 10, signal=1)]
        self._patch(bl, final_action=0)
        remaining, pnl = bl.execute_trade(positions, action=None, regime="trending",
                                           symbol=SYM)
        assert pnl == 0.0
        assert remaining == positions

    def test_other_symbol_positions_survive_close(self):
        """Closing SYM must not touch OTHER symbol positions."""
        bl = make_bl(open_price=120.0, balance=100_000.0)
        positions = [
            pos(1, 100.0, 10, signal=1, symbol=SYM),
            pos(2,  50.0, 20, signal=1, symbol="OTHER"),
        ]
        self._patch(bl, final_action=-1, volume_fraction=1.0)
        remaining, _ = bl.execute_trade(positions, action=None, regime="trending",
                                         symbol=SYM, strategy="fifo")
        other = [p for p in remaining if p.symbol == "OTHER"]
        assert len(other) == 1
        assert other[0].volume == 20