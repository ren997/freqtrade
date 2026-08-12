from datetime import UTC
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from freqtrade.enums import ExitCheckTuple, ExitType


STRATEGY_PATH = (
    Path(__file__).parents[2] / "user_data" / "strategies" / "DailyMa169Ma200ShortStrategy.py"
)
SPEC = spec_from_file_location("daily_ma169_ma200_short", STRATEGY_PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DailyMa169Ma200ShortStrategy = MODULE.DailyMa169Ma200ShortStrategy


def _candles(count: int = 214) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=count, freq="D", tz=UTC)
    dataframe = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(count, 100.0),
            "high": np.full(count, 102.0),
            "low": np.full(count, 98.0),
            "close": np.full(count, 100.0),
            "volume": np.full(count, 1.0),
        }
    )
    return dataframe


def _with_fixed_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["ma169"] = 100.0
    dataframe["ma200"] = 105.0
    dataframe["lower_ma"] = 100.0
    dataframe["trigger_ma"] = 105.0
    dataframe["atr"] = 2.0
    dataframe["below_lower_ma"] = dataframe["high"] < dataframe["lower_ma"]
    return dataframe


def _confirmed_cycle(dataframe: pd.DataFrame, start: int) -> None:
    dataframe.loc[start : start + 4, ["high", "low", "close"]] = [90.0, 80.0, 85.0]


def test_enters_on_first_valid_touch_and_records_fixed_levels():
    strategy = DailyMa169Ma200ShortStrategy({})
    dataframe = _candles()
    _confirmed_cycle(dataframe, 200)
    dataframe.loc[205, ["high", "low", "close"]] = [110.0, 94.0, 95.0]

    result = strategy.populate_entry_trend(_with_fixed_indicators(dataframe), {"pair": "BTC/USDT:USDT"})

    assert result.loc[205, "enter_short"] == 1
    assert result.loc[205, "entry_target_price"] == 80.0
    assert result.loc[205, "entry_trigger_ma"] == 105.0
    assert result.loc[205, "entry_atr"] == 2.0
    assert result.loc[205, "entry_stop_price"] == 112.0
    assert result.loc[205, "cycle_start_date"] == result.loc[200, "date"]


def test_invalid_first_touch_requires_a_fresh_five_candle_cycle():
    strategy = DailyMa169Ma200ShortStrategy({})
    dataframe = _candles()
    _confirmed_cycle(dataframe, 195)
    # It touches the trigger MA but closes above it, invalidating the entire cycle.
    dataframe.loc[200, ["high", "low", "close"]] = [110.0, 100.0, 106.0]
    # A later valid-looking touch may not reuse the invalidated cycle.
    dataframe.loc[201, ["high", "low", "close"]] = [110.0, 94.0, 95.0]

    result = strategy.populate_entry_trend(_with_fixed_indicators(dataframe), {"pair": "BTC/USDT:USDT"})

    assert result["enter_short"].sum() == 0


def test_target_must_remain_below_the_entry_close():
    strategy = DailyMa169Ma200ShortStrategy({})
    dataframe = _candles()
    _confirmed_cycle(dataframe, 200)
    dataframe.loc[205, ["high", "low", "close"]] = [110.0, 94.0, 80.0]

    result = strategy.populate_entry_trend(_with_fixed_indicators(dataframe), {"pair": "BTC/USDT:USDT"})

    assert result["enter_short"].sum() == 0


def test_custom_stoploss_uses_the_persisted_absolute_stop():
    strategy = DailyMa169Ma200ShortStrategy({})
    trade = SimpleNamespace(
        is_short=True,
        leverage=1.0,
        get_custom_data=lambda key: {strategy.STOP_KEY: 112.0}.get(key),
    )

    stoploss = strategy.custom_stoploss(
        pair="BTC/USDT:USDT",
        trade=trade,
        current_time=pd.Timestamp("2024-08-01", tz=UTC).to_pydatetime(),
        current_rate=100.0,
        current_profit=0.0,
        after_fill=False,
    )

    assert stoploss == pytest.approx(0.12)


def test_order_fill_persists_the_entry_candle_levels():
    strategy = DailyMa169Ma200ShortStrategy({})
    signal_date = pd.Timestamp("2024-08-01", tz=UTC)
    signal_timestamp = int(signal_date.timestamp())
    signal_dataframe = pd.DataFrame(
        {
            "date": [signal_date],
            "entry_target_price": [80.0],
            "entry_stop_price": [112.0],
            "entry_trigger_ma": [105.0],
            "entry_atr": [2.0],
        }
    )
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (signal_dataframe, signal_date.to_pydatetime())
    )
    stored = {}
    trade = SimpleNamespace(
        enter_tag=f"{strategy.SIGNAL_PREFIX}:{signal_timestamp}:{signal_timestamp - 432000}",
        nr_of_successful_entries=1,
        entry_side="sell",
        open_date_utc=signal_date.to_pydatetime(),
        set_custom_data=lambda key, value: stored.__setitem__(key, value),
    )
    order = SimpleNamespace(ft_order_side="sell")

    strategy.order_filled("BTC/USDT:USDT", trade, order, signal_date.to_pydatetime())

    assert stored == {
        strategy.TARGET_KEY: 80.0,
        strategy.STOP_KEY: 112.0,
        strategy.TRIGGER_KEY: 105.0,
        strategy.ATR_KEY: 2.0,
    }


def test_entry_confirmation_requires_a_cycle_after_the_previous_exit(monkeypatch):
    strategy = DailyMa169Ma200ShortStrategy({})
    last_exit = pd.Timestamp("2024-08-01", tz=UTC).to_pydatetime()
    previous_trade = SimpleNamespace(
        strategy=strategy.get_strategy_name(), close_date_utc=last_exit
    )
    monkeypatch.setattr(MODULE.Trade, "get_trades_proxy", lambda **kwargs: [previous_trade])

    assert not strategy.confirm_trade_entry(
        pair="BTC/USDT:USDT",
        order_type="market",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=last_exit,
        entry_tag=f"{strategy.SIGNAL_PREFIX}:1722643200:1722124800",
        side="short",
    )
    assert strategy.confirm_trade_entry(
        pair="BTC/USDT:USDT",
        order_type="market",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=last_exit,
        entry_tag=f"{strategy.SIGNAL_PREFIX}:1723161600:1722643201",
        side="short",
    )


def test_exit_ordering_is_stop_then_target_then_ma_reclaim(monkeypatch):
    strategy = DailyMa169Ma200ShortStrategy({})
    result = [
        ExitCheckTuple(ExitType.CUSTOM_EXIT, "daily_trigger_ma_reclaimed"),
        ExitCheckTuple(ExitType.ROI),
        ExitCheckTuple(ExitType.CUSTOM_EXIT, "daily_target"),
        ExitCheckTuple(ExitType.STOP_LOSS),
    ]
    monkeypatch.setattr(MODULE.IStrategy, "should_exit", lambda *args, **kwargs: result)

    exits = strategy.should_exit(
        trade=SimpleNamespace(),
        rate=100.0,
        current_time=pd.Timestamp("2024-08-01", tz=UTC).to_pydatetime(),
        enter=False,
        exit_=False,
    )

    assert [exit_check.exit_reason for exit_check in exits] == [
        "stop_loss",
        "daily_target",
        "roi",
        "daily_trigger_ma_reclaimed",
    ]
