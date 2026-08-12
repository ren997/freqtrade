"""Daily simulated short strategy based on the first rejection of MA169/MA200."""

from __future__ import annotations

from datetime import datetime
from math import isfinite

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.enums import ExitCheckTuple, ExitType
from freqtrade.persistence import Order, Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute, timeframe_to_prev_date


class DailyMa169Ma200ShortStrategy(IStrategy):
    """Short the first rejection of the higher daily MA after a confirmed decline."""

    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = True

    MA_169_PERIOD = 169
    MA_200_PERIOD = 200
    ATR_PERIOD = 14
    BELOW_LOWER_MA_CANDLES = 5

    # No fixed ROI: the entry signal supplies an absolute take-profit level.
    minimal_roi = {}
    use_custom_roi = True

    # Wide fallback only. The entry candle's absolute stop is persisted after entry fill.
    stoploss = -0.99
    use_custom_stoploss = True

    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 220

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ma169": {"color": "blue"},
            "ma200": {"color": "purple"},
            "lower_ma": {"color": "lightblue"},
            "trigger_ma": {"color": "orange"},
        },
        "subplots": {"Risk": {"atr": {"color": "red"}}},
    }

    SIGNAL_PREFIX = "daily_ma_short"
    TARGET_KEY = "daily_ma_short_target"
    STOP_KEY = "daily_ma_short_stop"
    TRIGGER_KEY = "daily_ma_short_trigger"
    ATR_KEY = "daily_ma_short_atr"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ma169"] = ta.SMA(dataframe, timeperiod=self.MA_169_PERIOD)
        dataframe["ma200"] = ta.SMA(dataframe, timeperiod=self.MA_200_PERIOD)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.ATR_PERIOD)
        dataframe["lower_ma"] = dataframe[["ma169", "ma200"]].min(axis=1)
        dataframe["trigger_ma"] = dataframe[["ma169", "ma200"]].max(axis=1)
        dataframe["below_lower_ma"] = dataframe["high"] < dataframe["lower_ma"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Build each pair's entry state in chronological order without future data."""
        # Keep it timezone-compatible with the exchange-provided UTC candle timestamps.
        dataframe["cycle_start_date"] = pd.Series(pd.NaT, index=dataframe.index, dtype="object")
        dataframe["entry_target_price"] = np.nan
        dataframe["entry_stop_price"] = np.nan
        dataframe["entry_trigger_ma"] = np.nan
        dataframe["entry_atr"] = np.nan
        dataframe["enter_short"] = 0

        awaiting_first_touch = False
        below_count = 0
        cycle_start_position: int | None = None
        cycle_low: float | None = None

        for position, (index, candle) in enumerate(dataframe.iterrows()):
            is_below_lower_ma = bool(candle["below_lower_ma"])

            if not awaiting_first_touch:
                below_count = below_count + 1 if is_below_lower_ma else 0
                if below_count == self.BELOW_LOWER_MA_CANDLES:
                    cycle_start_position = position - self.BELOW_LOWER_MA_CANDLES + 1
                    cycle_low = float(
                        dataframe["low"].iloc[position - self.BELOW_LOWER_MA_CANDLES + 1 : position + 1]
                        .min()
                    )
                    awaiting_first_touch = True
                continue

            # The confirmation candle is never a touch: its high is below lower_ma,
            # while trigger_ma is always at least as high as lower_ma.
            trigger_ma = candle["trigger_ma"]
            if not isfinite(float(trigger_ma)):
                continue

            if candle["high"] > trigger_ma:
                target_is_valid = cycle_low is not None and cycle_low < candle["close"]
                rejection_is_valid = candle["close"] < trigger_ma
                atr_is_valid = isfinite(float(candle["atr"])) and candle["atr"] > 0

                if target_is_valid and rejection_is_valid and atr_is_valid:
                    stop_price = float(candle["high"] + candle["atr"])
                    dataframe.at[index, "cycle_start_date"] = dataframe["date"].iloc[
                        cycle_start_position
                    ]
                    dataframe.at[index, "entry_target_price"] = float(cycle_low)
                    dataframe.at[index, "entry_stop_price"] = stop_price
                    dataframe.at[index, "entry_trigger_ma"] = float(trigger_ma)
                    dataframe.at[index, "entry_atr"] = float(candle["atr"])
                    dataframe.at[index, "enter_short"] = 1
                    signal_timestamp = int(pd.Timestamp(candle["date"]).timestamp())
                    cycle_start_timestamp = int(
                        pd.Timestamp(dataframe["date"].iloc[cycle_start_position]).timestamp()
                    )
                    dataframe.at[index, "enter_tag"] = (
                        f"{self.SIGNAL_PREFIX}:{signal_timestamp}:{cycle_start_timestamp}"
                    )

                # The first high crossing trigger_ma has consumed this cycle.  Both an
                # accepted entry and a rejected touch require a fresh five-candle confirmation.
                awaiting_first_touch = False
                below_count = 0
                cycle_start_position = None
                cycle_low = None
            elif cycle_low is None:
                cycle_low = float(candle["low"])
            else:
                cycle_low = min(cycle_low, float(candle["low"]))

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Target and moving-average exits depend on values stored per trade.
        return dataframe

    @classmethod
    def _signal_timestamps(cls, entry_tag: str | None) -> tuple[int, int] | None:
        """Return the signal and five-candle-cycle start times encoded in an entry tag."""
        if not entry_tag:
            return None
        parts = entry_tag.split(":")
        if len(parts) != 3 or parts[0] != cls.SIGNAL_PREFIX:
            return None
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            return None

    def _entry_signal_candle(self, pair: str, trade: Trade) -> pd.Series | None:
        """Find this trade's completed signal candle from its encoded entry tag."""
        signal_timestamps = self._signal_timestamps(trade.enter_tag)
        if not self.dp or signal_timestamps is None:
            return None

        signal_timestamp, _ = signal_timestamps
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        candle_timestamps = pd.to_datetime(dataframe["date"], utc=True).map(
            lambda date: int(date.timestamp())
        )
        entry_candles = dataframe.loc[candle_timestamps.eq(signal_timestamp)]
        if entry_candles.empty:
            return None
        return entry_candles.iloc[-1]

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> None:
        """Persist the entry candle's risk levels before the exchange stop is created."""
        if order.ft_order_side != trade.entry_side or trade.nr_of_successful_entries != 1:
            return

        signal_candle = self._entry_signal_candle(pair, trade)
        if signal_candle is None:
            return

        values = {
            self.TARGET_KEY: signal_candle["entry_target_price"],
            self.STOP_KEY: signal_candle["entry_stop_price"],
            self.TRIGGER_KEY: signal_candle["entry_trigger_ma"],
            self.ATR_KEY: signal_candle["entry_atr"],
        }
        for key, value in values.items():
            if pd.notna(value) and isfinite(float(value)):
                trade.set_custom_data(key, float(value))

    @staticmethod
    def _custom_float(trade: Trade, key: str) -> float | None:
        value = trade.get_custom_data(key)
        if value is None:
            return None
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return None
        return value_float if isfinite(value_float) else None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """Keep the absolute entry stop at high(entry candle) + ATR(entry candle)."""
        stop_price = self._custom_float(trade, self.STOP_KEY)
        if stop_price is None:
            return self.stoploss
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def custom_roi(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float | None:
        """Model the fixed target as ROI so a daily backtest can detect an intraday touch."""
        target_price = self._custom_float(trade, self.TARGET_KEY)
        if target_price is None or not trade.is_short:
            return None
        return trade.calc_profit_ratio(target_price)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """Exit at the fixed target in real time or after a completed MA reclaim."""
        target_price = self._custom_float(trade, self.TARGET_KEY)
        if target_price is not None and current_rate <= target_price:
            return "daily_target"

        trigger_ma = self._custom_float(trade, self.TRIGGER_KEY)
        if trigger_ma is None or not self.dp:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        current_candle_start = timeframe_to_prev_date(self.timeframe, current_time)
        completed = dataframe.loc[pd.to_datetime(dataframe["date"], utc=True) < current_candle_start]
        if not completed.empty and float(completed.iloc[-1]["close"]) > trigger_ma:
            return "daily_trigger_ma_reclaimed"
        return None

    def should_exit(
        self,
        trade: Trade,
        rate: float,
        current_time: datetime,
        *,
        enter: bool,
        exit_: bool,
        low: float | None = None,
        high: float | None = None,
        force_stoploss: float = 0,
    ) -> list[ExitCheckTuple]:
        """Preserve the requested stop, target, then MA-reclaim exit precedence."""
        exits = super().should_exit(
            trade,
            rate,
            current_time,
            enter=enter,
            exit_=exit_,
            low=low,
            high=high,
            force_stoploss=force_stoploss,
        )

        def exit_priority(exit_check: ExitCheckTuple) -> int:
            if exit_check.exit_type in (
                ExitType.STOP_LOSS,
                ExitType.TRAILING_STOP_LOSS,
                ExitType.LIQUIDATION,
            ):
                return 0
            if exit_check.exit_type == ExitType.CUSTOM_EXIT and exit_check.exit_reason == "daily_target":
                return 1
            if exit_check.exit_type == ExitType.ROI:
                # Backtesting uses custom ROI to detect a target touch inside the daily candle.
                return 2
            if (
                exit_check.exit_type == ExitType.CUSTOM_EXIT
                and exit_check.exit_reason == "daily_trigger_ma_reclaimed"
            ):
                return 3
            return 4

        return sorted(exits, key=exit_priority)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """Reject a signal whose five-candle confirmation predates the last simulated exit."""
        signal_timestamps = self._signal_timestamps(entry_tag)
        if side != "short" or signal_timestamps is None:
            return False

        _, cycle_start_timestamp = signal_timestamps
        previous_trades = [
            previous_trade
            for previous_trade in Trade.get_trades_proxy(pair=pair, is_open=False)
            if previous_trade.strategy == self.get_strategy_name()
        ]
        if not previous_trades:
            return True

        last_exit_timestamp = max(
            previous_trade.close_date_utc.timestamp() for previous_trade in previous_trades
        )
        return cycle_start_timestamp > last_exit_timestamp
