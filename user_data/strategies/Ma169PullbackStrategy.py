# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
    AnnotationType,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class Ma169PullbackStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/stable/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # This setup is evaluated on four-hour candles.
    timeframe = "4h"

    # Can this strategy go short?
    can_short: bool = False

    # Exit only on the MA169 touch signal below; do not take a fixed ROI exit first.
    minimal_roi = {}

    # Wide fallback only.  custom_stoploss pins each trade to its signal candle's low.
    stoploss = -0.99
    use_custom_stoploss = True

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # MA169 plus five preceding candles are needed before a signal can be evaluated.
    startup_candle_count: int = 175

    MA_PERIOD = 169
    MA_ZONE_UPPER_PCT = 0.001
    CLEAR_CANDLES_REQUIRED = 5
    FALLBACK_STOPLOSS = -0.10

    order_types = {
        # A signal is known only once the confirmation candle has closed.  Submit a
        # market order on that evaluation cycle so the position (and entry_fill
        # webhook) is not delayed while waiting for a limit order to be revisited.
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    plot_config = {
        "main_plot": {
            "ma169": {"color": "blue"},
            "ma169_zone_upper": {"color": "lightblue"},
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # MA169 and the upper boundary of its 0.1% pullback zone.
        dataframe["ma169"] = ta.SMA(dataframe, timeperiod=self.MA_PERIOD)
        dataframe["ma169_zone_upper"] = dataframe["ma169"] * (1 + self.MA_ZONE_UPPER_PCT)

        # A candle touches the zone when its high-low range overlaps [MA169, MA169 * 1.001].
        dataframe["ma169_zone_touched"] = (
            (dataframe["low"] <= dataframe["ma169_zone_upper"])
            & (dataframe["high"] >= dataframe["ma169"])
        )

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ("live", "dry_run"):
                ob = self.dp.orderbook(metadata["pair"], 1)
                dataframe["best_bid"] = ob["bids"][0][0]
                dataframe["best_ask"] = ob["asks"][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # Each of the five candles before the signal candle must be completely above the
        # MA169-to-MA169*1.001 zone.  Therefore every one is above MA169 and none can touch
        # the zone.  shift(2) aligns this lookback with the confirmation candle.
        prior_candles_above_zone = (
            (dataframe["low"] > dataframe["ma169_zone_upper"])
            .shift(2)
            .rolling(self.CLEAR_CANDLES_REQUIRED, min_periods=self.CLEAR_CANDLES_REQUIRED)
            .sum()
            .eq(self.CLEAR_CANDLES_REQUIRED)
        )

        signal_candle_close = dataframe["close"].shift(1)
        dataframe["signal_candle_low"] = dataframe["low"].shift(1)

        dataframe.loc[
            (
                prior_candles_above_zone
                # Previous candle: touches the zone and closes at or above MA169.
                & dataframe["ma169_zone_touched"].shift(1)
                & (signal_candle_close >= dataframe["ma169"].shift(1))
                # Current candle: confirmation requires a bullish close.
                & (dataframe["close"] > dataframe["open"])
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ma169_pullback_confirmed")
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1
        """

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # Exit an open long at the close of the first later candle whose low-high range
        # touches MA169.  Market exit makes the live order execute immediately after the
        # completed candle is evaluated.
        dataframe.loc[
            (
                (dataframe["low"] <= dataframe["ma169"])
                & (dataframe["high"] >= dataframe["ma169"])
                # The confirmation candle opens the trade; it must not also block that entry.
                & dataframe["enter_long"].ne(1)
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "ma169_touched")
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1
        """
        return dataframe

    def _entry_signal_candle(self, pair: str, trade: Trade) -> Optional[pd.Series]:
        """Return the confirmation candle that opened this MA169 trade."""
        if not self.dp:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        entry_candles = dataframe.loc[
            (dataframe["date"] < trade.open_date_utc)
            & dataframe["enter_long"].eq(1)
            & dataframe["enter_tag"].eq(trade.enter_tag)
        ]
        if entry_candles.empty:
            return None
        return entry_candles.iloc[-1]

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> None:
        """Persist the signal candle low at the first entry fill for a stable stop price."""
        if trade.nr_of_successful_entries != 1 or order.ft_order_side != trade.entry_side:
            return

        entry_candle = self._entry_signal_candle(pair, trade)
        if entry_candle is not None:
            trade.set_custom_data("signal_candle_low", float(entry_candle["signal_candle_low"]))

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """Keep the stoploss at the low of the signal candle that triggered this trade."""
        signal_candle_low = trade.get_custom_data("signal_candle_low")
        if signal_candle_low is None or pd.isna(signal_candle_low):
            # Protect pre-existing trades created before this strategy version.
            return self.FALLBACK_STOPLOSS

        return stoploss_from_absolute(
            float(signal_candle_low),
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
