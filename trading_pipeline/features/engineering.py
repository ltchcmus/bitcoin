from typing import List, Tuple

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    feat["return_30s"] = feat["close"].pct_change(fill_method=None)
    feat["return_2m"] = feat["close"].pct_change(4, fill_method=None)
    feat["return_5m"] = feat["close"].pct_change(10, fill_method=None)
    feat["return_15m"] = feat["close"].pct_change(30, fill_method=None)
    feat["log_return"] = np.log(feat["close"]).diff()

    feat["ema_9"] = feat["close"].ewm(span=9, adjust=False).mean()
    feat["ema_21"] = feat["close"].ewm(span=21, adjust=False).mean()
    feat["ema_ratio"] = feat["ema_9"] / (feat["ema_21"] + 1e-12) - 1

    ema_12 = feat["close"].ewm(span=12, adjust=False).mean()
    ema_26 = feat["close"].ewm(span=26, adjust=False).mean()
    feat["macd"] = ema_12 - ema_26
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    delta = feat["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = feat["close"].shift(1)
    tr1 = feat["high"] - feat["low"]
    tr2 = (feat["high"] - prev_close).abs()
    tr3 = (feat["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()

    feat["range_pct"] = (feat["high"] - feat["low"]) / (feat["close"] + 1e-12)
    roll_high_20 = feat["high"].rolling(20).max()
    roll_low_20 = feat["low"].rolling(20).min()
    feat["price_position_20"] = (feat["close"] - roll_low_20) / (
        roll_high_20 - roll_low_20 + 1e-12
    )

    feat["momentum_10"] = feat["close"] / (feat["close"].shift(10) + 1e-12) - 1
    feat["momentum_30"] = feat["close"] / (feat["close"].shift(30) + 1e-12) - 1

    feat["volatility_10"] = feat["log_return"].rolling(10).std()
    feat["volatility_30"] = feat["log_return"].rolling(30).std()
    feat["volatility_60"] = feat["log_return"].rolling(60).std()

    vol_mean_30 = feat["volume"].rolling(30).mean()
    vol_std_30 = feat["volume"].rolling(30).std()
    feat["volume_zscore_30"] = (feat["volume"] - vol_mean_30) / (vol_std_30 + 1e-12)
    feat["volume_ema_20"] = feat["volume"].ewm(span=20, adjust=False).mean()
    feat["volume_ema_ratio"] = feat["volume"] / (feat["volume_ema_20"] + 1e-12) - 1

    feat["trade_intensity"] = feat["number_of_trades"] / (feat["volume"] + 1e-12)
    trades_mean_30 = feat["number_of_trades"].rolling(30).mean()
    trades_std_30 = feat["number_of_trades"].rolling(30).std()
    feat["trade_count_zscore_30"] = (feat["number_of_trades"] - trades_mean_30) / (
        trades_std_30 + 1e-12
    )
    feat["order_imbalance_proxy"] = (
        (2.0 * feat["taker_buy_base"] / (feat["volume"] + 1e-12)) - 1.0
    ).clip(-1, 1)
    feat["buy_pressure_proxy"] = (
        (2.0 * feat["taker_buy_quote"] / (feat["quote_asset_volume"] + 1e-12)) - 1.0
    ).clip(-1, 1)

    minute = feat["open_time"].dt.hour * 60 + feat["open_time"].dt.minute
    feat["minute_sin"] = np.sin(2 * np.pi * minute / 1440)
    feat["minute_cos"] = np.cos(2 * np.pi * minute / 1440)

    if "funding_rate" in feat.columns:
        feat["funding_rate"] = pd.to_numeric(feat["funding_rate"], errors="coerce")

    if "open_interest" in feat.columns:
        feat["open_interest"] = pd.to_numeric(feat["open_interest"], errors="coerce")
        feat["oi_change_5m"] = feat["open_interest"].pct_change(10, fill_method=None)

    if "open_interest_value" in feat.columns:
        feat["open_interest_value"] = pd.to_numeric(
            feat["open_interest_value"], errors="coerce"
        )
        feat["oi_value_change_5m"] = feat["open_interest_value"].pct_change(
            10, fill_method=None
        )

    if "taker_buy_sell_ratio" in feat.columns:
        feat["taker_buy_sell_ratio"] = pd.to_numeric(
            feat["taker_buy_sell_ratio"], errors="coerce"
        )

    if "taker_buy_vol" in feat.columns and "taker_sell_vol" in feat.columns:
        buy_v = pd.to_numeric(feat["taker_buy_vol"], errors="coerce")
        sell_v = pd.to_numeric(feat["taker_sell_vol"], errors="coerce")
        feat["futures_aggr_pressure"] = np.log((buy_v + 1e-9) / (sell_v + 1e-9))

    if "spread_bps" in feat.columns:
        feat["spread_bps"] = pd.to_numeric(feat["spread_bps"], errors="coerce")
    if "depth_imbalance_10" in feat.columns:
        feat["depth_imbalance_10"] = pd.to_numeric(
            feat["depth_imbalance_10"], errors="coerce"
        )
    if "depth_imbalance_20" in feat.columns:
        feat["depth_imbalance_20"] = pd.to_numeric(
            feat["depth_imbalance_20"], errors="coerce"
        )

    return feat


def add_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["future_return"] = out["close"].shift(-horizon) / out["close"] - 1
    out["target"] = 0
    out.loc[out["future_return"] > threshold, "target"] = 1
    out.loc[out["future_return"] < -threshold, "target"] = -1
    return out


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "return_30s",
        "return_2m",
        "return_5m",
        "return_15m",
        "log_return",
        "ema_ratio",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "atr_14",
        "range_pct",
        "price_position_20",
        "momentum_10",
        "momentum_30",
        "volatility_10",
        "volatility_30",
        "volatility_60",
        "volume_zscore_30",
        "volume_ema_ratio",
        "trade_intensity",
        "trade_count_zscore_30",
        "order_imbalance_proxy",
        "buy_pressure_proxy",
        "funding_rate",
        "open_interest",
        "oi_change_5m",
        "open_interest_value",
        "oi_value_change_5m",
        "taker_buy_sell_ratio",
        "futures_aggr_pressure",
        "spread_bps",
        "depth_imbalance_10",
        "depth_imbalance_20",
        "minute_sin",
        "minute_cos",
    ]
    return [col for col in candidates if col in df.columns]


def encode_target(y: pd.Series) -> pd.Series:
    return y + 1


def decode_target(y_encoded: np.ndarray) -> np.ndarray:
    return y_encoded - 1


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()
    return train_df, valid_df, test_df
