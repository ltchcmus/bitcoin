import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from trading_pipeline.config import (
    BINANCE_AGG_TRADES_REST,
    BINANCE_DEPTH_REST,
    BINANCE_FUTURES_REST,
    BINANCE_REST,
    DEFAULT_LOOKBACK_DAYS,
    SECONDS_PER_BAR,
)


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


def to_millis(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
) -> List[List]:
    session = requests.Session()
    all_rows: List[List] = []
    cursor = start_ms
    interval_ms = interval_to_millis(interval)

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(BINANCE_REST, params=params, timeout=15)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        rows = response.json() if response is not None else []
        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = int(rows[-1][0])
        cursor = last_open_time + interval_ms

        if len(rows) < limit:
            break

        time.sleep(0.05)

    return all_rows


def fetch_agg_trades(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> List[Dict]:
    all_rows: List[Dict] = []
    for rows, _evt in iter_agg_trades_batches(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
        max_retries=max_retries,
        progress_callback=progress_callback,
    ):
        all_rows.extend(rows)
    return all_rows


def iter_agg_trades_batches(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Iterator[Tuple[List[Dict], Dict]]:
    session = requests.Session()
    cursor = start_ms
    batch_count = 0
    fetched_total = 0
    batch_limit = min(limit, 1000)

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "startTime": cursor,
            "endTime": end_ms,
            "limit": batch_limit,
        }
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(
                    BINANCE_AGG_TRADES_REST, params=params, timeout=15
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        rows = response.json() if response is not None else []
        if not rows:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "agg_trades",
                        "batch": batch_count,
                        "fetched": fetched_total,
                        "cursor_ms": cursor,
                        "end_ms": end_ms,
                        "done": True,
                    }
                )
            break

        batch_count += 1
        fetched_total += len(rows)
        last_ts = int(rows[-1]["T"])
        cursor = last_ts + 1
        done = len(rows) < batch_limit or cursor >= end_ms

        event = {
            "stage": "agg_trades",
            "batch": batch_count,
            "fetched": fetched_total,
            "cursor_ms": cursor,
            "end_ms": end_ms,
            "done": done,
        }

        if progress_callback is not None:
            progress_callback(event)

        yield rows, event

        if done:
            break

        time.sleep(0.05)


def fetch_funding_rate_history(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
) -> List[Dict]:
    session = requests.Session()
    all_rows: List[Dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "startTime": cursor,
            "endTime": end_ms,
            "limit": min(limit, 1000),
        }
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(
                    f"{BINANCE_FUTURES_REST}/fapi/v1/fundingRate",
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        rows = response.json() if response is not None else []
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = int(rows[-1].get("fundingTime", cursor))
        if last_ts <= cursor:
            cursor += 1
        else:
            cursor = last_ts + 1

        if len(rows) < limit:
            break

        time.sleep(0.05)

    return all_rows


def fetch_open_interest_hist(
    symbol: str,
    start_ms: int,
    end_ms: int,
    period: str = "5m",
    limit: int = 500,
    max_retries: int = 5,
) -> List[Dict]:
    session = requests.Session()
    all_rows: List[Dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": min(limit, 500),
        }
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(
                    f"{BINANCE_FUTURES_REST}/futures/data/openInterestHist",
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        rows = response.json() if response is not None else []
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = int(rows[-1].get("timestamp", cursor))
        if last_ts <= cursor:
            cursor += 1
        else:
            cursor = last_ts + 1

        if len(rows) < limit:
            break

        time.sleep(0.05)

    return all_rows


def fetch_taker_long_short_ratio(
    symbol: str,
    start_ms: int,
    end_ms: int,
    period: str = "5m",
    limit: int = 500,
    max_retries: int = 5,
) -> List[Dict]:
    session = requests.Session()
    all_rows: List[Dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": min(limit, 500),
        }
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(
                    f"{BINANCE_FUTURES_REST}/futures/data/takerlongshortRatio",
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        rows = response.json() if response is not None else []
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = int(rows[-1].get("timestamp", cursor))
        if last_ts <= cursor:
            cursor += 1
        else:
            cursor = last_ts + 1

        if len(rows) < limit:
            break

        time.sleep(0.05)

    return all_rows


def build_futures_feature_frame(
    symbol: str,
    start_ms: int,
    end_ms: int,
    period: str = "5m",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    funding_rows = fetch_funding_rate_history(
        symbol=symbol, start_ms=start_ms, end_ms=end_ms
    )
    oi_rows = fetch_open_interest_hist(
        symbol=symbol, start_ms=start_ms, end_ms=end_ms, period=period
    )
    ratio_rows = fetch_taker_long_short_ratio(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        period=period,
    )

    frames: List[pd.DataFrame] = []
    if funding_rows:
        fdf = pd.DataFrame(funding_rows)
        fdf = fdf[["fundingTime", "fundingRate"]].copy()
        fdf["open_time"] = pd.to_datetime(fdf["fundingTime"], unit="ms", utc=True)
        fdf["funding_rate"] = pd.to_numeric(fdf["fundingRate"], errors="coerce")
        frames.append(fdf[["open_time", "funding_rate"]])

    if oi_rows:
        odf = pd.DataFrame(oi_rows)
        odf = odf[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].copy()
        odf["open_time"] = pd.to_datetime(odf["timestamp"], unit="ms", utc=True)
        odf["open_interest"] = pd.to_numeric(odf["sumOpenInterest"], errors="coerce")
        odf["open_interest_value"] = pd.to_numeric(
            odf["sumOpenInterestValue"], errors="coerce"
        )
        frames.append(odf[["open_time", "open_interest", "open_interest_value"]])

    if ratio_rows:
        rdf = pd.DataFrame(ratio_rows)
        rdf = rdf[["timestamp", "buySellRatio", "buyVol", "sellVol"]].copy()
        rdf["open_time"] = pd.to_datetime(rdf["timestamp"], unit="ms", utc=True)
        rdf["taker_buy_sell_ratio"] = pd.to_numeric(
            rdf["buySellRatio"], errors="coerce"
        )
        rdf["taker_buy_vol"] = pd.to_numeric(rdf["buyVol"], errors="coerce")
        rdf["taker_sell_vol"] = pd.to_numeric(rdf["sellVol"], errors="coerce")
        frames.append(
            rdf[
                ["open_time", "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]
            ]
        )

    if not frames:
        return pd.DataFrame(columns=["open_time"]), {
            "funding_rows": 0,
            "open_interest_rows": 0,
            "taker_ratio_rows": 0,
        }

    merged = frames[0].sort_values("open_time").drop_duplicates("open_time")
    for frame in frames[1:]:
        merged = pd.merge(
            merged,
            frame.sort_values("open_time").drop_duplicates("open_time"),
            on="open_time",
            how="outer",
        )

    merged = merged.sort_values("open_time").reset_index(drop=True)
    stats = {
        "funding_rows": int(len(funding_rows)),
        "open_interest_rows": int(len(oi_rows)),
        "taker_ratio_rows": int(len(ratio_rows)),
    }
    return merged, stats


def enrich_bars_with_futures_features(
    bars_df: pd.DataFrame,
    symbol: str,
    period: str = "5m",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if bars_df.empty:
        return bars_df, {
            "funding_rows": 0,
            "open_interest_rows": 0,
            "taker_ratio_rows": 0,
        }

    start_ms = int(
        pd.to_datetime(bars_df["open_time"].min(), utc=True).timestamp() * 1000
    )
    end_ms = int(
        pd.to_datetime(bars_df["close_time"].max(), utc=True).timestamp() * 1000
    )

    futures_df, stats = build_futures_feature_frame(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        period=period,
    )
    if futures_df.empty:
        return bars_df, stats

    left = bars_df.sort_values("open_time").copy()
    right = futures_df.sort_values("open_time").copy()
    enriched = pd.merge_asof(left, right, on="open_time", direction="backward")
    return enriched, stats


def fetch_depth_snapshot_features(
    symbol: str,
    limit: int = 100,
    max_retries: int = 5,
) -> Dict[str, float]:
    session = requests.Session()
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                BINANCE_DEPTH_REST,
                params={"symbol": symbol.upper(), "limit": min(max(limit, 20), 5000)},
                timeout=15,
            )
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(1.2 * attempt)

    data = response.json() if response is not None else {"bids": [], "asks": []}
    bids = np.array(
        [[float(px), float(qty)] for px, qty in data.get("bids", [])], dtype=float
    )
    asks = np.array(
        [[float(px), float(qty)] for px, qty in data.get("asks", [])], dtype=float
    )

    if len(bids) == 0 or len(asks) == 0:
        return {
            "best_bid": np.nan,
            "best_ask": np.nan,
            "mid_price": np.nan,
            "spread_bps": np.nan,
            "depth_imbalance_10": np.nan,
            "depth_imbalance_20": np.nan,
        }

    best_bid = bids[0, 0]
    best_ask = asks[0, 0]
    mid_price = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / (mid_price + 1e-12)) * 10_000

    bid_notional_10 = float(np.sum(bids[:10, 0] * bids[:10, 1]))
    ask_notional_10 = float(np.sum(asks[:10, 0] * asks[:10, 1]))
    depth_imbalance_10 = (bid_notional_10 - ask_notional_10) / (
        bid_notional_10 + ask_notional_10 + 1e-12
    )

    bid_notional_20 = float(np.sum(bids[:20, 0] * bids[:20, 1]))
    ask_notional_20 = float(np.sum(asks[:20, 0] * asks[:20, 1]))
    depth_imbalance_20 = (bid_notional_20 - ask_notional_20) / (
        bid_notional_20 + ask_notional_20 + 1e-12
    )

    return {
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "mid_price": float(mid_price),
        "spread_bps": float(spread_bps),
        "depth_imbalance_10": float(depth_imbalance_10),
        "depth_imbalance_20": float(depth_imbalance_20),
    }


def agg_trades_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["trade_id", "price", "quantity", "timestamp", "is_buyer_maker"]
        )

    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "a": "trade_id",
            "p": "price",
            "q": "quantity",
            "T": "timestamp",
            "m": "is_buyer_maker",
        }
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    df = df[["trade_id", "price", "quantity", "timestamp", "is_buyer_maker"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def aggregate_trades_to_30s(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
            ]
        )

    df = trades_df.copy()
    bucket = f"{SECONDS_PER_BAR}s"
    df["open_time"] = df["timestamp"].dt.floor(bucket)
    df["quote"] = df["price"] * df["quantity"]
    # In Binance aggTrades, m=True means buyer is maker -> taker side is sell.
    df["taker_buy_qty"] = np.where(df["is_buyer_maker"], 0.0, df["quantity"])
    df["taker_buy_quote"] = np.where(df["is_buyer_maker"], 0.0, df["quote"])

    grouped = df.groupby("open_time", as_index=False).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        quote_asset_volume=("quote", "sum"),
        number_of_trades=("trade_id", "count"),
        taker_buy_base=("taker_buy_qty", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
    )
    grouped["close_time"] = grouped["open_time"] + pd.to_timedelta(
        SECONDS_PER_BAR, unit="s"
    )
    grouped["close_time"] = grouped["close_time"] - pd.to_timedelta(1, unit="ms")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    return grouped[cols].sort_values("open_time").reset_index(drop=True)


def klines_to_dataframe(rows: List[List]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = (
        df.drop(columns=["ignore"])
        .sort_values("open_time")
        .drop_duplicates("open_time")
    )
    df = df.reset_index(drop=True)
    return df


def read_market_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, parse_dates=["open_time", "close_time"])
    raise ValueError("Supported formats: .parquet, .csv")


def write_market_data(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
        return
    raise ValueError("Supported formats: .parquet, .csv")


def get_existing_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return read_market_data(path)


def update_30s_dataset(
    symbol: str,
    data_path: str,
    default_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    include_futures_features: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    existing = get_existing_data(data_path)
    now = datetime.now(timezone.utc)

    if existing.empty:
        start_time = now - timedelta(days=default_lookback_days)
    else:
        existing = (
            existing.sort_values("open_time")
            .drop_duplicates("open_time")
            .reset_index(drop=True)
        )
        last_open = pd.to_datetime(existing["open_time"].iloc[-1], utc=True)
        start_time = last_open + timedelta(seconds=1)

    if verbose:
        print(
            "[download] Start aggTrades backfill "
            f"symbol={symbol.upper()} from={start_time.isoformat()} to={now.isoformat()}"
        )

    merged = existing.copy()
    last_print = time.time()
    last_checkpoint = time.time()
    checkpoint_every_batches = 5000
    staged_bars: List[pd.DataFrame] = []
    staged_bar_rows = 0
    pending_trades = pd.DataFrame(
        columns=["trade_id", "price", "quantity", "timestamp", "is_buyer_maker"]
    )
    raw_trade_count = 0
    new_bars_count = 0

    def _merge_bars_in_memory(new_bars_df: pd.DataFrame) -> None:
        nonlocal merged
        if new_bars_df.empty:
            return
        if merged.empty:
            merged = new_bars_df.copy()
            return
        merged = pd.concat([merged, new_bars_df], ignore_index=True)
        merged = (
            merged.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

    def _flush_stage(checkpoint_write: bool) -> None:
        nonlocal staged_bars, staged_bar_rows, new_bars_count, last_checkpoint
        if not staged_bars:
            return

        stage_df = pd.concat(staged_bars, ignore_index=True)
        new_bars_count += int(len(stage_df))
        _merge_bars_in_memory(stage_df)

        staged_bars = []
        staged_bar_rows = 0

        if checkpoint_write:
            write_market_data(merged, data_path)
            last_checkpoint = time.time()
            if verbose:
                print(
                    "[download] Checkpoint saved "
                    f"path={data_path} bars={len(merged):,}"
                )

    def _progress(evt: Dict) -> None:
        nonlocal last_print
        if not verbose:
            return
        now_t = time.time()
        if not evt.get("done", False) and (now_t - last_print) < 2.0:
            return
        last_print = now_t

        start_ms_local = to_millis(start_time)
        end_ms_local = evt.get("end_ms", start_ms_local)
        cursor_ms_local = evt.get("cursor_ms", start_ms_local)
        denom = max(end_ms_local - start_ms_local, 1)
        progress = max(
            0.0, min(100.0, 100.0 * (cursor_ms_local - start_ms_local) / denom)
        )
        print(
            "[download] aggTrades "
            f"batch={evt.get('batch', 0)} fetched={evt.get('fetched', 0):,} "
            f"progress~{progress:.2f}%"
        )

    for rows, evt in iter_agg_trades_batches(
        symbol=symbol,
        start_ms=to_millis(start_time),
        end_ms=to_millis(now),
        progress_callback=_progress,
    ):
        raw_trade_count += len(rows)
        batch_df = agg_trades_to_dataframe(rows)
        if not pending_trades.empty:
            batch_df = pd.concat([pending_trades, batch_df], ignore_index=True)

        if batch_df.empty:
            continue

        bucket = f"{SECONDS_PER_BAR}s"
        batch_df["bar_open_time"] = batch_df["timestamp"].dt.floor(bucket)
        last_bucket = batch_df["bar_open_time"].iloc[-1]

        closed_df = batch_df[batch_df["bar_open_time"] < last_bucket].drop(
            columns=["bar_open_time"]
        )
        pending_trades = (
            batch_df[batch_df["bar_open_time"] == last_bucket]
            .drop(columns=["bar_open_time"])
            .reset_index(drop=True)
        )

        if not closed_df.empty:
            bars_chunk = aggregate_trades_to_30s(closed_df)
            if not bars_chunk.empty:
                staged_bars.append(bars_chunk)
                staged_bar_rows += int(len(bars_chunk))

        need_checkpoint = (
            evt.get("batch", 0) % checkpoint_every_batches == 0
            or staged_bar_rows >= 25_000
            or (time.time() - last_checkpoint) >= 180
        )
        if need_checkpoint:
            _flush_stage(checkpoint_write=True)

    if not pending_trades.empty:
        last_chunk = aggregate_trades_to_30s(pending_trades)
        if not last_chunk.empty:
            staged_bars.append(last_chunk)
            staged_bar_rows += int(len(last_chunk))

    _flush_stage(checkpoint_write=False)

    if verbose:
        print(f"[download] aggTrades done total={raw_trade_count:,}")

    if verbose:
        print(
            "[download] Aggregation done "
            f"new_bars_30s={new_bars_count:,} existing_bars={len(existing):,}"
        )

    if merged.empty and "open_time" not in merged.columns:
        merged = pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base",
                "taker_buy_quote",
            ]
        )

    futures_stats = {
        "funding_rows": 0,
        "open_interest_rows": 0,
        "taker_ratio_rows": 0,
    }
    if include_futures_features and not merged.empty:
        try:
            if verbose:
                print("[download] Enriching futures features...")
            merged, futures_stats = enrich_bars_with_futures_features(
                merged, symbol=symbol, period="5m"
            )
            if verbose:
                print(
                    "[download] Futures enrichment done "
                    f"funding={futures_stats['funding_rows']:,} "
                    f"open_interest={futures_stats['open_interest_rows']:,} "
                    f"taker_ratio={futures_stats['taker_ratio_rows']:,}"
                )
        except Exception:
            futures_stats = {
                "funding_rows": 0,
                "open_interest_rows": 0,
                "taker_ratio_rows": 0,
            }
            if verbose:
                print("[download] Futures enrichment skipped due to API/read error")

    write_market_data(merged, data_path)
    if verbose:
        print(
            f"[download] Saved dataset path={data_path} total_bars_30s={len(merged):,}"
        )
    stats = {
        "raw_trades": int(raw_trade_count),
        "new_bars": int(new_bars_count),
        "total_bars": int(len(merged)),
        "funding_rows": int(futures_stats["funding_rows"]),
        "open_interest_rows": int(futures_stats["open_interest_rows"]),
        "taker_ratio_rows": int(futures_stats["taker_ratio_rows"]),
    }
    return merged, stats
