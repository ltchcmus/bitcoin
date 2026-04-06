import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import websocket

from trading_pipeline.data.binance_client import (
    agg_trades_to_dataframe,
    aggregate_trades_to_30s,
    enrich_bars_with_futures_features,
    fetch_depth_snapshot_features,
    fetch_agg_trades,
    to_millis,
)
from trading_pipeline.features.engineering import add_features


@dataclass
class LiveConfig:
    model_path: str
    symbol: str
    bootstrap_bars: int = 1500


class LivePredictor:
    def __init__(self, cfg: LiveConfig):
        artifact = joblib.load(cfg.model_path)
        runtime_cfg = artifact.get("runtime", {})
        self.use_gpu = bool(runtime_cfg.get("use_gpu", False))
        if "models" in artifact:
            self.model = artifact["models"]["xgb"]
            self.transformer_payload = artifact["models"].get("transformer")
            ensemble_cfg = artifact.get("ensemble", {})
            self.ensemble_weight_xgb = float(ensemble_cfg.get("weight_xgb", 0.55))
            self.require_model_agreement = bool(
                ensemble_cfg.get("require_model_agreement", True)
            )
        else:
            self.model = artifact["model"]
            self.transformer_payload = None
            self.ensemble_weight_xgb = 1.0
            self.require_model_agreement = False
        self.feature_cols = artifact["feature_cols"]
        self.confidence = float(artifact["confidence_threshold"])
        self.symbol = cfg.symbol.upper()
        self.bootstrap_bars = cfg.bootstrap_bars
        self.buffer = pd.DataFrame()
        self.pending_trades: List[Dict] = []

    def bootstrap(self) -> None:
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=max(120, self.bootstrap_bars // 2))
        rows = fetch_agg_trades(
            symbol=self.symbol,
            start_ms=to_millis(start),
            end_ms=to_millis(now),
        )
        trades_df = agg_trades_to_dataframe(rows)
        self.buffer = aggregate_trades_to_30s(trades_df)
        self.buffer = self.buffer.tail(max(self.bootstrap_bars, 500)).reset_index(
            drop=True
        )
        print(f"Bootstrap bars: {len(self.buffer)}")

    def _append_trade(self, trade_payload: Dict) -> None:
        self.pending_trades.append(
            {
                "a": int(trade_payload["t"]),
                "p": str(trade_payload["p"]),
                "q": str(trade_payload["q"]),
                "T": int(trade_payload["T"]),
                "m": bool(trade_payload["m"]),
            }
        )

    def _flush_closed_bars(self, event_ts: pd.Timestamp) -> None:
        if not self.pending_trades:
            return

        cutoff = event_ts.floor("30s")
        trades_df = agg_trades_to_dataframe(self.pending_trades)
        closed_df = trades_df[trades_df["timestamp"] < cutoff].copy()
        open_df = trades_df[trades_df["timestamp"] >= cutoff].copy()

        self.pending_trades = [
            {
                "a": int(row["trade_id"]),
                "p": str(row["price"]),
                "q": str(row["quantity"]),
                "T": int(row["timestamp"].timestamp() * 1000),
                "m": bool(row["is_buyer_maker"]),
            }
            for _, row in open_df.iterrows()
        ]

        if closed_df.empty:
            return

        new_bars = aggregate_trades_to_30s(closed_df)
        if new_bars.empty:
            return

        self.buffer = pd.concat([self.buffer, new_bars], ignore_index=True)
        self.buffer = self.buffer.drop_duplicates("open_time").sort_values("open_time")
        self.buffer = self.buffer.tail(max(self.bootstrap_bars, 500)).reset_index(
            drop=True
        )

        if len(self.buffer) > 80:
            self._predict_latest()

    def _predict_latest(self) -> None:
        source_df = self.buffer.copy()
        try:
            source_df, _ = enrich_bars_with_futures_features(
                source_df, symbol=self.symbol, period="5m"
            )
        except Exception:
            pass

        try:
            depth_snapshot = fetch_depth_snapshot_features(
                symbol=self.symbol, limit=100
            )
            for key, value in depth_snapshot.items():
                source_df[key] = value
        except Exception:
            pass

        feature_df = add_features(source_df)
        latest = feature_df.iloc[-1]
        x_live = pd.Series({col: latest.get(col, np.nan) for col in self.feature_cols})
        if x_live.isna().any():
            return

        probs_xgb = self.model.predict_proba(x_live.to_numpy().reshape(1, -1))[0]
        probs_tf = None
        if self.transformer_payload is not None:
            from trading_pipeline.model.transformer_pipeline import (
                transformer_predict_last_window,
            )

            probs_tf = transformer_predict_last_window(
                feature_frame=feature_df,
                feature_cols=self.feature_cols,
                payload=self.transformer_payload,
                use_gpu=self.use_gpu,
            )

        if probs_tf is not None:
            probs = (
                self.ensemble_weight_xgb * probs_xgb
                + (1.0 - self.ensemble_weight_xgb) * probs_tf
            )
            is_agree = int(np.argmax(probs_xgb)) == int(np.argmax(probs_tf))
        else:
            probs = probs_xgb
            is_agree = True

        direction = int(np.array([-1, 0, 1])[int(np.argmax(probs))])
        confidence = float(np.max(probs))
        is_confirmed = is_agree or (not self.require_model_agreement)
        action = direction if confidence >= self.confidence and is_confirmed else 0

        stamp = str(latest["close_time"])
        print(
            f"[{stamp}] close={latest['close']:.2f} prob_down={probs[0]:.3f} "
            f"prob_flat={probs[1]:.3f} prob_up={probs[2]:.3f} conf={confidence:.3f} "
            f"confirmed={is_confirmed} action={action}"
        )

    def on_message(self, _ws, message: str) -> None:
        payload = json.loads(message)
        if payload.get("e") != "trade":
            return

        event_ts = pd.to_datetime(int(payload["T"]), unit="ms", utc=True)
        self._append_trade(payload)
        self._flush_closed_bars(event_ts)

    def on_error(self, _ws, err) -> None:
        print(f"WebSocket error: {err}")

    def on_close(self, _ws, close_status_code, close_msg) -> None:
        print(f"WebSocket closed: {close_status_code} {close_msg}")

    def on_open(self, _ws) -> None:
        print("WebSocket connected")

    def run(self) -> None:
        self.bootstrap()
        stream = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"

        while True:
            ws = websocket.WebSocketApp(
                stream,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
            print("Reconnecting in 3s...")
            time.sleep(3)
