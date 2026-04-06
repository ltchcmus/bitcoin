from datetime import datetime, timedelta, timezone
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from trading_pipeline.backtest.metrics import run_backtest
from trading_pipeline.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_ENSEMBLE_WEIGHT_XGB,
    DEFAULT_ENABLE_DEPTH_FEATURES,
    DEFAULT_ENABLE_FUTURES_FEATURES,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_OPTUNA_STORAGE,
    DEFAULT_OPTUNA_STUDY_NAME,
    DEFAULT_REQUIRE_MODEL_AGREEMENT,
    DEFAULT_SYMBOL,
    DEFAULT_TRANSFORMER_EPOCHS,
    DEFAULT_TRANSFORMER_SEQ_LEN,
    DEFAULT_TUNE_TRIALS,
    DEFAULT_THRESHOLD_30S,
    DEFAULT_USE_GPU,
    DEFAULT_USE_TRANSFORMER,
    MIN_TRAIN_BARS_30S,
)
from trading_pipeline.data.binance_client import (
    agg_trades_to_dataframe,
    aggregate_trades_to_30s,
    enrich_bars_with_futures_features,
    fetch_depth_snapshot_features,
    fetch_agg_trades,
    read_market_data,
    update_30s_dataset,
)
from trading_pipeline.features.engineering import (
    add_features,
    add_target,
    decode_target,
    select_feature_columns,
    split_by_time,
)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_from_dataframe(
    raw_df,
    symbol: str = DEFAULT_SYMBOL,
    threshold: float = DEFAULT_THRESHOLD_30S,
    horizon: int = 1,
    confidence: float = DEFAULT_CONFIDENCE,
    fee: float = 0.0004,
    tune_trials: int = DEFAULT_TUNE_TRIALS,
    optuna_storage: str = DEFAULT_OPTUNA_STORAGE,
    optuna_study_name: str = DEFAULT_OPTUNA_STUDY_NAME,
    use_transformer: bool = DEFAULT_USE_TRANSFORMER,
    use_gpu: bool = DEFAULT_USE_GPU,
    transformer_seq_len: int = DEFAULT_TRANSFORMER_SEQ_LEN,
    transformer_epochs: int = DEFAULT_TRANSFORMER_EPOCHS,
    ensemble_weight_xgb: float = DEFAULT_ENSEMBLE_WEIGHT_XGB,
    require_model_agreement: bool = DEFAULT_REQUIRE_MODEL_AGREEMENT,
    verbose: bool = False,
):
    from trading_pipeline.model.xgb_pipeline import train_xgboost
    from trading_pipeline.model.transformer_pipeline import train_transformer

    if verbose:
        print(f"[train] Start feature engineering rows={len(raw_df):,}")

    feat_df = add_features(raw_df)
    labeled_df = add_target(feat_df, horizon=horizon, threshold=threshold)

    feature_cols = select_feature_columns(labeled_df)
    model_df = labeled_df.dropna(
        subset=feature_cols + ["target", "future_return"]
    ).reset_index(drop=True)

    if verbose:
        print(
            "[train] Feature+target ready "
            f"rows_after_dropna={len(model_df):,} features={len(feature_cols)}"
        )

    if len(model_df) < MIN_TRAIN_BARS_30S:
        raise ValueError(
            f"Data qua it de train on dinh: {len(model_df)} bars. Can it nhat ~{MIN_TRAIN_BARS_30S:,} bars 30s."
        )

    train_df, valid_df, test_df = split_by_time(model_df)
    if verbose:
        print(
            "[train] Split done "
            f"train={len(train_df):,} valid={len(valid_df):,} test={len(test_df):,}"
        )

    model, model_params = train_xgboost(
        train_df,
        valid_df,
        feature_cols,
        tune_trials=tune_trials,
        optuna_storage=optuna_storage,
        optuna_study_name=optuna_study_name,
        use_gpu=use_gpu,
        verbose=verbose,
    )

    transformer_result = {
        "payload": None,
        "test_probs": np.empty((0, 3), dtype=np.float32),
        "test_offset": 0,
        "params": {},
    }
    if use_transformer:
        transformer_result = train_transformer(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
            seq_len=transformer_seq_len,
            epochs=transformer_epochs,
            use_gpu=use_gpu,
            verbose=verbose,
        )

    x_test = test_df[feature_cols]
    y_test_true_full = test_df["target"].to_numpy()
    probs_xgb_full = model.predict_proba(x_test)

    tf_probs = transformer_result["test_probs"]
    tf_offset = int(transformer_result["test_offset"])
    has_transformer = transformer_result["payload"] is not None and len(tf_probs) > 0

    if has_transformer and tf_offset < len(probs_xgb_full):
        probs_xgb = probs_xgb_full[tf_offset:]
        y_test_true = y_test_true_full[tf_offset:]
        future_returns = test_df["future_return"].iloc[tf_offset:]

        probs_ensemble = (
            float(ensemble_weight_xgb) * probs_xgb
            + (1.0 - float(ensemble_weight_xgb)) * tf_probs
        )
        agreement_mask = np.argmax(probs_xgb, axis=1) == np.argmax(tf_probs, axis=1)
    else:
        probs_xgb = probs_xgb_full
        y_test_true = y_test_true_full
        future_returns = test_df["future_return"]
        probs_ensemble = probs_xgb
        agreement_mask = np.ones(len(probs_ensemble), dtype=bool)

    y_pred_enc = np.argmax(probs_ensemble, axis=1)
    y_pred = decode_target(y_pred_enc)

    confirmation_mask = agreement_mask if require_model_agreement else None
    metrics = run_backtest(
        future_returns=future_returns,
        probs=probs_ensemble,
        confidence_threshold=confidence,
        fee_per_trade=fee,
        confirmation_mask=confirmation_mask,
    )

    report = classification_report(y_test_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test_true, y_pred, labels=[-1, 0, 1])
    cls_metrics = _classification_metrics(y_test_true, y_pred)

    artifact = {
        "model_type": "ensemble" if has_transformer else "xgboost",
        "models": {
            "xgb": model,
            "transformer": transformer_result["payload"],
        },
        "feature_cols": feature_cols,
        "confidence_threshold": confidence,
        "threshold": threshold,
        "horizon": horizon,
        "symbol": symbol.upper(),
        "ensemble": {
            "weight_xgb": float(ensemble_weight_xgb),
            "require_model_agreement": bool(require_model_agreement),
        },
        "runtime": {
            "use_gpu": bool(use_gpu),
        },
        "eval_summary": {
            "classification_metrics": cls_metrics,
            "backtest_metrics": metrics,
            "has_transformer": bool(has_transformer),
            "agreement_rate": (
                float(agreement_mask.mean()) if len(agreement_mask) else 0.0
            ),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    details = {
        "classification_report": report,
        "confusion_matrix": cm,
        "classification_metrics": cls_metrics,
        "backtest_metrics": metrics,
        "bars_used": int(len(model_df)),
        "train_bars": int(len(train_df)),
        "valid_bars": int(len(valid_df)),
        "test_bars": int(len(test_df)),
        "test_bars_evaluated": int(len(y_test_true)),
        "model_params": model_params,
        "transformer_params": transformer_result["params"],
        "has_transformer": bool(has_transformer),
        "agreement_rate": float(agreement_mask.mean()) if len(agreement_mask) else 0.0,
    }
    return artifact, details


def train_from_dataset(
    data_path: str,
    model_output: str,
    symbol: str = DEFAULT_SYMBOL,
    threshold: float = DEFAULT_THRESHOLD_30S,
    horizon: int = 1,
    confidence: float = DEFAULT_CONFIDENCE,
    fee: float = 0.0004,
    tune_trials: int = DEFAULT_TUNE_TRIALS,
    optuna_storage: str = DEFAULT_OPTUNA_STORAGE,
    optuna_study_name: str = DEFAULT_OPTUNA_STUDY_NAME,
    include_futures_features: bool = DEFAULT_ENABLE_FUTURES_FEATURES,
    use_transformer: bool = DEFAULT_USE_TRANSFORMER,
    use_gpu: bool = DEFAULT_USE_GPU,
    transformer_seq_len: int = DEFAULT_TRANSFORMER_SEQ_LEN,
    transformer_epochs: int = DEFAULT_TRANSFORMER_EPOCHS,
    ensemble_weight_xgb: float = DEFAULT_ENSEMBLE_WEIGHT_XGB,
    require_model_agreement: bool = DEFAULT_REQUIRE_MODEL_AGREEMENT,
    verbose: bool = False,
) -> Dict:
    if verbose:
        print(f"[train] Loading dataset path={data_path}")

    raw_df = read_market_data(data_path)
    if verbose:
        print(f"[train] Loaded rows={len(raw_df):,}")

    futures_stats = {
        "funding_rows": 0,
        "open_interest_rows": 0,
        "taker_ratio_rows": 0,
    }
    if include_futures_features and not raw_df.empty:
        try:
            if verbose:
                print("[train] Enriching futures features...")
            raw_df, futures_stats = enrich_bars_with_futures_features(
                raw_df, symbol=symbol, period="5m"
            )
            if verbose:
                print(
                    "[train] Futures enrichment done "
                    f"funding={futures_stats['funding_rows']:,} "
                    f"oi={futures_stats['open_interest_rows']:,} "
                    f"ratio={futures_stats['taker_ratio_rows']:,}"
                )
        except Exception:
            futures_stats = {
                "funding_rows": 0,
                "open_interest_rows": 0,
                "taker_ratio_rows": 0,
            }
            if verbose:
                print("[train] Futures enrichment failed, continue without futures")

    artifact, details = train_from_dataframe(
        raw_df=raw_df,
        symbol=symbol,
        threshold=threshold,
        horizon=horizon,
        confidence=confidence,
        fee=fee,
        tune_trials=tune_trials,
        optuna_storage=optuna_storage,
        optuna_study_name=optuna_study_name,
        use_transformer=use_transformer,
        use_gpu=use_gpu,
        transformer_seq_len=transformer_seq_len,
        transformer_epochs=transformer_epochs,
        ensemble_weight_xgb=ensemble_weight_xgb,
        require_model_agreement=require_model_agreement,
        verbose=verbose,
    )
    if verbose:
        print(f"[train] Saving model path={model_output}")
    joblib.dump(artifact, model_output)
    details["model_output"] = model_output
    details["futures_stats"] = futures_stats
    return details


def get_latest_feature_vector_and_prediction(
    model_path: str,
    symbol: str = DEFAULT_SYMBOL,
    lookback_minutes: int = 360,
    include_futures_features: bool = DEFAULT_ENABLE_FUTURES_FEATURES,
    include_depth_features: bool = DEFAULT_ENABLE_DEPTH_FEATURES,
    use_gpu: bool = DEFAULT_USE_GPU,
) -> Dict:
    artifact = joblib.load(model_path)
    feature_cols = artifact["feature_cols"]
    confidence_threshold = float(artifact["confidence_threshold"])

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=lookback_minutes)
    rows = fetch_agg_trades(
        symbol=symbol,
        start_ms=int(start_time.timestamp() * 1000),
        end_ms=int(end_time.timestamp() * 1000),
    )
    trades_df = agg_trades_to_dataframe(rows)
    bars_df = aggregate_trades_to_30s(trades_df)

    if include_futures_features and not bars_df.empty:
        try:
            bars_df, _ = enrich_bars_with_futures_features(
                bars_df, symbol=symbol, period="5m"
            )
        except Exception:
            pass

    depth_snapshot = {}
    if include_depth_features:
        try:
            depth_snapshot = fetch_depth_snapshot_features(symbol=symbol, limit=100)
            for key, value in depth_snapshot.items():
                bars_df[key] = value
        except Exception:
            depth_snapshot = {}

    feat_df = add_features(bars_df)

    if feat_df.empty:
        raise ValueError("Khong lay duoc du lieu trade hien tai tu Binance.")

    latest = feat_df.iloc[-1]
    x_live = np.array([latest.get(col, np.nan) for col in feature_cols], dtype=float)
    if np.isnan(x_live).any():
        raise ValueError(
            "Feature hien tai chua du du lieu lich su, hay doi them it nhat vai phut."
        )

    if "models" in artifact:
        xgb_model = artifact["models"]["xgb"]
        tf_payload = artifact["models"].get("transformer")
        ensemble_cfg = artifact.get("ensemble", {})
    else:
        xgb_model = artifact["model"]
        tf_payload = None
        ensemble_cfg = {"weight_xgb": 1.0, "require_model_agreement": False}

    probs_xgb = xgb_model.predict_proba(x_live.reshape(1, -1))[0]
    probs_tf = None
    if tf_payload is not None:
        from trading_pipeline.model.transformer_pipeline import (
            transformer_predict_last_window,
        )

        probs_tf = transformer_predict_last_window(
            feat_df,
            feature_cols,
            tf_payload,
            use_gpu=use_gpu,
        )

    if probs_tf is not None:
        weight_xgb = float(ensemble_cfg.get("weight_xgb", 0.55))
        probs = weight_xgb * probs_xgb + (1.0 - weight_xgb) * probs_tf
        agreement = int(np.argmax(probs_xgb)) == int(np.argmax(probs_tf))
    else:
        probs = probs_xgb
        agreement = True

    direction = int(np.array([-1, 0, 1])[int(np.argmax(probs))])
    confidence = float(np.max(probs))
    require_agreement = bool(ensemble_cfg.get("require_model_agreement", False))
    is_confirmed = agreement or (not require_agreement)
    action = direction if confidence >= confidence_threshold and is_confirmed else 0

    return {
        "timestamp": str(latest["close_time"]),
        "close": float(latest["close"]),
        "prob_down": float(probs[0]),
        "prob_flat": float(probs[1]),
        "prob_up": float(probs[2]),
        "direction": direction,
        "confidence": confidence,
        "action": action,
        "is_confirmed": bool(is_confirmed),
        "prob_xgb": probs_xgb.tolist(),
        "prob_transformer": probs_tf.tolist() if probs_tf is not None else None,
        "eval_summary": artifact.get("eval_summary", {}),
        "feature_vector": {k: float(v) for k, v in zip(feature_cols, x_live)},
        "depth_snapshot": depth_snapshot,
    }


def upgrade_data_and_retrain(
    symbol: str,
    data_path: str,
    model_output: str,
    lookback_days_if_empty: int = DEFAULT_LOOKBACK_DAYS,
    threshold: float = DEFAULT_THRESHOLD_30S,
    horizon: int = 1,
    confidence: float = DEFAULT_CONFIDENCE,
    fee: float = 0.0004,
    include_futures_features: bool = DEFAULT_ENABLE_FUTURES_FEATURES,
    tune_trials: int = DEFAULT_TUNE_TRIALS,
    optuna_storage: str = DEFAULT_OPTUNA_STORAGE,
    optuna_study_name: str = DEFAULT_OPTUNA_STUDY_NAME,
    use_transformer: bool = DEFAULT_USE_TRANSFORMER,
    use_gpu: bool = DEFAULT_USE_GPU,
    transformer_seq_len: int = DEFAULT_TRANSFORMER_SEQ_LEN,
    transformer_epochs: int = DEFAULT_TRANSFORMER_EPOCHS,
    ensemble_weight_xgb: float = DEFAULT_ENSEMBLE_WEIGHT_XGB,
    require_model_agreement: bool = DEFAULT_REQUIRE_MODEL_AGREEMENT,
    verbose: bool = False,
) -> Dict:
    merged_df, update_stats = update_30s_dataset(
        symbol=symbol,
        data_path=data_path,
        default_lookback_days=lookback_days_if_empty,
        include_futures_features=include_futures_features,
        verbose=verbose,
    )

    artifact, train_stats = train_from_dataframe(
        raw_df=merged_df,
        symbol=symbol,
        threshold=threshold,
        horizon=horizon,
        confidence=confidence,
        fee=fee,
        tune_trials=tune_trials,
        optuna_storage=optuna_storage,
        optuna_study_name=optuna_study_name,
        use_transformer=use_transformer,
        use_gpu=use_gpu,
        transformer_seq_len=transformer_seq_len,
        transformer_epochs=transformer_epochs,
        ensemble_weight_xgb=ensemble_weight_xgb,
        require_model_agreement=require_model_agreement,
        verbose=verbose,
    )
    joblib.dump(artifact, model_output)

    return {
        "update": update_stats,
        "train": train_stats,
        "model_output": model_output,
    }
