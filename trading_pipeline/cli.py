import argparse
import time
from trading_pipeline.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_ENSEMBLE_WEIGHT_XGB,
    DEFAULT_ENABLE_DEPTH_FEATURES,
    DEFAULT_ENABLE_FUTURES_FEATURES,
    DEFAULT_FORCE_CONTINUOUS_TRADE,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_PREDICT_LOOKBACK_MINUTES,
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
)
from trading_pipeline.data.binance_client import update_30s_dataset
from trading_pipeline.pipeline import (
    get_latest_feature_vector_and_prediction,
    train_from_dataset,
    upgrade_data_and_retrain,
)


def command_download(args: argparse.Namespace) -> None:
    t0 = time.time()
    _, stats = update_30s_dataset(
        symbol=args.symbol,
        data_path=args.output,
        default_lookback_days=args.days,
        include_futures_features=args.include_futures_features,
        verbose=True,
    )
    print(f"Updated data file: {args.output}")
    print(f"- raw_trades: {stats['raw_trades']}")
    print(f"- new_bars_30s: {stats['new_bars']}")
    print(f"- total_bars_30s: {stats['total_bars']}")
    print(f"- funding_rows: {stats['funding_rows']}")
    print(f"- open_interest_rows: {stats['open_interest_rows']}")
    print(f"- taker_ratio_rows: {stats['taker_ratio_rows']}")
    print(f"- elapsed_seconds: {time.time() - t0:.1f}")


def command_train(args: argparse.Namespace) -> None:
    t0 = time.time()
    details = train_from_dataset(
        data_path=args.data,
        model_output=args.model_output,
        symbol=args.symbol,
        threshold=args.threshold,
        horizon=args.horizon,
        confidence=args.confidence,
        fee=args.fee,
        tune_trials=args.tune_trials,
        optuna_storage=args.optuna_storage,
        optuna_study_name=args.optuna_study_name,
        include_futures_features=args.include_futures_features,
        use_transformer=args.use_transformer,
        use_gpu=args.use_gpu,
        transformer_seq_len=args.transformer_seq_len,
        transformer_epochs=args.transformer_epochs,
        ensemble_weight_xgb=args.ensemble_weight_xgb,
        require_model_agreement=args.require_model_agreement,
        force_continuous_trade=args.force_continuous_trade,
        verbose=True,
    )
    print("Classification report (test):")
    print(details["classification_report"])
    print("Confusion matrix [rows=true, cols=pred] for [-1,0,1]:")
    print(details["confusion_matrix"])
    print("Backtest metrics (test):")
    for k, v in details["backtest_metrics"].items():
        print(f"- {k}: {v:.6f}")
    print(f"Classification metrics: {details['classification_metrics']}")
    print(f"Bars used: {details['bars_used']}")
    print(f"Bars evaluated: {details['test_bars_evaluated']}")
    print(f"Has transformer: {details['has_transformer']}")
    print(f"Agreement rate: {details['agreement_rate']:.4f}")
    print(f"Futures stats: {details['futures_stats']}")
    print(f"Model params: {details['model_params']}")
    print(f"Transformer params: {details['transformer_params']}")
    print(f"Model saved to {details['model_output']}")
    print(f"- elapsed_seconds: {time.time() - t0:.1f}")


def command_upgrade(args: argparse.Namespace) -> None:
    t0 = time.time()
    details = upgrade_data_and_retrain(
        symbol=args.symbol,
        data_path=args.data,
        model_output=args.model_output,
        lookback_days_if_empty=args.days,
        threshold=args.threshold,
        horizon=args.horizon,
        confidence=args.confidence,
        fee=args.fee,
        include_futures_features=args.include_futures_features,
        tune_trials=args.tune_trials,
        optuna_storage=args.optuna_storage,
        optuna_study_name=args.optuna_study_name,
        use_transformer=args.use_transformer,
        use_gpu=args.use_gpu,
        transformer_seq_len=args.transformer_seq_len,
        transformer_epochs=args.transformer_epochs,
        ensemble_weight_xgb=args.ensemble_weight_xgb,
        require_model_agreement=args.require_model_agreement,
        force_continuous_trade=args.force_continuous_trade,
        verbose=True,
    )
    update = details["update"]
    train = details["train"]
    print("Data update:")
    print(f"- raw_trades: {update['raw_trades']}")
    print(f"- new_bars_30s: {update['new_bars']}")
    print(f"- total_bars_30s: {update['total_bars']}")
    print(f"- funding_rows: {update['funding_rows']}")
    print(f"- open_interest_rows: {update['open_interest_rows']}")
    print(f"- taker_ratio_rows: {update['taker_ratio_rows']}")
    print("Train result:")
    print(f"- bars_used: {train['bars_used']}")
    for k, v in train["backtest_metrics"].items():
        print(f"- {k}: {v:.6f}")
    print(f"- classification_metrics: {train['classification_metrics']}")
    print(f"- has_transformer: {train['has_transformer']}")
    print(f"- agreement_rate: {train['agreement_rate']:.4f}")
    print(f"- model_params: {train['model_params']}")
    print(f"- transformer_params: {train['transformer_params']}")
    print(f"Model saved to {details['model_output']}")
    print(f"- elapsed_seconds: {time.time() - t0:.1f}")


def command_predict(args: argparse.Namespace) -> None:
    result = get_latest_feature_vector_and_prediction(
        model_path=args.model,
        symbol=args.symbol,
        lookback_minutes=args.lookback_minutes,
        include_futures_features=args.include_futures_features,
        include_depth_features=args.include_depth_features,
        use_gpu=args.use_gpu,
        force_continuous_trade=args.force_continuous_trade,
    )
    side = (
        "BUY" if result["action"] > 0 else ("SELL" if result["action"] < 0 else "HOLD")
    )
    side_prob = max(float(result["prob_up"]), float(result["prob_down"]))
    print(
        f"signal={side} probability={side_prob*100:.2f}% "
        f"price={result['close']:.2f} ts={result['timestamp']}"
    )


def command_live(args: argparse.Namespace) -> None:
    from trading_pipeline.live.predictor import LiveConfig, LivePredictor

    cfg = LiveConfig(
        model_path=args.model,
        symbol=args.symbol,
        bootstrap_bars=args.bootstrap_bars,
        force_continuous_trade=args.force_continuous_trade,
    )
    live = LivePredictor(cfg)
    live.run()


def command_gui(_args: argparse.Namespace) -> None:
    from trading_pipeline.gui_app import run_gui

    run_gui()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BTC 30s direction prediction pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser(
        "download", help="Update 30s dataset from Binance aggTrades"
    )
    p_download.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p_download.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p_download.add_argument("--output", default="btc_30s.parquet")
    p_download.add_argument(
        "--include-futures-features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_FUTURES_FEATURES,
    )
    p_download.set_defaults(func=command_download)

    p_train = subparsers.add_parser(
        "train", help="Train and backtest model on 30s bars"
    )
    p_train.add_argument("--data", default="btc_30s.parquet")
    p_train.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p_train.add_argument("--horizon", type=int, default=1)
    p_train.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_30S)
    p_train.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    p_train.add_argument("--fee", type=float, default=0.0004)
    p_train.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS)
    p_train.add_argument("--optuna-storage", default=DEFAULT_OPTUNA_STORAGE)
    p_train.add_argument("--optuna-study-name", default=DEFAULT_OPTUNA_STUDY_NAME)
    p_train.add_argument(
        "--use-transformer",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_TRANSFORMER,
    )
    p_train.add_argument(
        "--transformer-seq-len", type=int, default=DEFAULT_TRANSFORMER_SEQ_LEN
    )
    p_train.add_argument(
        "--transformer-epochs", type=int, default=DEFAULT_TRANSFORMER_EPOCHS
    )
    p_train.add_argument(
        "--ensemble-weight-xgb", type=float, default=DEFAULT_ENSEMBLE_WEIGHT_XGB
    )
    p_train.add_argument(
        "--require-model-agreement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_MODEL_AGREEMENT,
    )
    p_train.add_argument(
        "--force-continuous-trade",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FORCE_CONTINUOUS_TRADE,
    )
    p_train.add_argument(
        "--include-futures-features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_FUTURES_FEATURES,
    )
    p_train.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_GPU,
    )
    p_train.add_argument("--model-output", default=DEFAULT_MODEL_FILENAME)
    p_train.set_defaults(func=command_train)

    p_upgrade = subparsers.add_parser(
        "upgrade",
        help="Append new Binance data to existing dataset and retrain on full data",
    )
    p_upgrade.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p_upgrade.add_argument("--data", default="btc_30s.parquet")
    p_upgrade.add_argument("--model-output", default=DEFAULT_MODEL_FILENAME)
    p_upgrade.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p_upgrade.add_argument("--horizon", type=int, default=1)
    p_upgrade.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_30S)
    p_upgrade.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    p_upgrade.add_argument("--fee", type=float, default=0.0004)
    p_upgrade.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS)
    p_upgrade.add_argument("--optuna-storage", default=DEFAULT_OPTUNA_STORAGE)
    p_upgrade.add_argument("--optuna-study-name", default=DEFAULT_OPTUNA_STUDY_NAME)
    p_upgrade.add_argument(
        "--use-transformer",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_TRANSFORMER,
    )
    p_upgrade.add_argument(
        "--transformer-seq-len", type=int, default=DEFAULT_TRANSFORMER_SEQ_LEN
    )
    p_upgrade.add_argument(
        "--transformer-epochs", type=int, default=DEFAULT_TRANSFORMER_EPOCHS
    )
    p_upgrade.add_argument(
        "--ensemble-weight-xgb", type=float, default=DEFAULT_ENSEMBLE_WEIGHT_XGB
    )
    p_upgrade.add_argument(
        "--require-model-agreement",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_MODEL_AGREEMENT,
    )
    p_upgrade.add_argument(
        "--force-continuous-trade",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FORCE_CONTINUOUS_TRADE,
    )
    p_upgrade.add_argument(
        "--include-futures-features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_FUTURES_FEATURES,
    )
    p_upgrade.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_GPU,
    )
    p_upgrade.set_defaults(func=command_upgrade)

    p_predict = subparsers.add_parser(
        "predict",
        help="Fetch current Binance trades, build latest feature vector, and predict next 30s",
    )
    p_predict.add_argument("--model", default=DEFAULT_MODEL_FILENAME)
    p_predict.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p_predict.add_argument(
        "--lookback-minutes", type=int, default=DEFAULT_PREDICT_LOOKBACK_MINUTES
    )
    p_predict.add_argument(
        "--include-futures-features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_FUTURES_FEATURES,
    )
    p_predict.add_argument(
        "--include-depth-features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_DEPTH_FEATURES,
    )
    p_predict.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_GPU,
    )
    p_predict.add_argument(
        "--force-continuous-trade",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p_predict.set_defaults(func=command_predict)

    p_live = subparsers.add_parser(
        "live", help="Run live prediction from trade stream aggregated to 30s"
    )
    p_live.add_argument("--model", default=DEFAULT_MODEL_FILENAME)
    p_live.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p_live.add_argument("--bootstrap-bars", type=int, default=1500)
    p_live.add_argument(
        "--force-continuous-trade",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p_live.set_defaults(func=command_live)

    p_gui = subparsers.add_parser(
        "gui", help="Run desktop GUI with Predict and Upgrade buttons"
    )
    p_gui.set_defaults(func=command_gui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
