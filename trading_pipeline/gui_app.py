import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from trading_pipeline.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_ENSEMBLE_WEIGHT_XGB,
    DEFAULT_ENABLE_DEPTH_FEATURES,
    DEFAULT_ENABLE_FUTURES_FEATURES,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_REQUIRE_MODEL_AGREEMENT,
    DEFAULT_SYMBOL,
    DEFAULT_TRANSFORMER_EPOCHS,
    DEFAULT_TRANSFORMER_SEQ_LEN,
    DEFAULT_TUNE_TRIALS,
    DEFAULT_THRESHOLD_30S,
    DEFAULT_USE_GPU,
    DEFAULT_USE_TRANSFORMER,
)
from trading_pipeline.pipeline import (
    get_latest_feature_vector_and_prediction,
    upgrade_data_and_retrain,
)


class TradingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BTC 30s Predictor")
        self.root.geometry("940x620")

        self.symbol_var = tk.StringVar(value=DEFAULT_SYMBOL)
        self.data_var = tk.StringVar(value="btc_30s.parquet")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL_FILENAME)
        self.days_var = tk.StringVar(value=str(DEFAULT_LOOKBACK_DAYS))
        self.horizon_var = tk.StringVar(value="1")
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD_30S))
        self.confidence_var = tk.StringVar(value=str(DEFAULT_CONFIDENCE))
        self.fee_var = tk.StringVar(value="0.0004")
        self.lookback_var = tk.StringVar(value="360")
        self.tune_trials_var = tk.StringVar(value=str(DEFAULT_TUNE_TRIALS))
        self.transformer_seq_len_var = tk.StringVar(
            value=str(DEFAULT_TRANSFORMER_SEQ_LEN)
        )
        self.transformer_epochs_var = tk.StringVar(
            value=str(DEFAULT_TRANSFORMER_EPOCHS)
        )
        self.ensemble_weight_xgb_var = tk.StringVar(
            value=str(DEFAULT_ENSEMBLE_WEIGHT_XGB)
        )
        self.use_futures_var = tk.BooleanVar(value=DEFAULT_ENABLE_FUTURES_FEATURES)
        self.use_depth_var = tk.BooleanVar(value=DEFAULT_ENABLE_DEPTH_FEATURES)
        self.use_transformer_var = tk.BooleanVar(value=DEFAULT_USE_TRANSFORMER)
        self.require_agreement_var = tk.BooleanVar(
            value=DEFAULT_REQUIRE_MODEL_AGREEMENT
        )
        self.use_gpu_var = tk.BooleanVar(value=DEFAULT_USE_GPU)

        self._build_layout()

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Symbol").grid(
            row=0, column=0, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.symbol_var, width=16).grid(
            row=0, column=1, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Data file").grid(
            row=0, column=2, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.data_var, width=28).grid(
            row=0, column=3, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Model file").grid(
            row=0, column=4, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.model_var, width=28).grid(
            row=0, column=5, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Days if empty").grid(
            row=1, column=0, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.days_var, width=16).grid(
            row=1, column=1, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Horizon bars").grid(
            row=1, column=2, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.horizon_var, width=8).grid(
            row=1, column=3, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Threshold").grid(
            row=1, column=4, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.threshold_var, width=10).grid(
            row=1, column=5, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Confidence").grid(
            row=2, column=0, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.confidence_var, width=16).grid(
            row=2, column=1, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Fee per trade").grid(
            row=2, column=2, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.fee_var, width=8).grid(
            row=2, column=3, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Predict lookback minutes").grid(
            row=2, column=4, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.lookback_var, width=10).grid(
            row=2, column=5, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Tune trials").grid(
            row=3, column=0, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.tune_trials_var, width=16).grid(
            row=3, column=1, sticky=tk.W, padx=4, pady=4
        )

        ttk.Checkbutton(
            controls,
            text="Use futures features",
            variable=self.use_futures_var,
        ).grid(row=3, column=2, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(
            controls,
            text="Use depth features (predict)",
            variable=self.use_depth_var,
        ).grid(row=3, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(
            controls,
            text="Use Transformer",
            variable=self.use_transformer_var,
        ).grid(row=3, column=4, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(
            controls,
            text="Require model agreement",
            variable=self.require_agreement_var,
        ).grid(row=3, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(
            controls,
            text="Use GPU (train/predict)",
            variable=self.use_gpu_var,
        ).grid(row=5, column=0, sticky=tk.W, padx=4, pady=4)

        ttk.Label(controls, text="Transformer seq len").grid(
            row=4, column=0, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.transformer_seq_len_var, width=16).grid(
            row=4, column=1, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Transformer epochs").grid(
            row=4, column=2, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.transformer_epochs_var, width=8).grid(
            row=4, column=3, sticky=tk.W, padx=4, pady=4
        )

        ttk.Label(controls, text="Ensemble weight XGB").grid(
            row=4, column=4, sticky=tk.W, padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.ensemble_weight_xgb_var, width=10).grid(
            row=4, column=5, sticky=tk.W, padx=4, pady=4
        )

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=8)

        ttk.Button(actions, text="Predict", command=self._predict_clicked).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            actions, text="Upgrade (append + retrain)", command=self._upgrade_clicked
        ).pack(side=tk.LEFT, padx=4)

        self.log_box = ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 10))
        self.log_box.pack(fill=tk.BOTH, expand=True)

        self._log(
            "Ready. Predict: lay feature vector hien tai va du doan 30s tiep theo."
        )
        self._log("Upgrade: noi data moi vao data cu roi train lai tren full dataset.")

    def _log(self, message: str) -> None:
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def _read_inputs(self):
        return {
            "symbol": self.symbol_var.get().strip().upper(),
            "data": self.data_var.get().strip(),
            "model": self.model_var.get().strip(),
            "days": int(self.days_var.get().strip()),
            "horizon": int(self.horizon_var.get().strip()),
            "threshold": float(self.threshold_var.get().strip()),
            "confidence": float(self.confidence_var.get().strip()),
            "fee": float(self.fee_var.get().strip()),
            "lookback": int(self.lookback_var.get().strip()),
            "tune_trials": int(self.tune_trials_var.get().strip()),
            "transformer_seq_len": int(self.transformer_seq_len_var.get().strip()),
            "transformer_epochs": int(self.transformer_epochs_var.get().strip()),
            "ensemble_weight_xgb": float(self.ensemble_weight_xgb_var.get().strip()),
            "include_futures": bool(self.use_futures_var.get()),
            "include_depth": bool(self.use_depth_var.get()),
            "use_transformer": bool(self.use_transformer_var.get()),
            "require_model_agreement": bool(self.require_agreement_var.get()),
            "use_gpu": bool(self.use_gpu_var.get()),
        }

    def _predict_clicked(self) -> None:
        def worker():
            try:
                cfg = self._read_inputs()
                self._log("[Predict] Dang lay du lieu trade hien tai tu Binance...")
                result = get_latest_feature_vector_and_prediction(
                    model_path=cfg["model"],
                    symbol=cfg["symbol"],
                    lookback_minutes=cfg["lookback"],
                    include_futures_features=cfg["include_futures"],
                    include_depth_features=cfg["include_depth"],
                    use_gpu=cfg["use_gpu"],
                )
                self._log(
                    f"[Predict] ts={result['timestamp']} close={result['close']:.2f} "
                    f"down={result['prob_down']:.3f} flat={result['prob_flat']:.3f} up={result['prob_up']:.3f} "
                    f"conf={result['confidence']:.3f} action={result['action']}"
                )
                self._log(f"[Predict] confirmed={result.get('is_confirmed', False)}")

                px = result.get("prob_xgb")
                if px is not None:
                    self._log(
                        f"[Predict] XGB probs: down={px[0]:.3f} flat={px[1]:.3f} up={px[2]:.3f}"
                    )

                pt = result.get("prob_transformer")
                if pt is not None:
                    self._log(
                        f"[Predict] Transformer probs: down={pt[0]:.3f} flat={pt[1]:.3f} up={pt[2]:.3f}"
                    )

                eval_summary = result.get("eval_summary", {})
                cls_metrics = eval_summary.get("classification_metrics", {})
                if cls_metrics:
                    acc = float(cls_metrics.get("accuracy", 0.0)) * 100.0
                    bacc = float(cls_metrics.get("balanced_accuracy", 0.0)) * 100.0
                    f1m = float(cls_metrics.get("f1_macro", 0.0)) * 100.0
                    self._log(
                        f"[Predict] Historical test: accuracy={acc:.2f}% balanced_acc={bacc:.2f}% f1_macro={f1m:.2f}%"
                    )

                fv = result["feature_vector"]
                preview = ", ".join([f"{k}={v:.5f}" for k, v in list(fv.items())[:8]])
                self._log(f"[Predict] feature_vector_preview: {preview}")
            except Exception as exc:
                self._log(f"[Predict][Error] {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _upgrade_clicked(self) -> None:
        def worker():
            try:
                cfg = self._read_inputs()
                self._log(
                    "[Upgrade] Bat dau append data moi vao data cu + train lai full dataset..."
                )
                result = upgrade_data_and_retrain(
                    symbol=cfg["symbol"],
                    data_path=cfg["data"],
                    model_output=cfg["model"],
                    lookback_days_if_empty=cfg["days"],
                    threshold=cfg["threshold"],
                    horizon=cfg["horizon"],
                    confidence=cfg["confidence"],
                    fee=cfg["fee"],
                    include_futures_features=cfg["include_futures"],
                    tune_trials=cfg["tune_trials"],
                    use_transformer=cfg["use_transformer"],
                    use_gpu=cfg["use_gpu"],
                    transformer_seq_len=cfg["transformer_seq_len"],
                    transformer_epochs=cfg["transformer_epochs"],
                    ensemble_weight_xgb=cfg["ensemble_weight_xgb"],
                    require_model_agreement=cfg["require_model_agreement"],
                )
                update = result["update"]
                train = result["train"]
                self._log(
                    f"[Upgrade] Data updated raw_trades={update['raw_trades']} "
                    f"new_bars_30s={update['new_bars']} total_bars_30s={update['total_bars']} "
                    f"funding={update['funding_rows']} oi={update['open_interest_rows']} ratio={update['taker_ratio_rows']}"
                )
                self._log(
                    f"[Upgrade] Train bars_used={train['bars_used']} "
                    f"profit={train['backtest_metrics']['profit']:.6f} "
                    f"sharpe={train['backtest_metrics']['sharpe']:.6f} "
                    f"max_dd={train['backtest_metrics']['max_drawdown']:.6f}"
                )
                self._log(f"[Upgrade] Model params: {train['model_params']}")
                self._log(f"[Upgrade] Model saved: {result['model_output']}")
            except Exception as exc:
                self._log(f"[Upgrade][Error] {exc}")

        threading.Thread(target=worker, daemon=True).start()


def run_gui() -> None:
    root = tk.Tk()
    TradingGUI(root)
    root.mainloop()
