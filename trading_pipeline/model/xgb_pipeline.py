import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from trading_pipeline.features.engineering import encode_target


def _build_sample_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    num_classes = max(len(counts), 1)
    total = float(len(y))
    class_weight = {k: total / (num_classes * v) for k, v in counts.items()}
    weights = y.map(class_weight).astype(float).to_numpy()
    return weights


def _normalize_optuna_storage(storage: Optional[str]) -> Optional[str]:
    if storage is None:
        return None
    value = str(storage).strip()
    if value == "" or value.lower() in {"none", "null", "memory", ":memory:"}:
        return None
    if "://" in value:
        return value
    abs_path = os.path.abspath(value)
    return f"sqlite:///{abs_path.replace('\\', '/')}"


def _tune_hyperparams_optuna(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    train_w: np.ndarray,
    valid_w: np.ndarray,
    trials: int,
    optuna_storage: Optional[str] = None,
    optuna_study_name: Optional[str] = None,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    import optuna

    storage_url = _normalize_optuna_storage(optuna_storage)
    load_if_exists = storage_url is not None
    study_name = optuna_study_name if load_if_exists else None

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=load_if_exists,
    )
    completed_state = optuna.trial.TrialState.COMPLETE
    completed_before = len([t for t in study.trials if t.state == completed_state])
    remaining_trials = max(int(trials) - completed_before, 0)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 500, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "eval_metric": "mlogloss",
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            x_train,
            y_train,
            sample_weight=train_w,
            eval_set=[(x_valid, y_valid)],
            sample_weight_eval_set=[valid_w],
            verbose=False,
        )
        probs = model.predict_proba(x_valid)
        return float(log_loss(y_valid, probs, sample_weight=valid_w, labels=[0, 1, 2]))

    callbacks = []
    if verbose:
        where = storage_url if storage_url is not None else "in-memory"
        print(
            "[train] Optuna tuning started "
            f"requested_trials={trials} completed_before={completed_before} "
            f"remaining={remaining_trials} storage={where}"
        )

        def _cb(study: optuna.Study, trial: optuna.Trial) -> None:
            done = len([t for t in study.trials if t.state == completed_state])
            print(
                "[train] optuna "
                f"trial={done}/{trials} "
                f"value={trial.value:.6f} "
                f"best={study.best_value:.6f}"
            )

        callbacks.append(_cb)

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=verbose,
            callbacks=callbacks,
        )
    elif verbose:
        print("[train] Optuna resume: enough completed trials, skip tuning")

    completed_after = len([t for t in study.trials if t.state == completed_state])
    summary = {
        "study_name": study.study_name,
        "storage": storage_url,
        "requested_trials": int(trials),
        "completed_before": int(completed_before),
        "completed_after": int(completed_after),
        "ran_new_trials": int(max(completed_after - completed_before, 0)),
        "best_value": float(study.best_value),
    }
    if verbose:
        print(f"[train] Optuna tuning done best_value={study.best_value:.6f}")
    return dict(study.best_params), summary


def train_xgboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    tune_trials: int = 0,
    optuna_storage: Optional[str] = None,
    optuna_study_name: Optional[str] = None,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Tuple[xgb.XGBClassifier, Dict]:
    x_train = train_df[feature_cols]
    y_train = encode_target(train_df["target"]).astype(int)

    x_valid = valid_df[feature_cols]
    y_valid = encode_target(valid_df["target"]).astype(int)

    train_w = _build_sample_weight(y_train)
    valid_w = _build_sample_weight(y_valid)

    best_params: Dict = {}
    optuna_summary: Dict = {}
    if tune_trials > 0:
        best_params, optuna_summary = _tune_hyperparams_optuna(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            train_w=train_w,
            valid_w=valid_w,
            trials=tune_trials,
            optuna_storage=optuna_storage,
            optuna_study_name=optuna_study_name,
            use_gpu=use_gpu,
            verbose=verbose,
        )

    base_params: Dict = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "gamma": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "eval_metric": "mlogloss",
    }
    base_params.update(best_params)

    model = xgb.XGBClassifier(**base_params)

    if verbose:
        print(
            "[train] XGBoost fit started "
            f"train_rows={len(x_train):,} valid_rows={len(x_valid):,} features={len(feature_cols)}"
        )

    try:
        model.fit(
            x_train,
            y_train,
            sample_weight=train_w,
            eval_set=[(x_valid, y_valid)],
            sample_weight_eval_set=[valid_w],
            verbose=False,
        )
    except Exception:
        if not use_gpu:
            raise
        if verbose:
            print("[train] XGBoost GPU unavailable, fallback to CPU")
        base_params["device"] = "cpu"
        model = xgb.XGBClassifier(**base_params)
        model.fit(
            x_train,
            y_train,
            sample_weight=train_w,
            eval_set=[(x_valid, y_valid)],
            sample_weight_eval_set=[valid_w],
            verbose=False,
        )
    if verbose:
        print("[train] XGBoost fit done")

    return_params = dict(base_params)
    if optuna_summary:
        return_params["_optuna"] = optuna_summary
    return model, return_params
