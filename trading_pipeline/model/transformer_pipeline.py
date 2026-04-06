from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from trading_pipeline.features.engineering import encode_target


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_features]
        h = self.input_proj(x)
        h = h + self.pos_embed[:, : h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.cls_head(h)


def _make_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if len(features) < seq_len:
        return (
            np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            0,
        )

    xs = []
    ys = []
    for i in range(seq_len - 1, len(features)):
        xs.append(features[i - seq_len + 1 : i + 1])
        ys.append(labels[i])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), seq_len - 1


def _class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    n_cls = max(len(classes), 1)
    weights = np.ones(3, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        weights[int(c)] = total / (n_cls * float(cnt))
    return torch.tensor(weights, dtype=torch.float32)


def train_transformer(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 120,
    epochs: int = 12,
    batch_size: int = 256,
    test_batch_size: int = 512,
    learning_rate: float = 1e-3,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Dict:
    n_features = len(feature_cols)
    if len(train_df) < seq_len + 200:
        return {
            "payload": None,
            "test_probs": np.empty((0, 3), dtype=np.float32),
            "test_offset": 0,
            "params": {},
        }

    scaler = StandardScaler()
    train_x_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    scaler.fit(train_x_raw)

    def _prep(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
        x = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32))
        y = encode_target(df["target"]).astype(int).to_numpy()
        return _make_sequences(x, y, seq_len=seq_len)

    train_x, train_y, _ = _prep(train_df)
    valid_x, valid_y, _ = _prep(valid_df)
    test_x, _, test_offset = _prep(test_df)

    if len(train_x) == 0 or len(valid_x) == 0:
        return {
            "payload": None,
            "test_probs": np.empty((0, 3), dtype=np.float32),
            "test_offset": 0,
            "params": {},
        }

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = SequenceTransformer(
        n_features=n_features,
        seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        num_classes=3,
    ).to(device)

    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    valid_ds = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    class_w = _class_weights(train_y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    best_state = None
    best_valid_loss = float("inf")
    patience = 3
    no_improve = 0

    if verbose:
        print(
            "[train] Transformer started "
            f"device={device.type} seq_len={seq_len} epochs={epochs}"
        )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.size(0)

        train_loss = train_loss_sum / max(len(train_ds), 1)

        model.eval()
        valid_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                valid_loss_sum += float(loss.item()) * xb.size(0)

        valid_loss = valid_loss_sum / max(len(valid_ds), 1)
        if verbose:
            print(
                f"[train] transformer epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}"
            )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print("[train] transformer early stop")
                break

    if best_state is None:
        return {
            "payload": None,
            "test_probs": np.empty((0, 3), dtype=np.float32),
            "test_offset": 0,
            "params": {},
        }

    model.load_state_dict(best_state)
    model.eval()

    test_probs = np.empty((0, 3), dtype=np.float32)
    if len(test_x) > 0:
        infer_device = device

        def _batched_predict_probs(
            infer_model: SequenceTransformer,
            x_arr: np.ndarray,
            device_obj: torch.device,
            chunk_size: int,
        ) -> np.ndarray:
            out_parts = []
            chunk_size = max(int(chunk_size), 1)
            with torch.no_grad():
                for i in range(0, len(x_arr), chunk_size):
                    xb = torch.from_numpy(x_arr[i : i + chunk_size]).to(device_obj)
                    logits = infer_model(xb)
                    probs = (
                        F.softmax(logits, dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    out_parts.append(probs)
            return (
                np.concatenate(out_parts, axis=0)
                if out_parts
                else np.empty((0, 3), dtype=np.float32)
            )

        try:
            test_probs = _batched_predict_probs(
                model,
                test_x,
                infer_device,
                test_batch_size,
            )
        except torch.OutOfMemoryError:
            if infer_device.type != "cuda":
                raise
            if verbose:
                print("[train] Transformer test inference OOM on GPU, fallback to CPU")
            torch.cuda.empty_cache()
            infer_device = torch.device("cpu")
            model_cpu = SequenceTransformer(
                n_features=n_features,
                seq_len=seq_len,
                d_model=64,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                num_classes=3,
            ).to(infer_device)
            model_cpu.load_state_dict(best_state)
            model_cpu.eval()
            test_probs = _batched_predict_probs(
                model_cpu,
                test_x,
                infer_device,
                test_batch_size,
            )

    payload = {
        "state_dict": best_state,
        "seq_len": seq_len,
        "n_features": n_features,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }

    params = {
        "seq_len": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "learning_rate": learning_rate,
        "device": device.type,
    }
    return {
        "payload": payload,
        "test_probs": test_probs,
        "test_offset": int(test_offset),
        "params": params,
    }


def _build_model_from_payload(
    payload: Dict, device: torch.device
) -> SequenceTransformer:
    model = SequenceTransformer(
        n_features=int(payload["n_features"]),
        seq_len=int(payload["seq_len"]),
        d_model=int(payload["d_model"]),
        nhead=int(payload["nhead"]),
        num_layers=int(payload["num_layers"]),
        dropout=float(payload["dropout"]),
        num_classes=3,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def transformer_predict_last_window(
    feature_frame: pd.DataFrame,
    feature_cols: List[str],
    payload: Optional[Dict],
    use_gpu: bool = False,
) -> Optional[np.ndarray]:
    if payload is None:
        return None

    seq_len = int(payload["seq_len"])
    if len(feature_frame) < seq_len:
        return None

    x = feature_frame[feature_cols].to_numpy(dtype=np.float32)
    mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
    x = (x - mean) / (scale + 1e-12)
    x = x[-seq_len:]

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = _build_model_from_payload(payload, device)

    with torch.no_grad():
        tx = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(tx)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float32)
    return probs
