# BTC 30s Direction Pipeline

Pipeline du doan huong BTCUSDT trong 30 giay toi voi 2 cach su dung: GUI (nhan nut) hoac CLI.

## Dung gi

- Streaming: Binance trade stream + aggTrades REST
- Exogenous data: Funding rate, Open Interest, Taker long/short ratio (Binance Futures)
- Live microstructure: Order book depth snapshot (spread + imbalance)
- Data processing: Python, Pandas, NumPy
- Feature engineering: multi-timeframe return/momentum, EMA/MACD/RSI/ATR, volatility regime, futures + depth signals
- Model ensemble: XGBoost + Transformer (2-layer confirmation) + Optuna tuning
- Backtest metrics: profit sau fee, Sharpe ratio, max drawdown
- Mo rong nang cao: PyTorch (LSTM/Transformer), Redis, Kafka, vectorbt

## Cau truc

- main.py: entrypoint
- trading_pipeline/data: fetch aggTrades + gom nen 30s + update incremental
- trading_pipeline/features: tao feature/target
- trading_pipeline/model: train model
- trading_pipeline/backtest: tinh metrics trading
- trading_pipeline/live: live predictor tu trade stream
- trading_pipeline/pipeline.py: service train/predict/upgrade
- trading_pipeline/gui_app.py: GUI Predict + Upgrade
- trading_pipeline/cli.py: command line interface

## Cai thu vien

pip install -r requirements.txt

## Chay GUI

python main.py

- Nut Predict: lay feature vector hien tai tu Binance va du doan 30s tiep theo.
- Nut Upgrade: noi data moi vao data cu, bo sung futures features, sau do train lai tren full dataset.

## Chay nhanh

python main.py download --symbol BTCUSDT --days 365 --output btc_30s.parquet

python main.py train --data btc_30s.parquet --symbol BTCUSDT --horizon 1 --threshold 0.0007 --confidence 0.60 --fee 0.0004 --tune-trials 25 --use-transformer --transformer-seq-len 120 --transformer-epochs 12 --ensemble-weight-xgb 0.55 --require-model-agreement --include-futures-features --use-gpu --model-output ensemble_btc_30s.joblib

python main.py predict --model ensemble_btc_30s.joblib --symbol BTCUSDT --lookback-minutes 360 --include-futures-features --include-depth-features --use-gpu

python main.py upgrade --symbol BTCUSDT --data btc_30s.parquet --model-output ensemble_btc_30s.joblib --days 365 --horizon 1 --threshold 0.0007 --confidence 0.60 --fee 0.0004 --tune-trials 25 --use-transformer --transformer-seq-len 120 --transformer-epochs 12 --ensemble-weight-xgb 0.55 --require-model-agreement --include-futures-features --use-gpu

python main.py live --model ensemble_btc_30s.joblib --symbol BTCUSDT --bootstrap-bars 1500

## Data cho train

- Toi thieu nen co 20,000 bars 30s (~7 ngay) de train baseline.
- Tot hon nen co >= 200,000 bars 30s (~70 ngay) de giam overfit.
- Moc 365 ngay se tao ~1,051,200 bars (30s), thuong ra file parquet khoang 40MB-180MB (tuy compression va kieu du lieu).
- Neu muc tieu la file ~10GB, ban can luu raw trade data (khong aggregate) hoac luu them rat nhieu symbol/timeframe.

## Ghi chu

- Upgrade KHONG xoa data cu: he thong append data moi vao file cu roi moi train lai.
- Accuracy trong trading khong quan trong bang profit sau fee + max drawdown + Sharpe.
- Khong co mo hinh nao dam bao "luon thang"; tuning cao hon se tang thoi gian train.
