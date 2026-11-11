# FinanceML

A cross-sectional, multi-asset machine learning framework for forecasting directional market regimes and price movements using **XGBoost** with incremental warm-starting.

## Overview

FinanceML avoids traditional single-stock overfitting by training sequentially across a curated basket of **40 high-liquidity, institutional assets** (`PRETRAIN_40`) spanning tech, consumer defensives, banking, energy, and global monopolies.

By leveraging **scale-free, stationary indicators**, the model learns universal laws of market dynamics that generalize across disparate equities and crypto assets without lookahead bias.

---

## Key Features & Architecture

* **Stationary Feature Engineering:**
  * **Robust Median Returns:** 5-day & 10-day rolling medians (immune to extreme one-off tail spikes).
  * **Volatility Regimes:** 5-day & 10-day rolling standard deviations of daily returns.
  * **Multi-Timeframe Normalized EMAs:** Percentage distances from 9, 21, 55, and 100-day EMAs:
    $$\text{dist\_ema} = \frac{\text{Close}}{\text{EMA}} - 1.0$$
  * **Trend Crossovers:** Fast (9 vs 21 EMA) and Slow (21 vs 55 EMA) momentum spreads.
  * **Relative Strength Index (RSI):** 14-period momentum oscillator.

* **Sequential Warm-Starting (XGBoost Booster Chaining):**
  * Rather than retraining from scratch on each asset, the model chains boosted decision trees using `xgb_model=booster`.
  * Preserves macro market memory while incrementally adapting to new sector dynamics.

* **Training vs. Inference Separation:**
  * Dedicated training phase for curated assets (`PRETRAIN_40`).
  * Read-only inference mode for ad-hoc user ticker testing to prevent data poisoning from illiquid or corrupted tickers.

---

## Curated Foundation Basket (`PRETRAIN_40`)

The base model pre-trains across 40 diversified assets:
1. **Mega-Cap Tech & Growth:** `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `META`, `TSLA`, `AVGO`, `NFLX`, `CRM`
2. **Defensive & Consumer Staples:** `JNJ`, `PG`, `KO`, `PEP`, `WMT`, `COST`, `MCD`, `UNH`, `BMY`, `CL`
3. **Financials & Banking:** `JPM`, `BAC`, `GS`, `MS`, `BLK`, `SCHW`, `C`, `AXP`, `SPG`, `AMT`
4. **Global Leaders & Moats:** `BRK-B`, `V`, `MA`, `LLY`, `TSM`, `ASML`, `HD`, `UNP`, `ABT`, `LIN`

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Model
```bash
python main.py
```

The script will:
1. Automatically run through the 40-stock pre-training bootcamp.
2. Report out-of-sample directional accuracy and feature importance rankings.
3. Launch an interactive console allowing you to evaluate any stock, ETF, or crypto ticker in real time.
