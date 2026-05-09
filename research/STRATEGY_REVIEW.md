# Strategy Review: RSI Mean-Reversion Bot
## Against Quantopian Research Literature

**Date:** 2026-05-09  
**Bot:** Long-only RSI mean-reversion on US equities (Alpaca)  
**Sources:** Quantopian lecture series — Mean Reversion, Mean Reversion on Futures, Kelly Criterion, Position Concentration Risk, Factor-Based Risk Management, Universe Selection, Fed Sentiment Volatility

---

## Executive Summary

The core strategy logic is empirically grounded. RSI thresholds, ATR position sizing, and the ADX/SMA regime filters are all validated by the literature. Two material bugs were identified — one in Kelly input data (silent P&L normalization error) and one in the high-volatility RSI threshold (direction inversion). The Bollinger + VWAP filter combination is likely overfit relative to the signal it adds. The QQQ regime gate does more risk-reduction work than all three technical filters combined.

| Finding | Severity | Category |
|---------|----------|----------|
| `HIGH_VOL_RSI_ENTRY=35` direction is inverted | Medium | RSI Thresholds |
| Kelly computed on raw dollar P&L, not % returns | High | Kelly Implementation |
| No realized correlation check between positions | Medium | ATR / Concentration |
| Bollinger + VWAP redundant with RSI | Low–Medium | Filter Stack / Overfitting |

---

## 1. RSI Thresholds

### What the Literature Says

The Quantopian mean reversion notebook frames the core signal as a z-score deviation: entry when price is ≥1σ below its rolling mean (z < −1), exit when price returns within ±0.5σ of mean. RSI is a bounded, momentum-normalized proxy for this same deviation. RSI(14) < 30 on daily bars corresponds to approximately 1.5–2σ below the recent 14-period mean in most US equity studies — placing it squarely inside the empirically tested entry zone.

> *"We expect [the price] to go up if it is unusually low... We can quantify lower than expected as the price having a z-score of less than −1."*  
> — Quantopian, Mean Reversion notebook

### Current Settings Assessment

| Setting | Default Value | Assessment |
|---------|--------------|------------|
| `DEFAULT_RSI_ENTRY` | 30 | Validated — corresponds to ~1.5–2σ oversold |
| `RSI_EXIT` | 50 | Exits at mean — conservative but correct |
| `DYNAMIC_RSI_MIN` | 20 | Aggressive; legitimate for extreme oversold |
| `DYNAMIC_RSI_MAX` | 35 | Appropriate upper bound for low-volatility regime |
| `HIGH_VOL_RSI_ENTRY` | **35** | **Inverted — see below** |

### Bug: `HIGH_VOL_RSI_ENTRY` Direction Is Inverted

**File:** `src/config/settings.py:79`

`HIGH_VOL_RSI_ENTRY = 35` makes entry *more permissive* (enter when RSI < 35 rather than < 30) in high-volatility environments. The apparent rationale is "high-vol stocks bounce faster, so enter sooner." However, the literature points the opposite direction: high ATR produces noisier oscillations and a higher rate of false oversold signals. Mean reversion requires the deviation to be a temporary fluctuation, not the start of a persistent trend — and high volatility increases the probability of the latter.

**Empirically supported direction:** higher volatility → *lower* entry threshold (deeper oversold required).

```
Current:  HIGH_VOL → RSI < 35  (more permissive) ← WRONG
Correct:  HIGH_VOL → RSI < 25  (more restrictive)
```

The dynamic RSI system (`DYNAMIC_RSI_ENABLED`) handles this correctly by scaling the threshold proportionally to the ATR ratio — lower threshold as ATR ratio rises. The static `HIGH_VOL_RSI_ENTRY` fallback path contradicts this and should be set to ~25, or removed in favour of always requiring `DYNAMIC_RSI_ENABLED=true`.

### RSI Exit at 50

Exiting at RSI = 50 (the midpoint) is sometimes criticised as leaving alpha on the table — practitioners argue for 55–65 to let winners run. In this bot, the trailing stop compensates: once profit exceeds `TRAIL_TRIGGER_PCT`, the trail activates and holds the position beyond RSI 50 as long as price keeps climbing. The combination of (RSI exit OR trailing stop) is sound.

---

## 2. Kelly Criterion Implementation

### What the Literature Says

The Quantopian Kelly notebook presents the continuous Kelly as an optimal leverage ratio:

```
K = (mean_excess_return) / variance
```

For per-trade binary outcomes, the equivalent discrete formula is:

```
f* = (b × p − q) / b
```

where `b = avg_win / avg_loss`, `p = win rate`, `q = 1 − p`. The notebook demonstrates that even for SPY over 13 years (Sharpe ~0.11), full Kelly implies only ~0.55× leverage — confirming that fractional Kelly (typically 25–50% of full) is the only safe practical application due to estimation error.

> *"The formula can be applied as a useful heuristic when deciding what percentage of your total capital should be allocated to a given strategy."*  
> — Quantopian, Kelly Criterion notebook

### Formula Correctness

The implementation in `src/risk/kelly_sizer.py` uses the correct discrete form:

```python
def kelly_fraction_from_trade_stats(*, avg_win, avg_loss, win_rate):
    b = avg_win / avg_loss
    return (b * p - q) / b   # f* = p - q/b
```

The fractional application is also correct. `KELLY_FRACTION=0.25` (25% of full Kelly) is the appropriate conservative posture — full Kelly produces catastrophic drawdowns with estimation error on small samples.

### Bug: Input Data Is Raw Dollar P&L, Not Percentage Returns

**File:** `src/risk/kelly_sizer.py` — `_kelly_stats_from_pnls()`

The Kelly formula requires a **consistent payoff ratio**. The bot pulls raw dollar P&L from the database. ATR-based sizing deliberately varies position size by stock volatility, so:

- Trade A: $200 profit on a $10,000 position = **2% return**
- Trade B: $200 profit on a $2,000 position = **10% return**

Both are averaged as "$200 wins." The resulting `avg_win / avg_loss` ratio is meaningless — it conflates completely different-sized bets. The Kelly fraction computed from this produces incorrect sizing multipliers on every live trade, with error growing as position size variance increases.

**Fix:** Store `pnl_pct = realized_pnl / (avg_entry_price × abs(filled_qty))` in the database alongside raw dollar P&L, and pass percentage returns to the Kelly calculation. The Quantopian notebook confirms this — it uses `daily_returns = (close − prev_close) / prev_close` as the input series, not raw price changes.

```python
# Store per trade:
pnl_pct = realized_pnl / (avg_entry_price * abs(filled_qty))

# Pass to Kelly:
pnls = [trade.pnl_pct for trade in recent_trades]
```

---

## 3. ATR Position Sizing

### What the Literature Says

The Quantopian volatility notebook defines volatility as the rolling standard deviation of log returns. ATR is a natural proxy: for normally distributed returns, ATR ≈ 1.25σ per bar. The position concentration notebook establishes the key theorem: portfolio volatility decays as 1/√N for N uncorrelated positions, but stays flat for N correlated positions.

> *"You can think of correlated bets as identical to the original bet. If the outcome of the second bet is correlated with the first, then really you have just made the same bet twice."*  
> — Quantopian, Position Concentration Risk notebook

### ATR Sizing Is Well-Aligned

```
shares = (equity × risk_pct × kelly_mult × regime_mult) / (ATR_STOP_MULTIPLIER × ATR)
```

This delivers constant expected dollar-risk per trade regardless of individual stock volatility — higher ATR → fewer shares. This is the textbook approach and matches the academic formulation directly.

### Gap: No Realized Correlation Check

The bot enforces per-sector position caps (`MAX_PER_SECTOR`) as a proxy for correlation control. This is insufficient:

1. Subcategory stocks within the same sector can be highly correlated (e.g. NVDA, AMD, SMCI all classified as "technology").
2. In a risk-off macro event, cross-sector correlations spike toward 1.0 — the diversification benefit disappears precisely when it is most needed.

This is the dominant source of tail drawdown for long-only mean-reversion portfolios. The QQQ regime gate addresses it at the macro level, but intra-portfolio pairwise correlation is not monitored at all.

**Practical fix (low effort):** Limit concurrent positions to one per stock-pair whose rolling 90-day pairwise correlation exceeds 0.7, computed weekly in a background task. This catches the most dangerous clusters without requiring a full covariance matrix.

---

## 4. Filter Stack: Edge or Overfitting?

### What the Literature Says

The core warning from the mean reversion notebook:

> *"The danger of applying mean reversion to a single stock is that it exposes us to movement of the market and the success or failure of the individual company. If there is a persistent trend affecting the price, we will find ourselves consistently undervaluing [it]."*

The futures mean reversion notebook extends this: only price series that pass a stationarity test (Augmented Dickey-Fuller, p < 0.05) are reliable candidates for mean reversion. Non-stationary — persistently trending — stocks produce systematic losses from an RSI oversold strategy regardless of parameter tuning.

### Filter-by-Filter Assessment

**ADX < 25 (`ADX_RANGE_MAX`)** — Validated  
ADX measures trend strength. ADX < 25 selects ranging markets where mean reversion holds. When ADX rises above 25, price is in a directional move and RSI < 30 is a falling-knife trap, not a reversion signal. This is the most academically important filter in the stack.

**200 SMA + Positive Slope** — Justified  
Long-only mean reversion works best in uptrends because institutional buy-the-dip flows create the reverting force. Below the 200 SMA, reversion dynamics are weaker and downside is asymmetric. The slope requirement prevents entering on a decelerating uptrend about to break down.

**QQQ Macro Regime Gate** — Most Valuable Filter in the Stack  
Blocks all entries when QQQ is in `bear_volatile` mode (ATR ratio elevated, price below SMA50). Does more protective work than any single-stock technical filter because in a risk-off event all stocks are affected simultaneously. This filter should never be disabled.

**Bollinger Band (`price < lower_band`)** — Likely Redundant  
RSI < 30 and price < Bollinger lower band both measure price deviation below a rolling mean. They are highly correlated — on most bars where RSI(14) < 30, price is also below the lower Bollinger band. Adding Bollinger as a required second condition cuts signal frequency without adding independent information.

*Statistical test:* Compute the empirical overlap rate — what percentage of RSI < 30 bars also satisfy the Bollinger condition. If > 80%, the filter is eliminating 0–20% of signals while providing zero incremental predictive value.

**VWAP Z-Score** — Marginal at Bar Level  
Rolling multi-bar VWAP provides some independent information about volume-weighted deviation that equal-weighted RSI does not capture. However, combined with RSI + Bollinger, the joint requirement fires rarely enough that any parameter combination will appear to backtest well simply by selecting only the most extreme historical episodes — a classic data-mining signature.

### The Overfitting Pattern

Each individual filter has academic support. The problem is the **joint requirement**. With five concurrent entry conditions (RSI < threshold AND ADX < 25 AND price > SMA200 AND price < BB lower AND VWAP z < −threshold), signal frequency is very low. When signals are rare, parameter choices can be tuned to select historically strong episodes, producing high backtest Sharpe that does not generalise out-of-sample.

**Interim recommendation:** Disable Bollinger and VWAP filters (`BOLLINGER_ENABLED=false`, `VWAP_STRATEGY_ENABLED=false`). Run live for 60 days with RSI + ADX + SMA + QQQ regime only. Measure signal count, win rate, and average holding-period return. Re-enable each filter only if out-of-sample win rate improvement is statistically significant (p < 0.05, N > 50 trades).

---

## 5. Long-Only RSI Mean Reversion: Literature Consensus

The single-stock mean reversion notebook confirms positive expected return from RSI < 30 in calm ranging markets, with these caveats directly applicable to this bot:

**What works:**
- RSI < 30 has positive expected return over 1–5 bar holding periods in uptrending large/mid-cap stocks
- Effect is strongest in low-to-moderate ADX environments (ADX 15–25)
- Long-only constraint is manageable with a macro regime gate substituting for market-neutral hedging

**What breaks it:**
- Persistent downtrends make RSI < 30 a continuation signal — addressed by SMA + ADX filters
- Market-wide risk-off events create correlated drawdowns across all long positions simultaneously — addressed by QQQ regime gate, not by intraday technical filters
- Win rate is structurally 55–65%, not 80%+ — Kelly sizing and position caps are the correct response, not more entry filters

**On the long-only constraint:**  
The Quantopian portfolio mean reversion notebook notes the strategy works best market-neutral (long losers, short winners) because this eliminates market-beta exposure. The bot's substitute — blocking entries in bear regimes and using tight ATR stops — is the right practical approach for a long-only retail implementation.

---

## 6. Recommendations

### High Priority

**Fix Kelly P&L normalization**  
Files: `src/risk/kelly_sizer.py`, `src/core/database.py`  
Store `pnl_pct = realized_pnl / (entry_price × qty)` per trade. Pass percentage returns to Kelly. Current raw-dollar inputs produce incorrect multipliers that worsen as position size variance increases with live trading.

### Medium Priority

**Fix `HIGH_VOL_RSI_ENTRY` direction**  
File: `src/config/settings.py:79`  
Change from 35 to ~25, or enforce `DYNAMIC_RSI_ENABLED=true` and remove the static fallback. The current value is directionally inverted relative to the literature.

**Add intra-portfolio correlation cap**  
Limit concurrent positions to one per stock-pair with rolling 90-day correlation > 0.7. Weekly batch computation is sufficient. Addresses the dominant tail-drawdown risk not covered by sector caps.

### Lower Priority

**Empirically test Bollinger + VWAP contribution**  
Disable both for 60 live trading days. Re-enable each only if out-of-sample win rate improvement is statistically significant (p < 0.05, N > 50 trades).

**Walk-forward validation in autotune** (Phase 3, Item 3f)  
Current autotune fits on full history. An 80/20 walk-forward split will reveal which filter parameters are genuinely predictive vs. curve-fit to historical data.

---

## Appendix: Notebooks Referenced

| File | Topic | Key Insight Applied |
|------|--------|---------------------|
| `research/quantopian/mean_reversion.ipynb` | Single-stock and portfolio mean reversion | z-score framework; RSI threshold calibration; persistent trend risk |
| `research/quantopian/mean_reversion_futures.ipynb` | Spread stationarity, cointegration | Stationarity required for mean reversion to hold |
| `research/quantopian/kelly_criterion.ipynb` | Kelly formula derivation | Continuous vs. discrete Kelly; % return inputs required |
| `research/quantopian/position_concentration.ipynb` | Portfolio diversification math | Correlated bets are not diversification; correlation cap rationale |
| `research/quantopian/risk_management.ipynb` | Factor-based risk decomposition | 99.8% of equally-weighted portfolio variance is common-factor risk |
| `research/quantopian/volatility.ipynb` | Rolling volatility measurement | ATR as σ proxy; volatility regime detection |
| `research/quantopian/universe_selection.ipynb` | Universe construction | Liquidity and trend filters for signal quality |
