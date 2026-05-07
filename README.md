# Alpaca Live Trading Bot

A production-oriented, single-process live-trading service for US equities on
Alpaca, written using `alpaca-py` only.

> **Status:** safe-to-live but defaults to dry-run mode. The first live
> deployment must run as a canary with `MAX_EQUITY_USAGE_USD=50` (the default).

GitHub repository: <https://github.com/vardaan112/tradingbot>

---

## Highlights

- `alpaca-py` only - no deprecated SDKs.
- Strict, fail-fast settings layer (pydantic) with explicit live-trading
  confirmation phrase.
- Structured rotating log files: `app.log`, `orders.log`, `risk.log`,
  `heartbeat.log`, `errors.log`.
- Latching kill switch (default 5% drawdown) that survives process restarts.
- ATR position sizing with per-trade risk cap, USD cap, exposure cap, and
  position-count cap.
- Emergency flatten via marketable limit IOC (no market orders).
- Idempotent client_order_id values and post-failure reconciliation.
- Websocket streaming for quotes and trade updates with supervised reconnect.
- FINRA Rule 4210 / Alpaca 2026-06-04 transition built in.

---

## Architecture

```
src/
  main.py                          entry point
  config/
    settings.py                    env-driven validated settings
    constants.py                   compile-time constants
    logging_config.py              rotating-file logging
  communication/
    discord_client.py            DiscordCommandCenter + slash (/status,/kill,/report)
  ml/
    signal_filter.py             sklearn RandomForest gate (fail-open)
  core/
    alpaca_clients.py              alpaca-py client wiring + feed detection
    market_data.py                 quote cache + historical bar fetcher
    trading_stream.py              supervised websocket runner
    orders.py                      limit-only orders + reconciliation
    account.py                     account/position adapters (PDT-tolerant)
    market_clock.py                market hours guard
    retries.py                     exp-backoff retries (429-aware)
    state_store.py                 atomic JSON state persistence
    exceptions.py                  bot-specific exception hierarchy
  risk/
    killswitch.py                latching kill switch
    kelly_sizer.py               fractional Kelly scaler (SQLite)
    position_sizer.py            ATR-based sizing + Kelly hook + clamps
    compliance.py                  PDT vs intraday-margin mode
    exposure.py                    gross/net exposure limits
  strategies/
    base.py                        Strategy base + Signal types
    indicators.py                  RSI, ATR (Wilder)
    universe.py                    price/volume/spread filters (+ skip log)
    rsi_strategy.py                canonical RSI mean reversion (long-only)
    rsi_mean_reversion.py          backward-compat re-export of rsi_strategy
  services/
    orchestrator.py                top-level event loop
    heartbeat.py                   60s heartbeat task
    canary.py                      one-time live canary check at startup
  utils/                           ids, math, price, time helpers (+ offline `backtester`)
runtime/                           persistent state (gitignored)
logs/                              rotating log files (gitignored)
tests/                             unit tests (run offline, no network)
```

---

## Installation

> Tested on Python 3.11+. Linux VPS (Ubuntu 22.04 / Debian 12) is the target.

```bash
git clone https://github.com/vardaan112/tradingbot.git
cd tradingbot

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# then edit .env with your Alpaca keys and settings
```

If you also want the dev/test extras:

```bash
pip install -e .[dev]
```

Core Phase 8 / ops packages may also be installed explicitly:

```bash
pip install discord.py scikit-learn xgboost psutil python-dotenv alpaca-py
```

---

## Pre-flight checklist

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `.env` exists and contains Alpaca keys, Discord token / `DISCORD_CHANNEL_ID` / `DISCORD_ALLOWED_USER_IDS` when Discord is enabled, and an explicit `DRY_RUN` setting.
3. Run offline tests: `pytest`
4. Start in dry-run first: `DRY_RUN=true python main.py`
5. Confirm Discord shows the startup banner (with clear `DRY_RUN` truth) and `SIMULATED FILL` notifications if you exercised entries in dry-run.
6. Only then, deliberately set `DRY_RUN=false` when you intend real orders.

Never run `DRY_RUN=false` unless you have verified Discord alerts (unless you explicitly disable/require them), canary behaviour, kill switch semantics, and dry-run simulated fill notifications.

---

## Environment configuration (.env)

The full annotated reference lives in `.env.example`. The most important keys:

| Key                              | Default        | Purpose |
|----------------------------------|----------------|---------|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | -        | Alpaca credentials |
| `ALPACA_ENV`                     | `paper`        | `paper` or `live` |
| `ALPACA_FEED`                    | `auto`         | `sip`, `iex`, or `auto` |
| `LIVE_TRADING_ENABLED`           | `false`        | Master gate to send orders |
| `DRY_RUN`                        | `true`         | Do not POST orders even if enabled |
| `CONFIRM_LIVE_TRADING`           | empty          | Must equal `yes_i_understand` to enable live trading on the live endpoint |
| `MAX_EQUITY_USAGE_USD`           | `50`           | Hard USD cap on bot-managed exposure |
| `MAX_RISK_PER_TRADE_PCT`         | `0.01`         | 1% of capital base per trade (hard ceiling) |
| `BOT_CAPITAL_BASE_USD`           | `0`            | Bot's allocated capital slice in USD. When >0, risk is computed against this instead of full account equity. 0 = fall back to `min(equity, MAX_EQUITY_USAGE_USD)` |
| `KILL_SWITCH_DRAWDOWN_PCT`       | `0.05`         | 5% intraday drawdown latches the switch |
| `SPREAD_FILTER_PCT`              | `0.0005`       | 5 bps max relative spread (SIP and default) |
| `SPREAD_FILTER_PCT_IEX`          | *(unset)*      | Optional wider cap for `feed=iex` quotes only (e.g. `0.0012` = 12 bps) |
| `QUOTE_STALENESS_SECONDS`        | `5`            | Reject signals on quotes older than this |
| `REGULATORY_MODE`                | `auto`         | `auto`, `pdt`, or `intraday_margin` |
| `POST_RULE4210_SCALING_ENABLED`  | `false`        | Required to relax legacy throttles after 2026-06-04 |
| `SYMBOLS`                        | `SPY,QQQ,IWM,XLF,EEM` | Static ETF basket for first live rollout |
| `RUN_LIVE_CANARY_ON_STARTUP`     | `false`        | When true on the live endpoint, run a one-time round-trip trade at startup before the main loop |
| `CANARY_SYMBOL`                  | `XLF`          | Symbol used for the canary trade |
| `CANARY_NOTIONAL_USD`            | `10`           | Dollar size of the canary trade |
| `CANARY_TIMEOUT_SECONDS`         | `60`           | Max seconds for the round trip |
| `CANARY_PERSIST_FILENAME`        | `canary_state.json` | Filename under `STATE_DIR` for canary success persistence |

**Never commit `.env`** - it is excluded by `.gitignore`.

---

## Strategy Time Machine & dashboard replay

Historical **grid search** and **path-dependent simulation** live in `src/utils/backtester.py`. Bars are fetched with **Alpaca market data only** (`StockHistoricalDataClient`, adjusted bars). The backtester **never** imports the trading REST client, never submits orders, and never invokes Canary, Kill Switch, reconciliation, or live execution helpers.

Caches live under **`runtime/cache/`**. When `pyarrow` or `fastparquet` is available, caches are stored as **Parquet**; otherwise CSV is written. Passing **`--refresh-cache`** deletes matching cache files and refetches before the sweep.

Outputs (defaults):

- `reports/backtest_results.csv` — one row per grid cell (Sharpe, max drawdown, profit factor, win rate, total return, etc.)
- `reports/backtest_trades.csv` — one row per completed simulated trade (`parameter_set_id`, entry/exit, stop/trail semantics, regime)
- `reports/backtest_summary.md` — narrative summary with ranked parameter sets

The optimizer sweeps **RSI** entry thresholds (25 / 30 / 35), **ADX** floors (20 / 25 / 30), and **ATR** stop / trail multiples (1.5 / 2 / 3) — 27 combinations per symbol — using the same regime / filter intent as `filters.py` where applicable.

### How to run the backtester

From the repo root (pytest uses `pythonpath = ["src"]`; the primary importable tree is `utils`, `core`, … under `src/`):

```bash
pip install -e .

# Preferred module path (shim under `src/src/utils/`):
python -m src.utils.backtester

# Equivalent:
python -m utils.backtester
backtest-bot --symbols SPY QQQ --start 2025-01-01 --end 2026-05-01 --timeframe 15Min
```

Useful flags: `--symbols`, `--start`, `--end` (UTC calendar dates), `--timeframe` (`1Min`, `5Min`, `15Min`, `1Hour`, `1Day`), `--initial-equity`, `--risk-pct`, **`--refresh-cache`** (ignore disk cache), `--use-cache` (default on) / `--no-cache`. Output paths: **`--output-results`**, **`--output-trades`**, **`--summary`**.

### Replaying trades into SQLite (dashboard)

`scripts/replay_simulator.py` reads **`reports/backtest_trades.csv`** (or `--trades-csv`), groups trades by calendar day, and inserts rows with **`source='simulation'`** plus replay metadata (`replay_run_id`, original timestamps). **Dry-run is the default** (prints counts only). **`--confirm-simulation`** performs the SQLite writes via `Database`; no Alpaca or order APIs are called.

Approximate pacing between calendar days follows **`seconds_per_day = 1000 / speed`** (`--speed`, default `100`). A Markdown summary is always written (`--summary-out`, default **`reports/replay_summary.md`**).

Example:

```bash
python scripts/replay_simulator.py --database runtime/tradingbot.sqlite --speed 500
python scripts/replay_simulator.py --database runtime/tradingbot.sqlite --speed 500 --confirm-simulation
```

In the Streamlit dashboard, use the **trade source** filter (**live**, **simulation**, or **all**) to blend or isolate replayed rows.

### Metrics (what they mean)

- **Total return** — Ending equity vs starting simulation notional (includes a
  simple spread / slippage model and optional per-side fee bps).
- **Sharpe ratio** — Mean / standard deviation of **daily** equity returns,
  annualized with \(\sqrt{252}\). Higher is better *in-sample*; it punishes
  volatile equity paths.
- **Max drawdown** — Worst peak-to-trough decline on **resampled daily**
  equity; more negative is worse. Use it next to Sharpe to spot
  high-return-but-brutal-tail configurations.
- **Win rate** — Fraction of completed trades with positive simulated PnL.
- **Profit factor** — Gross winning dollars divided by gross losing dollars;
  above 1 means gross wins exceed gross losses (capped at 9999 in grid CSV when
  there are no losing trades).

### Interpretation & risk

Strong backtest numbers **do not guarantee** live performance: partial fills,
borrow, halts, latency, fees, and regime changes are only approximated.

**Overfitting** is easy when grid-searching many thresholds on the same year of
data. Treat the optimizer as a hypothesis generator—validate hold-out periods,
different symbols, and forward paper trading before scaling
`BOT_CAPITAL_BASE_USD` or `MAX_EQUITY_USAGE_USD`.

Replayed dashboard rows are **synthetic audit trails** for visualization only; they are not live fills.

---

## Phase 8: Adaptive Brain & Remote Command Center

All Phase 8 features default to **disabled** in code and `.env.example` (`false`).
Enable deliberately and monitor risk.

### Discord command center (`ENABLE_DISCORD_BOT`)

- Primary implementation: **`src/communication/discord_client.py`** (`DiscordCommandCenter`) with optional **`discord.py`** `[project.optional-dependencies]` **dev**.
- **`src/services/discord_bot.py`** re-exports the same surface for backwards compatibility.
- Set **`DISCORD_BOT_TOKEN`**, **`DISCORD_CHANNEL_ID`**, and **`DISCORD_ALLOWED_USER_IDS`** (comma-separated Snowflake IDs). Slash commands **`/status`**, **`/kill`**, **`/report`**, and **`/skip SYMBOL`** are accepted only when invoked from **`DISCORD_CHANNEL_ID`** and by an allowed user.
- **`/kill`** invokes the orchestrator hook that latches **`KillSwitch`**, **`cancel_all`**, then **`submit_emergency_flatten`** — the same emergency path as internal kill handling. Remote kill emits **`event=discord_remote_kill`** in logs before flatten.
- Alerts (embeds): bot startup/shutdown, **ENTER_LONG** / **EXIT_LONG**, **BLACK_SWAN_TRIGGER**, **KILL_SWITCH_LATCHED**, **CANARY_FAILED** (one-shot Discord client during startup abort), **ML_TRADE_BLOCKED**, **WEBSOCKET_STALE**, **WEEKLY_AUTOTUNE_COMPLETE**, plus daily recap when **`DAILY_REPORT_ENABLED`** is on.
- **Security warning:** **`/kill` is destructive**. Anyone with your bot token or a compromised Discord account could flatten positions. Rotate tokens on leak; use **`DISCORD_ALLOWED_USER_IDS`**; treat Discord as alerting/ops, not the sole risk boundary.

### Walk-forward autotune (`ENABLE_AUTOTUNE`)

- Implemented in **`src/services/autotune.py`**, scheduled from the orchestrator: **Sunday**, Eastern clock, hour ≥ **`AUTOTUNE_SUNDAY_HOUR_ET`** (default **21**).
- Runs the existing backtest grid on the last **`AUTOTUNE_LOOKBACK_DAYS`** (default **30**) of **15-minute** Alpaca bars (same stack as `python -m src.utils.backtester`).
- Selects parameters with a balanced composite score (**Sharpe − 2×|max drawdown| + bonus when profit factor > 1.2**), requiring at least **`AUTOTUNE_MIN_TRADES_PER_CONFIG`** completed trades per candidate (default **10**).
- Rejects outright if **`|max_drawdown|`** exceeds **`AUTOTUNE_MAX_DRAWDOWN_ABS`** (fraction of equity curve; default **0.45**) before persistence.
- Writes **`src/config/dynamic_params.json`** (path **`DYNAMIC_PARAMS_PATH`**) with `source: "autotune"`, **`backtest_start`** / **`backtest_end`** (calendar dates aligned with walk-forward window), **`atr_multiplier`** (alias of **`atr_stop_multiplier`**), **`lookback_*`**, **`score`**, **`sharpe_ratio`**, metrics — **only** if validation passes and the composite score is **not worse** than the previous autotune score. Prior files are copied to **`runtime/param_backups/`** before replacement.
- With **`ENABLE_AUTOTUNE=false`**, the live strategy uses static **`Settings`** only. JSON overrides **never** replace secrets in **`.env`**.

### ML signal filter (`ENABLE_ML_FILTER`)

- Canonical module: **`src/ml/signal_filter.py`** (`MLSignalFilter`, **`RandomForestClassifier`**). **`src/strategies/ml_filter.py`** re-exports for legacy imports.
- Trains from SQLite **`completed_trades`** (live/paper; **excludes** `simulation`). Gates long entries at **`ML_FILTER_THRESHOLD`** (default **0.55**) via **`should_allow_trade`** / inference (`event=ml_filter_inference`, **`event=ml_trade_blocked`** when blocked, **`event=ml_filter_fail_open`** when allowing without a model or on errors).
- **Fail-open** by design: missing model, insufficient rows (**`MIN_ML_TRAINING_TRADES`**, default **50**), or inference errors **allow** the trade.
- Manual training: **`python -m src.strategies.ml_filter --train`** or **`python -m ml.signal_filter --train`** (both resolve the same entry point).
- **Overfitting warning:** in-sample accuracy does not guarantee live edge.

### Modified Kelly sizing (`ENABLE_KELLY_SIZING`)

- **`src/risk/kelly_sizer.py`** (`KellySizer`, **`RiskSizingDecision`**) applies **fractional** Kelly to the existing per-trade **`base_risk_pct`** stack ( **`MAX_RISK_PER_TRADE_PCT` × conviction × anti-martingale** ), using recent realized PnLs (**`KELLY_LOOKBACK_TRADES`**, **`KELLY_MIN_TRADES`**, **`KELLY_FRACTION`**, **`KELLY_MAX_RISK_MULTIPLIER`** / **`KELLY_MIN_RISK_MULTIPLIER`**). Wired from **`PositionSizer`** with structured **`event=kelly_sizing`** lines.
- Thin history / non-finite Kelly **falls back** to baseline risk — behaviour remains **limit-order-only** downstream.

### Overfitting & capital

Autotune and ML both learn from **history**. Performance in live markets can differ sharply. Raise **`BOT_CAPITAL_BASE_USD`** only after sustained monitoring.

**Reminder:** Entries and exits use **limit logic only** — including **`submit_emergency_flatten`** (**marketable limit IOC**, not raw market orders per project policy).

---

## Running

### Dry-run (default, recommended first)

```bash
python src/main.py
```

This evaluates the full pipeline but never POSTs an order. Logs are written to
`logs/`, runtime state to `runtime/`. Tail the heartbeat:

```bash
tail -f logs/heartbeat.log
```

### Paper live

```bash
# .env:
ALPACA_ENV=paper
LIVE_TRADING_ENABLED=true
DRY_RUN=false

python src/main.py
```

### Live canary (real money)

You must explicitly opt in:

```bash
# .env:
ALPACA_ENV=live
LIVE_TRADING_ENABLED=true
DRY_RUN=false
CONFIRM_LIVE_TRADING=yes_i_understand
MAX_EQUITY_USAGE_USD=50            # keep tiny for the first deployment
BOT_CAPITAL_BASE_USD=200           # the slice this bot is allowed to risk against
RUN_LIVE_CANARY_ON_STARTUP=true    # run the one-time startup verification trade
CANARY_SYMBOL=XLF                  # liquid, low-priced ETF for the canary
CANARY_NOTIONAL_USD=50             # >= ~one share price of the canary symbol
```

When `RUN_LIVE_CANARY_ON_STARTUP=true` is combined with the live endpoint
(and the bot is *not* in dry-run), `main.py` invokes `canary_check(settings)`
**before** the main loop. The canary:

- Verifies credentials, market open, fresh in-spec quote, no kill-switch latch,
  no existing position/order in the canary symbol, and broker compliance.
- Submits a conservative DAY limit BUY for the smallest qty that fits
  `CANARY_NOTIONAL_USD` (whole-share when `CANARY_NOTIONAL_USD` >= one share
  price; fractional only if the asset is fractionable).
- Waits up to `CANARY_TIMEOUT_SECONDS` for fill, then submits a marketable
  limit IOC SELL at exactly the filled qty.
- Persists success to `runtime/canary_state.json` so it does not re-run
  again the same trading day.
- Aborts startup on any failure (no fallback to market orders, ever).

If your `CANARY_NOTIONAL_USD` is below one share price of the canary symbol,
either raise it above a single share price or pick a fractionable canary
symbol; the bot will refuse to fall back to a market order.

Then run on the VPS, ideally under `systemd` or `supervisord`. A minimal
systemd unit:

```ini
[Unit]
Description=Alpaca Trading Bot
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/tradingbot
EnvironmentFile=/opt/tradingbot/.env
ExecStart=/opt/tradingbot/.venv/bin/python /opt/tradingbot/src/main.py
Restart=on-failure
RestartSec=15
User=tradingbot
Group=tradingbot

[Install]
WantedBy=multi-user.target
```

---

## Canary deployment procedure

1. Provision the VPS, install Python 3.11+, clone the repo, install deps.
2. Generate a **dedicated paper API key** and run a full session in `DRY_RUN=true` mode.
3. Verify all five log files are populated, heartbeat is steady, and no
   `errors.log` entries appear.
4. Switch to paper live (`DRY_RUN=false`, paper endpoint) for at least one
   full session. Confirm orders and exits behave as expected.
5. Switch endpoint to live with `MAX_EQUITY_USAGE_USD=50`,
   `BOT_CAPITAL_BASE_USD=<your slice>`, and
   `RUN_LIVE_CANARY_ON_STARTUP=true`. The bot will run the canary
   automatically on first startup.
6. Tail `logs/orders.log` and grep for `event=canary_*` to confirm
   the round-trip flatten succeeded. The bot enters the main loop only
   after `event=canary_complete`.
7. Review `risk.log` and `orders.log`. Only then incrementally raise
   `MAX_EQUITY_USAGE_USD` and/or `BOT_CAPITAL_BASE_USD`.
8. After 2026-06-04, do **not** flip `POST_RULE4210_SCALING_ENABLED=true`
   until you have observed at least one full session in
   `intraday_margin` mode and reviewed account behavior end-to-end.

---

## FINRA Rule 4210 / 2026-06-04 transition

Effective 2026-06-04, FINRA Rule 4210 amendments restructure intraday margin
treatment. Alpaca has stated:

- The new logic activates in their Trading API on **2026-06-04**.
- The legacy PDT-related fields are being phased out:
  - `pattern_day_trader`
  - `daytrade_count`
  - `daytrading_buying_power`

This bot:

- **Treats those fields as deprecated, optional metadata only.** The account
  parser tolerates their absence.
- Resolves regulatory mode automatically:
  - Before 2026-06-04 -> PDT-conservative behavior.
  - On/after 2026-06-04 -> buying-power-centric (`buying_power` only).
- **Never** uses `daytrading_buying_power` in `intraday_margin` mode.
- Requires `POST_RULE4210_SCALING_ENABLED=true` before any post-rule
  trading-frequency relaxation actually takes effect.

You can override this with `REGULATORY_MODE=pdt` or
`REGULATORY_MODE=intraday_margin` to lock a single mode for testing.

---

## Log files

| File                      | What lives here |
|---------------------------|-----------------|
| `logs/app.log`            | Boot, lifecycle, strategy decisions, market data |
| `logs/orders.log`         | Every order submission, cancel, fill, reconciliation |
| `logs/risk.log`           | Sizing, kill switch, exposure, compliance |
| `logs/heartbeat.log`      | One line every 60 s: session, ws health, equity, etc. |
| `logs/errors.log`         | All WARNING+ from any logger |

Each line includes context: `mode=<env/dry|live>`, `reg=<mode>`, `symbol`,
`strategy`, `client_order_id` when applicable.

---

## Troubleshooting

### HTTP 429 (rate limited)
The retry layer honors `Retry-After`-style hints when available and uses
exponential backoff with jitter otherwise. Investigate by tailing
`errors.log` and checking how many simultaneous symbols you have configured.
Reduce `SYMBOLS` or extend `RETRY_MAX_DELAY_SECONDS`.

### Stale quotes
- Confirm `ALPACA_FEED` resolution by grepping `app.log` for "feed resolved".
  IEX-only accounts run the spread filter in degraded confidence.
- Check `latest_quote_age` in `heartbeat.log`; sustained values above
  `QUOTE_STALENESS_SECONDS` indicate a websocket problem.

### Websocket disconnects
- Search `app.log` for "stream crashed" or "Reconnecting".
- The bot auto-reconnects with exponential backoff; signal generation is
  paused while either stream is unhealthy.
- After reconnect the bot reconciles open orders via REST.

### Rejected orders
- See `orders.log` for the full request and broker response.
- Reconciliation by `client_order_id` runs automatically on ambiguous
  failures; only orders confirmed not to exist are re-submitted.

### Kill switch latched
- Inspect `runtime/kill_switch_state.json` for the trigger reason.
- To reset, call `KillSwitch.reset(force=True, operator_token="<>=6 chars>")`
  from a Python session OR delete the file (manual operation).
- The bot will not auto-clear the latch under any circumstances.

---

## Development

Install dev extras and run tests offline:

```bash
pip install -e .[dev]
pytest
```

The test suite never touches the Alpaca API.

---

## Local laptop stress testing

These tools help soak-test the bot on a laptop **before** moving to a VPS. Long-running
production is still safer on stable power and networking; laptops sleep, roam Wi‑Fi, and lose
battery.

### Crash recovery (`state_recovery`)

The bot reconciles Alpaca-held longs into the persisted bot ledger after restarts via
``reconcile_open_positions`` / ``event=state_recovery`` (trail/stop hooks via
``adopt_long_position``).

```bash
pytest tests/stress_test_recovery.py -q
```

### Flash crash detector (offline)

Demonstrates ~10% SPY drop inside the 15-minute window using mocks only:

```bash
python scripts/mock_flash_crash.py
```

### Websocket staleness alerts

While running, heartbeat evaluates stream health each interval. If both trading and market
sockets appear up but quotes/events are stale for more than ``STREAM_STALE_SECONDS`` (default 30),
or sockets are disconnected, the bot emits ``event=websocket_health`` and, when
``ENABLE_LOCAL_NOTIFICATIONS=true`` (default locally), fires a desktop notification with a
300-second cooldown via ``STREAM_NOTIFICATION_COOLDOWN_SECONDS``. Notification stacks: prefer
``plyer`` if installed; else macOS ``osascript``, Windows beep/MessageBox, or ``notify-send``.

```bash
pytest tests/test_stream_alerts.py -q
```

### Battery and resource warnings

Startup calls ``log_startup_local_health``. Each heartbeat logs ``event=local_resource_check``
with CPU/memory/disk and best-effort battery status. Tune with:

| Setting | Meaning |
|---------|---------|
| ``WARN_ON_LOW_BATTERY`` | Log warnings when below threshold |
| ``LOW_BATTERY_THRESHOLD_PCT`` | Default 20 |
| ``REQUIRE_POWER_FOR_LOCAL_LIVE`` | When ``true``, extra warnings if unplugged/low while live orders enabled |

Trading is **not** auto-stopped on low battery unless you add operational policy elsewhere.

```bash
pytest tests/test_local_health.py -q
```

---

## Trading Bot Command Center

A dark-themed **read-only Streamlit dashboard** (`src/utils/dashboard.py`) for equities: live
Alpaca balances/positions (**no order/mutation endpoints**), SQLite realized P&amp;L, kill-switch
JSON, and a bounded tail of `logs/app.log`.

**Install** (see `requirements.txt`):

```bash
pip install streamlit-autorefresh
pip install streamlit plotly pandas
```

Or install full project deps:

```bash
pip install "streamlit>=1.37,<2" "plotly>=5.18,<7" "streamlit-autorefresh>=1,<2"
```

**Run** from the repository root (reuse the same `.env` as the bot):

```bash
streamlit run src/utils/dashboard.py
```

The app loads **Alpaca keys, env, `DATABASE_PATH`, `LOG_DIR`, and `STATE_DIR`** from
`get_settings()`. Secrets are not printed in the UI. **`streamlit-autorefresh`** drives
automatic reruns at the sidebar **Watchlist refresh interval** (default **30s**); without it,
use **Refresh now**.

Treat the dashboard like **local-only** or **VPN/tunnel** tooling: do not expose Streamlit on
the public internet.

### What it shows

- **Live Watchlist** (cached ~30s): default basket **SPY, QQQ, IWM, XLF, EEM** from `SYMBOLS`
  plus **latest 5m bar close**, configurable-period **RSI** (`strategies.indicators.rsi`),
  freshness vs last bar age, optional **spread %** from latest quote (read-only market data).

- **Alpaca** (cached ~10s): equity, buying power, cash, open positions, open unrealized P/L;
  cumulative P&amp;L and win rate come from SQLite completed trades unless the chart uses API.
- **SQLite** (cached ~3s): schema-tolerant **`completed_trades`** (preferred) or **`trades`** /
  **`executions`**; cumulative realized P&amp;L, profit factor, daily bars, recent 25 rows.
- **Kill switch**: `STATE_DIR/kill_switch_state.json` (`latched` / `is_latched` / `kill_switch_latched`).
- **Logs**: last 50 lines of `LOG_DIR/app.log` (bounded read).
- **Live warning badge** when `ALPACA_ENV=live` and `DRY_RUN=false`.

Replayed SQLite rows (**simulation**) can be blended via the sidebar scope filter (**live** /
**simulation** / **all**) for daily realized totals sourced from `completed_trades`.

If a data source fails, that section degrades gracefully while the rest keeps working.

---

## Security posture

- No money-movement / ACH / transfer logic exists in the codebase.
- Secrets are loaded only from `.env` via the settings layer.
- Secrets are never logged, never embedded in exception messages, and never
  printed back to stdout.
- `.env`, `logs/`, and `runtime/` are excluded by `.gitignore`.

---

## License

Proprietary - all rights reserved.
