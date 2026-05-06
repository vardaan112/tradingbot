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
    killswitch.py                  latching kill switch
    position_sizer.py              ATR-based sizing with hard clamps
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
  utils/                           ids, math, price, time helpers
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
| `SPREAD_FILTER_PCT`              | `0.0005`       | 5 bps max relative spread |
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

## Security posture

- No money-movement / ACH / transfer logic exists in the codebase.
- Secrets are loaded only from `.env` via the settings layer.
- Secrets are never logged, never embedded in exception messages, and never
  printed back to stdout.
- `.env`, `logs/`, and `runtime/` are excluded by `.gitignore`.

---

## License

Proprietary - all rights reserved.
