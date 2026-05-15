# Phase 3 Handoff — Trading Bot Improvements

> **Phase 9 note:** This document is **historical backlog** (Phase 3 QoL). For current
> operator runbooks — multi-strategy replay, shadow mode, ensemble paper, live
> confirmation gates — see **README.md** → *Research workflow & rollout*.

**Branch to base off:** `main` (Phase 1+2 already merged via `fix/phase1-phase2-critical-bugs`)  
**Repo:** `vardaan112/tradingbot`  
**Date written:** 2026-05-09

Phase 1+2 fixed critical bugs (look-ahead bias, ML label corruption, TrailState atomicity, regime startup gap, dedup). Phase 3 is quality-of-life, reliability, and alpha improvements. None are blocking for production — implement in whatever order fits available time.

---

## Item 3a — Move `_fmt_qty` out of orchestrator (15 min)

**File:** `src/services/orchestrator.py:118`  
**Problem:** `_fmt_qty` is a pure formatting utility living at module scope inside a 2339-line god-file. It is referenced in ~7 places in orchestrator only, but belongs in a shared utils module.

**What to do:**
1. Create `src/utils/price_utils.py` (new file) with:
   ```python
   def fmt_qty(qty: float | int) -> str:
       """Format a share quantity for log lines and audit strings."""
       return f"{qty:,.4f}".rstrip("0").rstrip(".")
   ```
2. In `src/services/orchestrator.py`: remove the `_fmt_qty` definition at line 118, add `from utils.price_utils import fmt_qty`, replace all 7 `_fmt_qty(` calls with `fmt_qty(`.
3. No behaviour change — pure refactor.

---

## Item 3b — asyncio fire-and-forget task tracking (30 min)

**File:** `src/services/orchestrator.py:600, 1109, 1120`  
**Problem:** Two `asyncio.create_task(...)` calls are untracked — if the coroutine raises, the exception is silently swallowed by the event loop. This hides autotune crashes.

**Affected lines:**
- Line 600: `self._discord_task = asyncio.create_task(...)` — already stored, OK.
- Line 1109: `asyncio.create_task(tune())` — fire-and-forget, untracked.
- Line 1120: `asyncio.create_task(retr())` — fire-and-forget, untracked.

**What to do:**

Add a task-set to the orchestrator `__init__`:
```python
self._background_tasks: set[asyncio.Task] = set()
```

Add a `_spawn` helper and `_on_task_done` callback to the class:
```python
def _spawn(self, coro, *, name: str) -> asyncio.Task:
    t = asyncio.create_task(coro, name=name)
    self._background_tasks.add(t)
    t.add_done_callback(self._background_tasks.discard)
    t.add_done_callback(self._on_task_done)
    return t

def _on_task_done(self, task: asyncio.Task) -> None:
    if not task.cancelled() and (exc := task.exception()):
        self._log.error("event=background_task_failed name=%s err=%r", task.get_name(), exc)
```

Replace at line 1109:
```python
# before:  asyncio.create_task(tune())
self._spawn(tune(), name="autotune")
```
Replace at line 1120:
```python
# before:  asyncio.create_task(retr())
self._spawn(retr(), name="autotune_retry")
```

---

## Item 3c — Narrow `contextlib.suppress(Exception)` (45 min)

**Files:** `src/services/orchestrator.py` (lines 405, 681, 737, 740, 743, 906, 1061, 2200) and `src/strategies/rsi_strategy.py:689`

**Problem:** `suppress(Exception)` is a blanket silence that eats real logic errors (`AttributeError`, `TypeError`) alongside expected I/O failures, making bugs invisible.

**What to do for each site** — read the block, identify what can actually raise, then narrow:

| File | Line | Likely safe to narrow to |
|------|------|--------------------------|
| orchestrator.py | 405 | `suppress(KeyError, AttributeError)` — reading dict/obj fields |
| orchestrator.py | 681 | `suppress(OSError, ValueError)` — file or metric write |
| orchestrator.py | 737–743 | `suppress(OSError)` — SQLite write |
| orchestrator.py | 906 | `suppress(OSError, ValueError)` — file write |
| orchestrator.py | 1061 | `suppress(OSError)` — SQLite write |
| orchestrator.py | 2200 | Read context — narrow to specific DB/IO exception |
| rsi_strategy.py | 689 | Read context — narrow to specific exception |

For each: replace `suppress(Exception)` with the narrowest correct type(s). Add a `self._log.debug(...)` inside the block if the exception is operationally interesting.

---

## Item 3d — Continuous regime-to-sizing curve (1 hr)

**Files:** `src/services/orchestrator.py:1472–1505`, `src/config/settings.py`

**Problem:** The regime multiplier is currently binary — either `1.0` (normal) or `1.0 - REGIME_MAX_EQUITY_REDUCTION` (bear_volatile). A mildly elevated ATR ratio triggers the same size cut as extreme volatility.

**Current code** (`orchestrator.py:1502`):
```python
regime_mult = max(
    0.0,
    1.0 - float(self._settings.REGIME_MAX_EQUITY_REDUCTION),
)
```

**What to do:**

1. Add two new settings to `src/config/settings.py`:
   ```python
   REGIME_ATR_RATIO_CLAMP_LOW: float = 1.0   # atr_ratio where mult = 1.0 (no cut)
   REGIME_ATR_RATIO_CLAMP_HIGH: float = 2.0  # atr_ratio where mult hits the floor
   ```

2. Replace the binary block at `orchestrator.py:1502` with a linear taper:
   ```python
   floor = max(0.0, 1.0 - float(self._settings.REGIME_MAX_EQUITY_REDUCTION))
   lo = float(self._settings.REGIME_ATR_RATIO_CLAMP_LOW)
   hi = float(self._settings.REGIME_ATR_RATIO_CLAMP_HIGH)
   if hi > lo:
       t = max(0.0, min(1.0, (rs.atr_ratio - lo) / (hi - lo)))
   else:
       t = 1.0
   regime_mult = 1.0 - t * (1.0 - floor)
   ```

3. The existing `REGIME_BEAR_VOLATILE_BLOCK_ENTRIES` hard-block at line 1476 stays unchanged — the taper only applies when block is disabled.

---

## Item 3e — Scale-in uses ATR-risk sizing (1.5 hr)

**Files:** `src/services/orchestrator.py:1530–1600`, `src/config/settings.py`

**Problem:** Scale-in add quantity reads `SCALE_IN_ADD_QTY` (a fixed share count). A large-ATR day gets the same share addition as a small-ATR day, so dollar-risk per scale-in leg is inconsistent.

**Current code** (`orchestrator.py:1532`):
```python
float(self._settings.SCALE_IN_ADD_QTY)
```

**What to do:**

1. Add settings to `src/config/settings.py`:
   ```python
   SCALE_IN_ATR_RISK_USD: float = 50.0   # target dollar-risk per scale-in leg
   SCALE_IN_MAX_ADD_QTY: int = 20        # hard cap on shares added
   ```

2. In the scale-in sizing block (around line 1552), after `scale_in_stop_distance` is computed, replace the fixed qty with:
   ```python
   if self._settings.SCALE_IN_ATR_RISK_USD > 0 and scale_in_stop_distance > 0:
       atr_sized_qty = self._settings.SCALE_IN_ATR_RISK_USD / scale_in_stop_distance
       add_qty = min(atr_sized_qty, float(self._settings.SCALE_IN_MAX_ADD_QTY))
   else:
       add_qty = float(self._settings.SCALE_IN_ADD_QTY)
   ```

3. `scale_in_stop_distance` is already computed nearby at lines 1550–1570 — wire the above directly after it.

---

## Item 3f — Walk-forward folds in autotune (2 hr)

**Files:** `src/services/orchestrator.py:1094–1125`, `backtest_simulation.py`, `src/config/settings.py`

**Problem:** Weekly autotune fits parameters on the full available history — in-sample overfitting. The chosen parameters have seen the test data. A single 80/20 walk-forward split would give a less biased estimate of live performance.

**What to do:**

1. Add setting to `src/config/settings.py`:
   ```python
   AUTOTUNE_WALK_FORWARD_FOLDS: int = 0   # 0 = disabled (legacy), 1 = single 80/20 split
   ```

2. In the autotune block (`orchestrator.py:1094`), after backtest results are collected per candidate, add an OOS scoring path when `AUTOTUNE_WALK_FORWARD_FOLDS > 0`:
   - Pass the full bars DataFrame to `BacktestSimulator.run()`
   - Split: `split_idx = int(len(bars) * 0.8)`; fit on `bars.iloc[:split_idx]`, score on `bars.iloc[split_idx:]`
   - Rank candidates by OOS score rather than full-history score

3. `BacktestSimulator.run()` already accepts a `bars` DataFrame — just pass the sliced subset.

4. Default `AUTOTUNE_WALK_FORWARD_FOLDS=0` so existing deployments are unaffected.

---

## Item 3g — Decompose `rsi_strategy.py` (2 hr)

**File:** `src/strategies/rsi_strategy.py` (2188 lines — well over the 800-line target)

**Problem:** One file contains signal generation, trailing stop management, signal logging, and indicator computation. Hard to read, test, or modify one part without touching the rest.

**Extract `_log_signal` into `src/strategies/signal_logger.py`:**

1. `_log_signal` is at `rsi_strategy.py:454`. It is a self-contained method that formats a structured dict and writes to SQLite + logs. No circular dependency on other strategy logic.

2. Create `src/strategies/signal_logger.py`:
   ```python
   class SignalLogger:
       def __init__(self, db_path: str, log: logging.Logger) -> None: ...
       def log_signal(self, symbol: str, signal_type: str, metadata: dict) -> None: ...
   ```

3. In `RSIStrategy.__init__`, construct `self._signal_logger = SignalLogger(db_path, self._log)` and replace all `self._log_signal(...)` calls (lines 1049, 1117, 1157, 1595, 2153) with `self._signal_logger.log_signal(...)`.

4. No behaviour change — pure structural extraction.

**Stretch goal** (only if the above is complete): extract `TrailManager` (the `_trails_by_symbol` dict, `_with_trail_update`, `_persist_trailing_to_disk`, and the trail-update logic) into `src/strategies/trail_manager.py`. This is a larger refactor — roughly 300 lines.

---

## Recommended order

| Priority | Item | Time | Reason |
|----------|------|------|--------|
| 1 | 3b asyncio task tracking | 30 min | Silently-failing autotune is a live risk |
| 2 | 3c narrow suppress | 45 min | Hidden bugs become visible immediately |
| 3 | 3d continuous regime curve | 1 hr | Direct alpha — smoother sizing |
| 4 | 3a `_fmt_qty` move | 15 min | Clean-up, trivial |
| 5 | 3e scale-in ATR sizing | 1.5 hr | Better risk-per-trade on add legs |
| 6 | 3f walk-forward autotune | 2 hr | Reduces overfitting — high value, more complex |
| 7 | 3g decompose rsi_strategy | 2 hr | Code hygiene, lowest urgency |

**Total estimate:** ~8 hrs for all items.

---

## Environment notes

- Python 3.11+, async (asyncio), Alpaca broker via `alpaca-py`
- No venv is checked in — install deps with `pip install -r requirements.txt`
- No test suite currently exists for the strategy layer — add `pytest` tests alongside any new code
- SQLite databases are in the project root (`trades.db`, `kelly_stats.db`, etc.)
- Settings are `pydantic-settings` loaded from `.env` — see `src/config/settings.py`
- All new dataclasses should be `frozen=True` (project convention from Phase 1)
- Use `self._log.warning(...)` (not `print`) for all diagnostic output
