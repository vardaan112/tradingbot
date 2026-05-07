#!/usr/bin/env python3
"""Replay ``reports/backtest_trades.csv`` rows into SQLite for Streamlit.

Does **not** call Alpaca or order APIs. Requires ``--confirm-simulation`` to write.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.constants import LOGGER_APP
from config.settings import Settings, get_settings  # noqa: E402
from core.database import Database  # noqa: E402

_LOG = logging.getLogger(LOGGER_APP)


def _parse_day_from_ts(ts_raw: str) -> str:
    try:
        dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()
    except Exception:
        return str(ts_raw)[:10]


def _group_rows_by_day(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        day = _parse_day_from_ts(str(r.get("entry_time") or r.get("original_entry_time") or ""))
        buckets[day].append(r)
    for k in buckets:
        buckets[k].sort(key=lambda x: (str(x.get("entry_time")), str(x.get("exit_time")), str(x.get("parameter_set_id", ""))))
    return dict(sorted(buckets.items()))


def pacing_delay_seconds_per_day(speed: float) -> float:
    return 1000.0 / max(float(speed), 1e-6)


def replay(
    *,
    rows: list[dict[str, Any]],
    db: Database,
    speed: float,
    dry_run: bool,
    replay_run_id_override: str | None = None,
) -> tuple[str, int]:
    """Insert simulated trades/events. Returns (replay_run_id, n_inserted)."""

    rid = replay_run_id_override or hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    spd = pacing_delay_seconds_per_day(speed)
    by_day = _group_rows_by_day(rows)
    n_ins = 0

    print("SIMULATION REPLAY MODE: no live orders will be sent.")  # noqa: T201
    if dry_run:
        print(f"[dry-run] would insert {len(rows)} trades into {db.path}")  # noqa: T201
        _LOG.info("event=replay_start dry_run=true trades=%s", len(rows))
        return rid, 0

    db.apply_migrations()
    _LOG.info("event=replay_start dry_run=false trades=%s speed=%s", len(rows), speed)

    for _day_key, day_rows in by_day.items():
        n_evt = len(day_rows)
        slot = spd / float(max(n_evt * 3, 1))
        t0 = time.perf_counter()

        for j, row in enumerate(day_rows):
            target_sleep = spd * float(j + 1) / float(n_evt + 1) - (time.perf_counter() - t0)
            if target_sleep > 0:
                time.sleep(target_sleep)

            sym = str(row["symbol"]).upper()
            qty = float(row["qty"])
            entry_px = float(row["entry_price"])
            exit_px = float(row["exit_price"])
            net = float(row["net_pnl"])
            entry_ts = str(row["entry_time"])
            exit_ts = str(row["exit_time"])
            param = str(row.get("parameter_set_id", ""))

            trade_key = f"sim_{rid}_{param}_{sym}_{hashlib.sha256(entry_ts.encode()).hexdigest()[:8]}"

            db.record_execution_event(
                event_type="simulated_entry",
                symbol=sym,
                side="long",
                client_order_id=trade_key,
                order_id=None,
                status="filled",
                price=entry_px,
                quantity=qty,
                metadata={"message": "simulation replay entry", "run_id": rid},
                source="simulation",
                replay_run_id=rid,
                created_at=None,
                simulated_timestamp=entry_ts,
            )
            _LOG.info("event=replay_insert_execution_event kind=entry symbol=%s", sym)

            if str(row.get("trailing_stop_active", "")).lower() in ("true", "1", "yes"):
                db.record_execution_event(
                    event_type="simulated_trailing_active",
                    symbol=sym,
                    side="long",
                    client_order_id=f"{trade_key}_trail",
                    order_id=None,
                    status=None,
                    price=exit_px,
                    quantity=qty,
                    metadata={"message": "trailing_stop_active=true"},
                    source="simulation",
                    replay_run_id=rid,
                    simulated_timestamp=exit_ts,
                )
                _LOG.info("event=replay_insert_execution_event kind=trail symbol=%s", sym)

            ret = net / max(qty * entry_px, 1e-9)
            md = {
                "source": "simulation",
                "replay_run_id": rid,
                "parameter_set_id": param,
                "gross_pnl": row.get("gross_pnl"),
                "fees": row.get("fees"),
                "slippage": row.get("slippage"),
                "rsi_entry": row.get("rsi_entry"),
                "adx_entry": row.get("adx_entry"),
                "trailing_stop_active": row.get("trailing_stop_active"),
                "max_favorable_excursion": row.get("max_favorable_excursion"),
                "max_adverse_excursion": row.get("max_adverse_excursion"),
                "bars_held": row.get("bars_held"),
                "exit_reason": row.get("exit_reason"),
            }

            inserted = datetime.now(timezone.utc).isoformat()
            db.record_completed_trade(
                trade_id=trade_key,
                symbol=sym,
                side="long",
                quantity=qty,
                entry_price=entry_px,
                exit_price=exit_px,
                realized_pnl=net,
                realized_return=float(ret),
                opened_at=entry_ts,
                closed_at=exit_ts,
                strategy_name="rsi_meanrev_sim",
                risk_mode="simulation",
                regime_type=str(row.get("regime_type") or "") or None,
                sentiment_score=None,
                sentiment_label=None,
                is_canary=0,
                metadata=md,
                source="simulation",
                replay_run_id=rid,
                inserted_at=inserted,
                original_entry_time=entry_ts,
                original_exit_time=exit_ts,
            )
            _LOG.info("event=replay_insert_trade symbol=%s pnl=%.4f", sym, net)
            n_ins += 1

            db.record_execution_event(
                event_type="simulated_exit",
                symbol=sym,
                side="long",
                client_order_id=f"{trade_key}_x",
                order_id=None,
                status="filled",
                price=exit_px,
                quantity=qty,
                metadata={
                    "message": str(row.get("exit_reason") or ""),
                    "run_id": rid,
                },
                source="simulation",
                replay_run_id=rid,
                simulated_timestamp=exit_ts,
            )
            _LOG.info("event=replay_insert_execution_event kind=exit symbol=%s", sym)

            time.sleep(slot)

    _LOG.info("event=replay_complete run_id=%s inserted=%d", rid, n_ins)
    return rid, n_ins


def write_replay_summary(
    path: Path,
    *,
    replay_run_id: str,
    n_trades: int,
    pnl_total: float,
    start_date: str,
    end_date: str,
    speed: float,
    db_path: Path,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Replay summary",
        "",
        f"- **replay_run_id:** `{replay_run_id}`",
        f"- **trades inserted:** {n_trades}",
        f"- **total simulated net PnL:** {pnl_total:.4f}",
        f"- **start date:** {start_date}",
        f"- **end date:** {end_date}",
        f"- **speed:** {speed}",
        f"- **database:** `{db_path}`",
        f"- **source tag:** simulation",
        f"- **mode:** {'dry-run' if dry_run else 'confirmed insert'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    pa = argparse.ArgumentParser(description="Replay simulated trades CSV into SQLite.")
    pa.add_argument("--trades-csv", type=str, default="reports/backtest_trades.csv")
    pa.add_argument("--database", type=str, default=None, help="Override Settings.DATABASE_PATH")
    pa.add_argument("--speed", type=float, default=100.0)
    pa.add_argument("--confirm-simulation", action="store_true")
    pa.add_argument("--summary-out", type=str, default="reports/replay_summary.md")
    pa.add_argument("--run-id", type=str, default=None)
    ns = pa.parse_args()

    root = Path.cwd()
    csv_path = Path(ns.trades_csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    try:
        settings: Settings | None = get_settings()
    except Exception:
        settings = None

    db_path_raw = ns.database or (str(settings.DATABASE_PATH).strip() if settings else "")
    if not db_path_raw.strip():
        print(
            "DATABASE_PATH missing (configure .env or pass --database)",
            file=sys.stderr,
        )  # noqa: T201
        sys.exit(2)
    db_p = Path(db_path_raw)
    if not db_p.is_absolute():
        db_p = (root / db_p).resolve()

    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    dry = not ns.confirm_simulation

    pnl_tot = float(df["net_pnl"].sum()) if "net_pnl" in df.columns else 0.0
    dmin = ""
    dmax = ""
    if rows:
        days = {_parse_day_from_ts(str(r.get("entry_time"))) for r in rows}
        sd = sorted(days)
        if sd:
            dmin, dmax = sd[0], sd[-1]

    summary_path = Path(ns.summary_out)
    if not summary_path.is_absolute():
        summary_path = root / summary_path

    rid, n_done = replay(
        rows=rows,
        db=Database(db_p),
        speed=float(ns.speed),
        dry_run=dry,
        replay_run_id_override=ns.run_id,
    )

    write_replay_summary(
        summary_path,
        replay_run_id=rid,
        n_trades=n_done,
        pnl_total=pnl_tot,
        start_date=dmin,
        end_date=dmax,
        speed=float(ns.speed),
        db_path=db_p,
        dry_run=dry,
    )
    print(f"Wrote {summary_path}")  # noqa: T201


if __name__ == "__main__":
    main()
