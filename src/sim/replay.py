"""CLI entrypoint: ``python -m sim.replay``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import Settings
from core.database import Database

from .replay_engine import HistoricalReplayEngine, resolve_replay_window

_LOG = logging.getLogger("tradingbot.sim.replay_cli")


def _filesystem_slug(text: str, *, max_len: int) -> str:
    """ASCII-ish folder segment: alnum, dot, hyphen, underscore; others become underscore."""

    out: list[str] = []
    for ch in (text or "").strip():
        if ch.isascii() and (ch.isalnum() or ch in "-_."):
            out.append(ch)
        elif ch in " ,;/\\|:\t\n":
            out.append("_")
        else:
            out.append("_")
    s = "".join(out).strip("._-") or "x"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-") or "x"
    return s


def build_default_replay_run_dirname(
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    mode: str,
    strategies: list[str],
    run_id: str,
) -> str:
    """Human-readable default folder name under ``LOG_DIR/replay_runs/``.

    Includes UTC date range, bar size, run mode, strategy keys, and a short slice of
    ``run_id`` so identical configs still get distinct directories when ``run_id`` differs.
    """

    su = start.astimezone(timezone.utc)
    eu = end.astimezone(timezone.utc)
    d0, d1 = su.strftime("%Y-%m-%d"), eu.strftime("%Y-%m-%d")
    if d0 == d1:
        date_part = f"{d0}_{su.strftime('%H%M')}to{eu.strftime('%H%M')}Z"
    else:
        date_part = f"{d0}_to_{d1}"

    tf = _filesystem_slug(timeframe, max_len=16)
    md = _filesystem_slug(mode, max_len=16)
    strat = "-".join(_filesystem_slug(s, max_len=36) for s in strategies)
    if len(strat) > 100:
        strat = strat[:97].rstrip("-") + "-etc"

    rid = "".join(c for c in run_id if c.isalnum())
    rid_short = (rid[:10] if len(rid) >= 10 else rid) or "run"

    parts = ["replay", date_part, tf, md, strat, rid_short]
    name = "__".join(parts)
    if len(name) > 200:
        name = name[:197].rstrip("._-") + "..."
    return name


def _allocate_unique_replay_dir(parent: Path, dirname: str) -> Path:
    """Return ``parent / dirname`` or add ``__2``, ``__3``, … if that path already exists."""

    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / dirname
    if not candidate.exists():
        return candidate
    stem = dirname
    for n in range(2, 10_000):
        alt = parent / f"{stem}__{n}"
        if not alt.exists():
            return alt
    return parent / f"{stem}__{uuid.uuid4().hex[:8]}"


def _parse_dt_utc(s: str) -> datetime:
    """Parse ISO-8601-ish string to aware UTC."""

    raw = (s or "").strip()
    if not raw:
        raise ValueError("empty datetime")
    if raw.lower() == "now":
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Historical replay (sim layer; no live orders).")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. AAPL,MSFT")
    p.add_argument("--strategies", required=True, help="Comma-separated strategy registry keys")
    p.add_argument("--start", default=None, help="Window start (ISO UTC). Mutually exclusive with --lookback-days.")
    p.add_argument(
        "--end",
        default="now",
        help='Window end (ISO UTC) or "now" for current UTC (default: now).',
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="If set, start = end - N days (calendar). Do not pass together with --start.",
    )
    p.add_argument("--timeframe", default="1Day", help="Bar timeframe (Alpaca-style), e.g. 1Day, 5Min")
    p.add_argument("--initial-equity", type=float, default=100_000.0, dest="initial_equity")
    p.add_argument("--mode", default="independent", choices=("independent", "ensemble", "both"))
    p.add_argument("--feed", default="iex", help="Alpaca data feed key (passed to backtester resolver)")
    p.add_argument("--run-id", default=None, dest="run_id", help="Replay run id (default: random uuid)")
    p.add_argument("--output-dir", default=None, dest="output_dir", help="CSV + summary directory (default: LOG_DIR/replay_runs/<descriptive name>)")
    p.add_argument("--database", default=None, help="SQLite path for replay tables (optional)")
    p.add_argument("--cache-dir", default=None, dest="cache_dir", help="Bar cache directory (default: LOG_DIR/replay_cache)")
    p.add_argument(
        "--drop-thin-align",
        action="store_true",
        help=(
            "If bar timestamps do not overlap across symbols (inner join would be empty), "
            "greedily drop symbols until the rest share enough common 1m/5m/etc. bars. "
            "Useful when mixing liquid ETFs with sparse or mis-listed tickers."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.start is not None and args.lookback_days is not None:
        print("error: pass only one of --start or --lookback-days", file=sys.stderr)
        return 2
    if args.start is None and args.lookback_days is None:
        print("error: pass --start or --lookback-days", file=sys.stderr)
        return 2

    end = _parse_dt_utc(args.end)
    if args.lookback_days is not None:
        start, end = resolve_replay_window(end=end, lookback_days=args.lookback_days)
    else:
        start = _parse_dt_utc(args.start)
        start, end = resolve_replay_window(end=end, start=start)

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    if not symbols or not strategies:
        print("error: --symbols and --strategies must list at least one item", file=sys.stderr)
        return 2

    settings = Settings()
    run_id = (args.run_id or "").strip() or str(uuid.uuid4())
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        replay_parent = Path(settings.LOG_DIR) / "replay_runs"
        slug = build_default_replay_run_dirname(
            start=start,
            end=end,
            timeframe=args.timeframe,
            mode=args.mode,
            strategies=strategies,
            run_id=run_id,
        )
        out = _allocate_unique_replay_dir(replay_parent, slug)
    cache = Path(args.cache_dir) if args.cache_dir else None

    db: Optional[Database] = None
    if args.database:
        db = Database(Path(args.database))
        # Fresh paths (e.g. ./runtime/replay.sqlite3) have no tables until initialized.
        db.init_schema()

    engine = HistoricalReplayEngine(
        settings,
        symbols=symbols,
        strategy_names=strategies,
        start=start,
        end=end,
        timeframe=args.timeframe,
        initial_equity=float(args.initial_equity),
        mode=args.mode,
        run_id=run_id,
        output_dir=out,
        database=db,
        data_feed=args.feed,
        cache_dir=cache,
        drop_thin_align=bool(args.drop_thin_align),
    )
    res = engine.run()
    _LOG.info(
        "event=replay_cli_done run_id=%s output=%s portfolios=%s",
        res.run_id,
        res.output_dir,
        ",".join(res.portfolios.keys()),
    )
    print(json.dumps({"run_id": res.run_id, "output_dir": str(res.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
