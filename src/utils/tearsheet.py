"""Lightweight intra-day tearsheet summarizer for ``orders.log``.

Parses streamed trade-update lines emitted by ``OrderService`` and reconciles
FIFO lots to approximate realized intraday performance. Intended for heartbeat
telemetry—not a brokerage TCA report.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional, Sequence
from zoneinfo import ZoneInfo

from utils.time_utils import today_eastern

_NY = ZoneInfo("America/New_York")

_TRADE_PATTERN = re.compile(
    r"Trade update\s+coid=(?P<coid>\S+)\s+symbol=(?P<sym>\S+)\s+"
    r"side=(?P<side>\w+)\s+status=(?P<status>\S+)\s+"
    r"filled=(?P<filled>[\d.]+)\s+avg=(?P<avg>[\d.]+)",
    re.IGNORECASE,
)
_META_STRATEGY_PATTERN = re.compile(r"\|\s+strategy=(?P<v>\S+)\s")


@dataclass
class _Lot:
    qty: float
    price: float


def _extract_strategy(line: str) -> Optional[str]:
    m = _META_STRATEGY_PATTERN.search(line)
    if not m:
        return None
    val = m.group("v").strip()
    return None if val in {"-", ""} else val


def _eastern_calendar_date(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_NY).strftime("%Y-%m-%d")


def _parse_leading_datetime(line: str) -> datetime | None:
    try:
        head = line.split("|", 1)[0].strip()
        naive = datetime.fromisoformat(head)
        if naive.tzinfo is None:
            naive = naive.replace(tzinfo=timezone.utc)
        return naive
    except (ValueError, IndexError):
        return None


def _parse_trade_line(line: str) -> Optional[dict[str, Any]]:
    m = _TRADE_PATTERN.search(line)
    if not m:
        return None
    g = m.groupdict()
    return {
        "coid": str(g["coid"]),
        "symbol": str(g["sym"]).upper(),
        "side": str(g["side"]).lower(),
        "status": str(g["status"]).lower(),
        "filled": float(g["filled"]),
        "avg": float(g["avg"]),
    }


def summarize_piece_pnls(pnls: list[float]) -> dict[str, Any]:
    """Sharpe / PF / MDD from an ordered list of realized P&L pieces or whole trades."""

    if not pnls:
        return {
            "ok": True,
            "closed_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
            "profit_factor": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "win_rate_pct": None,
            "reason": "empty",
        }

    gross_profit = sum(x for x in pnls if x > 0.0)
    gross_loss = sum(-x for x in pnls if x < 0.0)
    net_pnl = float(sum(pnls))

    if gross_loss <= 1e-12:
        pf: Optional[float] = float("inf") if gross_profit > 1e-12 else None
    else:
        pf = gross_profit / gross_loss if gross_loss > 1e-12 else None

    sharpe_ratio: Optional[float]
    if len(pnls) < 2:
        sharpe_ratio = None
    else:
        n = len(pnls)
        mu = net_pnl / n
        var = sum((p - mu) ** 2 for p in pnls) / max(n - 1, 1)
        sigma = math.sqrt(var)
        sharpe_ratio = (mu / sigma) * math.sqrt(n) if sigma > 1e-12 else None

    run = 0.0
    peak = 0.0
    worst_dd = 0.0
    for piece in pnls:
        run += piece
        peak = max(peak, run)
        worst_dd = min(worst_dd, run - peak)
    max_drawdown = abs(float(worst_dd)) if worst_dd < 0 else 0.0

    decisive = [p for p in pnls if abs(p) > 1e-12]
    wins_ct = sum(1 for p in decisive if p > 0)
    win_rate_pct = (100.0 * wins_ct / len(decisive)) if decisive else None

    return {
        "ok": True,
        "closed_trades": len(pnls),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net_pnl,
        "profit_factor": pf,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate_pct": win_rate_pct,
        "reason": "ok",
    }


def build_summary_from_db_rows(rows: Sequence[Any]) -> dict[str, Any]:
    """Row-like objects exposing ``realized_pnl`` (one aggregate per closed trade)."""

    pnls = [float(r["realized_pnl"]) for r in rows if r["realized_pnl"] is not None]
    return summarize_piece_pnls(pnls)


def tearsheet_primary(
    settings: Any,
    *,
    db: Any | None,
    orders_log_path: Path,
) -> dict[str, Any]:
    """Prefer SQLite for today's ET ledger when TEARSHEET_PRIMARY=sqlite."""

    if getattr(settings, "TEARSHEET_PRIMARY", "orders_log") == "sqlite":
        try:
            if db is not None:
                day = today_eastern().strftime("%Y-%m-%d")
                rows = db.get_completed_trades_for_calendar_day_et(
                    trading_day_yyyy_mm_dd=day,
                    exclude_canary=True,
                )
                pnls = []
                for r in rows:
                    rv = r["realized_pnl"]
                    if rv is not None:
                        pnls.append(float(rv))
                summary = summarize_piece_pnls(pnls)
                summary["primary"] = "sqlite"
                return summary
        except Exception:
            pass
    return get_tearsheet_summary(orders_log_path)


def get_tearsheet_summary(log_path: Path) -> dict[str, Any]:
    """Summarize today's strategy trades from ``orders.log``.

    Fail-closed into a dict with ``ok=False`` rather than throwing.
    """
    blank: dict[str, Any] = {
        "ok": False,
        "closed_trades": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "net_pnl": 0.0,
        "profit_factor": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "win_rate_pct": None,
        "reason": "",
    }

    path = Path(log_path)
    if not path.exists():
        blank["reason"] = "missing_log"
        return blank

    target_day = today_eastern().strftime("%Y-%m-%d")

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        blank["reason"] = "read_failure"
        return blank

    fifo: dict[str, Deque[_Lot]] = defaultdict(deque)
    last_qty_acc: dict[str, float] = defaultdict(float)  # coid keyed
    pnls: list[float] = []

    for raw in lines:
        strat_raw = (_extract_strategy(raw) or "").lower()
        if "canary" in strat_raw:
            continue

        stamp = _parse_leading_datetime(raw)
        if stamp is None:
            continue
        if _eastern_calendar_date(stamp) != target_day:
            continue

        evt = _parse_trade_line(raw)
        if evt is None:
            continue
        if evt["status"] not in {"filled", "partially_filled"}:
            continue
        if evt["avg"] <= 0:
            continue

        sym = evt["symbol"]
        ckey = evt["coid"]

        accrued = float(last_qty_acc[ckey])
        cur = float(evt["filled"])
        inc = cur - accrued
        if inc <= 1e-12:
            continue
        last_qty_acc[ckey] = cur

        qty = float(inc)
        price = float(evt["avg"])
        side = evt["side"]

        dq = fifo[sym]

        if side.startswith("b"):
            dq.append(_Lot(qty=qty, price=price))
        elif side.startswith("s"):
            rem = qty
            while rem > 1e-12 and dq:
                lot = dq[0]
                take = float(min(lot.qty, rem))
                pnl_piece = take * (price - lot.price)
                pnls.append(pnl_piece)
                lot.qty -= take
                rem -= take
                if lot.qty <= 1e-12:
                    dq.popleft()

    out = summarize_piece_pnls(pnls)
    out["reason"] = "orders_log"
    return out


def format_tearsheet_markdown_table(summary: dict[str, Any]) -> str:
    """Return a Markdown table suitable for heartbeat logs."""

    def _flt(v: object, fmt: str) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            if v != v:
                return "n/a"
            if v == float("inf"):
                return "inf"
            return fmt.format(v)
        if isinstance(v, int):
            return str(v)
        return str(v)

    if not isinstance(summary, dict):
        return ""
    rows = (
        "| Metric | Value |\n| --- | --- |\n"
        + f"| closed_trades | {summary.get('closed_trades', 'n/a')} |\n"
        + f"| net_pnl | {_flt(summary.get('net_pnl'), '{:.4f}')} |\n"
        + f"| gross_profit | {_flt(summary.get('gross_profit'), '{:.4f}')} |\n"
        + f"| gross_loss | {_flt(summary.get('gross_loss'), '{:.4f}')} |\n"
        + f"| profit_factor | {_flt(summary.get('profit_factor'), '{:.4f}')} |\n"
        + f"| sharpe_ratio | {_flt(summary.get('sharpe_ratio'), '{:.4f}')} |\n"
        + f"| max_drawdown | {_flt(summary.get('max_drawdown'), '{:.4f}')} |\n"
        + f"| win_rate_pct | {_flt(summary.get('win_rate_pct'), '{:.2f}')} |\n"
    )
    return rows.strip()


__all__ = [
    "build_summary_from_db_rows",
    "format_tearsheet_markdown_table",
    "get_tearsheet_summary",
    "summarize_piece_pnls",
    "tearsheet_primary",
]
