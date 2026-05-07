"""Markdown daily brief (idempotent per Eastern calendar date)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from config.constants import LOGGER_APP
from config.settings import Settings
from core.database import Database
from utils.tearsheet import build_summary_from_db_rows
from utils.time_utils import today_eastern


_LOG = logging.getLogger(LOGGER_APP)


def sentiment_accuracy_pct(rows: list) -> tuple[str, int, int]:
    """Simple accuracy: bullish labels + wins, bearish labels + losses."""

    corr = elig = 0
    bullish = frozenset({"positive", "neutral"})
    bearish = frozenset({"negative", "strong_negative"})
    for r in rows:
        lbl = (getattr(r, "sentiment_label", "") or "").strip().lower()
        pnl_raw = getattr(r, "realized_pnl", None)
        if lbl in {"", "-"} or pnl_raw is None:
            continue
        pnl = float(pnl_raw)
        elig += 1
        ok = False
        if lbl in bullish and pnl > 1e-9:
            ok = True
        elif lbl in bearish and pnl < -1e-9:
            ok = True
        if ok:
            corr += 1
    if elig == 0:
        return ("n/a", 0, 0)
    return (f"{100.0 * corr / elig:.2f}%", corr, elig)


def generate_daily_report(
    settings: Settings,
    db: Database,
    *,
    trading_day: Optional[date] = None,
) -> Optional[Path]:
    """Write ``REPORTS_DIR/YYYY-MM-DD.md``."""

    if not settings.DAILY_REPORT_ENABLED:
        return None
    d = trading_day or today_eastern()
    yyyy_mm_dd = d.isoformat()

    trades = db.get_completed_trades_for_calendar_day_et(
        trading_day_yyyy_mm_dd=yyyy_mm_dd,
        exclude_canary=True,
    )
    summary = build_summary_from_db_rows(trades)

    best_sym_txt = "n/a"
    worst_sym_txt = "n/a"
    best_amt = float("-inf")
    worst_amt = float("inf")
    for r in trades:
        rv = getattr(r, "realized_pnl", None)
        if rv is None:
            continue
        pnl_f = float(rv)
        sym = getattr(r, "symbol", "?")
        if pnl_f > best_amt:
            best_amt = pnl_f
            best_sym_txt = f"{sym} ${pnl_f:.4f}"
        if pnl_f < worst_amt:
            worst_amt = pnl_f
            worst_sym_txt = f"{sym} ${pnl_f:.4f}"
    acc_pct, corr_n, elig_n = sentiment_accuracy_pct(trades)

    chase_att = db.count_execution_events(
        event_type="order_chase_attempt",
        trading_day_yyyy_mm_dd=yyyy_mm_dd,
    )
    chase_gup = db.count_execution_events(
        event_type="order_chase_giveup",
        trading_day_yyyy_mm_dd=yyyy_mm_dd,
    )
    skips = db.count_execution_events(
        event_type="strategy_skip_sentiment",
        trading_day_yyyy_mm_dd=yyyy_mm_dd,
    )
    canary_any = db.count_canary_results_for_calendar_day_et(
        trading_day_yyyy_mm_dd=yyyy_mm_dd,
        successes_only=False,
    )
    canary_ok = db.count_canary_results_for_calendar_day_et(
        trading_day_yyyy_mm_dd=yyyy_mm_dd,
        successes_only=True,
    )

    out_dir = settings.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{yyyy_mm_dd}.md"

    pf_raw = summary.get("profit_factor")
    if pf_raw is None:
        pf_txt = "n/a"
    elif pf_raw == float("inf"):
        pf_txt = "inf"
    else:
        pf_txt = f"{float(pf_raw):.4f}"
    sr = summary.get("sharpe_ratio")
    sr_txt = "n/a" if sr is None else f"{float(sr):.6f}"

    body = (
        f"# Daily brief {yyyy_mm_dd}\n\n"
        f"- Total P&L: **{float(summary.get('net_pnl', 0.0)):.4f}**\n"
        f"- Closed trades: **{int(summary.get('closed_trades', 0))}**\n"
        f"- Win rate (%): **{summary.get('win_rate_pct')}**\n"
        f"- Profit factor: **{pf_txt}**\n"
        f"- Sharpe Ratio: **{sr_txt}**\n"
        f"- Max drawdown: **{summary.get('max_drawdown')}**\n"
        f"- Best trade: **{best_sym_txt}**\n"
        f"- Worst trade: **{worst_sym_txt}**\n\n"
        "## Signals\n\n"
        f"- Sentiment accuracy ({corr_n}/{elig_n}): **{acc_pct}**\n"
        f"- Trades sentiment-blocked (exec events): **{skips}**\n"
        f"- Passive joiner attempts: **{chase_att}**\n"
        f"- Passive joiner giveups: **{chase_gup}**\n\n"
        "## Operational / safety (best-effort)\n\n"
        f"- Canary DB rows recorded today: **{canary_any}** (successful: **{canary_ok}**)\n"
        "- Kill switch latch / Black Swan flash events are **not** mirrored in this SQLite "
        "brief; search application logs (`heartbeat`, `risk`, strategy) for latch and "
        "`black_swan` messages.\n\n"
        "---\n_Bot artefact — not a brokerage statement._\n"
    )
    path.write_text(body, encoding="utf-8")
    _LOG.info(
        "event=daily_report_generated path=%s trading_date=%s total_pnl=%.4f "
        "max_drawdown=%s",
        path,
        yyyy_mm_dd,
        float(summary.get("net_pnl", 0.0)),
        str(summary.get("max_drawdown", "n_a")),
    )
    return path


__all__ = ["generate_daily_report", "sentiment_accuracy_pct"]
