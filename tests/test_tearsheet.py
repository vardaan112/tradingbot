"""Tests for intraday orders-log tearsheet summarization."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from utils.tearsheet import format_tearsheet_markdown_table, get_tearsheet_summary


def _line(
    *,
    ts: str,
    strategy: str,
    coid: str,
    sym: str,
    side: str,
    filled: float,
    avg: float,
    status: str = "filled",
) -> str:
    return (
        f"{ts} | INFO | tradingbot.orders | mode=paper | reg=auto | symbol={sym} | "
        f"strategy={strategy} | coid={coid} | Trade update coid={coid} "
        f"symbol={sym} side={side} status={status} filled={filled:.4f} avg={avg:.4f}"
    )


def test_missing_log_returns_safe_summary(tmp_path: Path):
    out = get_tearsheet_summary(tmp_path / "does_not_exist.log")
    assert out["ok"] is False
    assert out["reason"] == "missing_log"


def test_excludes_canary_and_computes_profit_factor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    d = date(2030, 4, 15)
    monkeypatch.setattr("utils.tearsheet.today_eastern", lambda: d)
    log = tmp_path / "orders.log"
    log.write_text(
        "\n".join(
            [
                _line(
                    ts="2030-04-15T14:02:00+00:00",
                    strategy="rsi_meanrev",
                    coid="a",
                    sym="SPY",
                    side="buy",
                    filled=10.0,
                    avg=100.0,
                ),
                _line(
                    ts="2030-04-15T14:03:00+00:00",
                    strategy="rsi_meanrev",
                    coid="b",
                    sym="SPY",
                    side="sell",
                    filled=10.0,
                    avg=110.0,
                ),
                _line(
                    ts="2030-04-15T14:04:00+00:00",
                    strategy="startup_canary_check",
                    coid="cx",
                    sym="XLF",
                    side="buy",
                    filled=1.0,
                    avg=30.0,
                ),
                _line(
                    ts="2030-04-15T14:05:00+00:00",
                    strategy="startup_canary_check",
                    coid="cy",
                    sym="XLF",
                    side="sell",
                    filled=1.0,
                    avg=31.0,
                ),
            ],
        ),
        encoding="utf-8",
    )
    summary = get_tearsheet_summary(log)
    assert summary["ok"] is True
    assert summary["closed_trades"] == 1
    assert summary["gross_profit"] == pytest.approx(100.0)
    assert summary["profit_factor"] == float("inf")


def test_sharpe_none_with_single_fill_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    d = date(2030, 8, 1)
    monkeypatch.setattr("utils.tearsheet.today_eastern", lambda: d)
    log = tmp_path / "orders.log"
    log.write_text(
        _line(
            ts="2030-08-01T15:00:00+00:00",
            strategy="rsi_meanrev",
            coid="z1",
            sym="QQQ",
            side="buy",
            filled=1.0,
            avg=400.0,
        )
        + "\n"
        + _line(
            ts="2030-08-01T15:01:00+00:00",
            strategy="rsi_meanrev",
            coid="z2",
            sym="QQQ",
            side="sell",
            filled=1.0,
            avg=395.0,
        ),
        encoding="utf-8",
    )
    summary = get_tearsheet_summary(log)
    assert summary["ok"] is True
    assert summary["closed_trades"] == 1
    assert summary["sharpe_ratio"] is None


def test_sharpe_computed_with_two_trades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    d = date(2031, 1, 10)
    monkeypatch.setattr("utils.tearsheet.today_eastern", lambda: d)
    lines: list[str] = []
    t = 0
    for coid, buy_p, sell_p, q in [("t1", 100.0, 105.0, 10.0), ("t2", 50.0, 48.0, 10.0)]:
        lines.append(
            _line(
                ts=f"2031-01-10T16:{t:02d}:00+00:00",
                strategy="rsi_meanrev",
                coid=f"{coid}b",
                sym="SPY",
                side="buy",
                filled=q,
                avg=buy_p,
            ),
        )
        t += 1
        lines.append(
            _line(
                ts=f"2031-01-10T16:{t:02d}:00+00:00",
                strategy="rsi_meanrev",
                coid=f"{coid}s",
                sym="SPY",
                side="sell",
                filled=q,
                avg=sell_p,
            ),
        )
        t += 1
    log = tmp_path / "orders.log"
    log.write_text("\n".join(lines), encoding="utf-8")
    summary = get_tearsheet_summary(log)
    assert summary["ok"] is True
    assert summary["closed_trades"] == 2
    assert summary["gross_loss"] > 1e-9
    assert summary["profit_factor"] == pytest.approx(
        summary["gross_profit"] / summary["gross_loss"],
    )
    assert summary["sharpe_ratio"] is not None


def test_max_drawdown_on_cumulative_slices_and_win_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """MDD applies to cumulative FIFO ``pnls`` slices; win rate counts winning slices."""

    d = date(2031, 2, 2)
    monkeypatch.setattr("utils.tearsheet.today_eastern", lambda: d)
    log = tmp_path / "orders.log"
    lines = [
        _line(
            ts="2031-02-02T15:01:00+00:00",
            strategy="rsi_meanrev",
            coid="b1",
            sym="SPY",
            side="buy",
            filled=10.0,
            avg=100.0,
        ),
        _line(
            ts="2031-02-02T15:02:00+00:00",
            strategy="rsi_meanrev",
            coid="s1",
            sym="SPY",
            side="sell",
            filled=10.0,
            avg=105.0,
        ),
        _line(
            ts="2031-02-02T15:03:00+00:00",
            strategy="rsi_meanrev",
            coid="b2",
            sym="SPY",
            side="buy",
            filled=10.0,
            avg=50.0,
        ),
        _line(
            ts="2031-02-02T15:04:00+00:00",
            strategy="rsi_meanrev",
            coid="s2",
            sym="SPY",
            side="sell",
            filled=10.0,
            avg=48.0,
        ),
    ]
    log.write_text("\n".join(lines), encoding="utf-8")
    summary = get_tearsheet_summary(log)
    assert summary["ok"] is True
    assert summary["closed_trades"] == 2
    # +50 then -20 on cumulative equity; peak 50 then 30 -> drawdown 20
    assert summary["max_drawdown"] == pytest.approx(20.0)
    assert summary["win_rate_pct"] == pytest.approx(50.0)


def test_format_tearsheet_markdown_table_includes_expected_rows():
    md = format_tearsheet_markdown_table(
        {
            "ok": True,
            "closed_trades": 3,
            "net_pnl": -1.5,
            "gross_profit": 4.0,
            "gross_loss": 5.5,
            "profit_factor": 0.73,
            "sharpe_ratio": 0.12,
            "max_drawdown": 2.0,
            "win_rate_pct": 33.3333,
            "reason": "ok",
        },
    )
    assert "| Metric | Value |" in md
    assert "max_drawdown" in md
    assert "win_rate_pct" in md
