"""Offline preflight / startup helper tests (no Alpaca or Discord IO)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from communication.discord_client import build_discord_first_contact_lines
from core.market_data import Quote, QuoteCache
from core.orders import OrderService
from core.state_store import StateStore
from utils.preflight import ensure_runtime_paths, path_writable_quick


class _StubBroker:
    def submit_order(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("broker should not be called in offline tests")

    def get_order_by_client_id(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("broker lookup should not be called in offline tests")


def _quote(sym: str = "SPY") -> Quote:
    from datetime import UTC, datetime

    return Quote(
        symbol=sym,
        bid=100.0,
        ask=100.08,
        bid_size=1.0,
        ask_size=1.0,
        timestamp=datetime.now(UTC),
        feed="iex",
    )


def test_requirements_lists_core_runtime_packages() -> None:
    root = Path(__file__).resolve().parent.parent
    text = (root / "requirements.txt").read_text()
    for name in (
        "discord.py",
        "scikit-learn",
        "xgboost",
        "psutil",
        "python-dotenv",
        "alpaca-py",
    ):
        assert name in text


def test_ensure_runtime_paths_creates_dirs_and_logs(
    caplog: pytest.LogCaptureFixture,
    make_settings_factory,
    tmp_path: Path,
) -> None:
    base = tmp_path / "pf"
    st = base / "state"
    logs = base / "logs"
    rep = base / "reports"
    dbp = base / "data" / "app.sqlite"

    settings = make_settings_factory(
        STATE_DIR=str(st),
        LOG_DIR=str(logs),
        REPORTS_DIR=str(rep),
        DATABASE_PATH=str(dbp),
    )

    caplog.set_level(logging.INFO)
    ensure_runtime_paths(settings)

    assert st.is_dir() and (st / "models").is_dir() and (st / "cache").is_dir()
    assert logs.is_dir() and rep.is_dir() and dbp.parent.is_dir()
    assert path_writable_quick(st)

    msgs = [r.getMessage() for r in caplog.records]
    hits = [m for m in msgs if "event=preflight_path_check" in m]
    assert len(hits) >= 4


@pytest.mark.skipif(sys.platform == "win32", reason="directory permission semantics differ")
def test_ensure_runtime_paths_raises_when_dir_not_writable(
    make_settings_factory,
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    good_state = tmp_path / "good_state"
    logs = tmp_path / "logs"
    dbp = tmp_path / "db.sqlite"

    settings = make_settings_factory(
        STATE_DIR=str(good_state),
        LOG_DIR=str(logs),
        REPORTS_DIR=str(blocked),
        DATABASE_PATH=str(dbp),
    )
    prev = blocked.stat().st_mode
    os.chmod(blocked, 0o555)
    try:
        with pytest.raises(RuntimeError, match=r"Preflight failed.*writable"):
            ensure_runtime_paths(settings)
    finally:
        os.chmod(blocked, prev)


def test_path_writable_quick_false_on_blocked_dir(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("chmod directory writability differs on Windows")
    d = tmp_path / "ro"
    d.mkdir()
    prev = d.stat().st_mode
    os.chmod(d, 0o555)
    try:
        assert path_writable_quick(d) is False
    finally:
        os.chmod(d, prev)


def test_discord_first_contact_banner_dry_vs_live(make_settings_factory) -> None:
    dry = make_settings_factory(DRY_RUN=True)
    t1, lines1, c1 = build_discord_first_contact_lines(dry, kill_switch_latched=False)
    assert "dry" in t1.lower() or "startup" in t1.lower()
    assert any("DRY RUN ENABLED" in ln for ln in lines1)
    assert c1 == 0x3498DB

    live = make_settings_factory(
        DRY_RUN=False,
        ALPACA_ENV="live",
        LIVE_TRADING_ENABLED=True,
        CONFIRM_LIVE_TRADING="yes_i_understand",
    )
    _t2, lines2, c2 = build_discord_first_contact_lines(live, kill_switch_latched=False)
    assert any("LIVE ACCOUNT" in ln or "REAL ORDERS" in ln for ln in lines2)
    assert c2 == 0xE67E22


def test_simulated_fill_sink_invoked_on_dry_run_limit_entry(
    caplog: pytest.LogCaptureFixture,
    make_settings_factory,
    tmp_path: Path,
) -> None:
    seen: list[dict] = []

    def sink(pl: dict) -> None:
        seen.append(pl)

    settings = make_settings_factory(
        DRY_RUN=True,
        LIVE_TRADING_ENABLED=False,
        STATE_DIR=str(tmp_path / "st"),
        LOG_DIR=str(tmp_path / "logs"),
        DATABASE_PATH=str(tmp_path / "d.sqlite"),
    )
    qc = QuoteCache(max_age_seconds=5.0, feed="iex")
    state = StateStore(tmp_path / "ost")
    svc = OrderService(
        _StubBroker(),
        settings,
        state,
        qc,
        strategy_name="test",
        simulated_fill_sink=sink,
    )
    caplog.set_level(logging.INFO)
    q = _quote()
    wo = svc.submit_limit_entry("SPY", 1, "buy", quote=q, intent_reason="unit_test")

    assert wo is not None and wo.status == "dry_run"
    assert len(seen) == 1
    assert seen[0]["symbol"] == "SPY"
    assert seen[0]["reason"] == "unit_test"
    assert seen[0]["dry_run"] is True
    assert any("event=simulated_fill" in r.getMessage() for r in caplog.records)
    sim_line = next(r.getMessage() for r in caplog.records if "event=simulated_fill" in r.getMessage())
    assert "discord_notified=true" in sim_line


def test_dry_run_limit_entry_preserves_fractional_quantity(
    make_settings_factory,
    tmp_path: Path,
) -> None:
    seen: list[dict] = []
    settings = make_settings_factory(
        DRY_RUN=True,
        LIVE_TRADING_ENABLED=False,
        ENABLE_FRACTIONAL=True,
        FRACTIONAL_MIN_QTY=0.001,
        STATE_DIR=str(tmp_path / "st_frac"),
        LOG_DIR=str(tmp_path / "logs_frac"),
        DATABASE_PATH=str(tmp_path / "frac.sqlite"),
    )
    svc = OrderService(
        _StubBroker(),
        settings,
        StateStore(tmp_path / "ost_frac"),
        QuoteCache(max_age_seconds=5.0, feed="iex"),
        strategy_name="test",
        simulated_fill_sink=seen.append,
    )
    wo = svc.submit_limit_entry("TSLA", 0.177, "buy", quote=_quote(), intent_reason="fractional_unit_test")

    assert wo is not None
    assert wo.qty == pytest.approx(0.177)
    assert seen[0]["qty"] == pytest.approx(0.177)


@pytest.mark.asyncio
async def test_require_discord_on_startup_exits_before_canary(
    monkeypatch: pytest.MonkeyPatch,
    make_settings_factory,
    tmp_path: Path,
) -> None:
    import main as main_mod

    s = make_settings_factory(
        ENABLE_DISCORD_BOT=True,
        REQUIRE_DISCORD_ON_STARTUP=True,
        STATE_DIR=str(tmp_path / "st"),
        DATABASE_PATH=str(tmp_path / "d.sqlite"),
        LOG_DIR=str(tmp_path / "logs"),
        REPORTS_DIR=str(tmp_path / "reports"),
    )

    async def _discord_failed(*_: object, **__: object) -> bool:
        return False

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)
    monkeypatch.setattr(main_mod, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "ensure_runtime_paths", lambda _settings: None)
    monkeypatch.setattr(main_mod, "discord_first_contact_standalone", _discord_failed)

    class _Ks:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def is_latched(self) -> bool:
            return False

    monkeypatch.setattr(main_mod, "KillSwitch", _Ks)

    rc = await main_mod._amain()
    assert rc == 7


def test_preflight_duplicate_roots_deduped(
    caplog: pytest.LogCaptureFixture,
    make_settings_factory,
    tmp_path: Path,
) -> None:
    """If STATE_DIR equals LOG_DIR parent tricks aren't needed; overlap should not explode."""
    one = tmp_path / "solo"
    one.mkdir()
    dbp = one / "x.sqlite"
    settings = make_settings_factory(
        STATE_DIR=str(one),
        LOG_DIR=str(one / "logs"),
        REPORTS_DIR=str(one / "reports"),
        DATABASE_PATH=str(dbp),
    )
    caplog.set_level(logging.INFO)
    ensure_runtime_paths(settings)
    paths_logged = sum(1 for r in caplog.records if "event=preflight_path_check" in r.getMessage())
    assert paths_logged >= 1
