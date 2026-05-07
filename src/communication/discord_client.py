"""Discord command center: privileged slash commands + outbound embed queue."""

from __future__ import annotations

import asyncio
import logging
import textwrap
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.constants import LOGGER_APP
from config.settings import Settings
from config.strategy_runtime import (
    load_dynamic_params_file,
    merge_strategy_thresholds,
    resolve_dynamic_params_path,
)
from utils.tearsheet import build_summary_from_db_rows
from utils.time_utils import today_eastern

if TYPE_CHECKING:
    pass

_LOG = logging.getLogger(LOGGER_APP)


@dataclass(frozen=True)
class DiscordCallbacks:
    status_text_fn: Callable[[], str]
    report_text_fn: Callable[[], str]
    kill_fn: Callable[[], Awaitable[None]]
    skip_fn: Callable[[str], None]


async def post_discord_standalone_embed(
    settings: Settings,
    *,
    title: str,
    lines: list[str],
    color: int = 0xE74C3C,
) -> bool:
    """One-shot client for paths that occur before orchestrator discord task (e.g. canary fail).

    Returns True when Discord send is skipped (disabled) or completed successfully.
    """

    if not settings.ENABLE_DISCORD_BOT:
        return True
    tok = str(settings.DISCORD_BOT_TOKEN or "").strip()
    ch_raw = str(settings.DISCORD_CHANNEL_ID or "").strip()
    if not tok or not ch_raw:
        _LOG.warning(
            "event=discord_init_failed reason=missing_credentials title=%s",
            title[:120],
        )
        return False

    try:
        import discord  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("event=discord_init_failed reason=import_error err=%s", exc)
        return False

    channel_id_int = int(ch_raw)
    intents = discord.Intents.default()
    logged_out = asyncio.Event()
    send_ok = False

    class Once(discord.Client):
        async def setup_hook(self) -> None:
            asyncio.create_task(self._send_and_stop())

        async def _send_and_stop(self) -> None:
            nonlocal send_ok
            try:
                ch = self.get_channel(channel_id_int) or await self.fetch_channel(channel_id_int)
                emb = discord.Embed(title=title[:256], description="\n".join(lines)[:3500], color=color)
                await ch.send(embed=emb)
                send_ok = True
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("event=discord_standalone_send_failed title=%s err=%s", title[:120], exc)
            finally:
                logged_out.set()
                await self.close()

    client = Once(intents=intents)
    try:
        await asyncio.wait_for(client.start(tok), timeout=25.0)
        await asyncio.wait_for(logged_out.wait(), timeout=10.0)
    except TimeoutError:
        _LOG.warning("event=discord_standalone_send_failed reason=timeout title=%s", title[:120])
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "event=discord_standalone_send_failed reason=client_exc title=%s err=%s",
            title[:120],
            exc,
        )
    return bool(send_ok)


def startup_trading_mode_label(settings: Settings) -> str:
    """Human trading mode for startup banners (no broker calls)."""

    if settings.DRY_RUN:
        return "DRY_RUN"
    if (
        settings.ALPACA_ENV == "live"
        and settings.LIVE_TRADING_ENABLED
        and settings.can_submit_real_orders
    ):
        return "LIVE"
    return "PAPER"


def build_discord_first_contact_lines(
    settings: Settings,
    *,
    kill_switch_latched: bool,
) -> tuple[str, list[str], int]:
    """Title, embed lines, color for pre-orchestrator startup message."""

    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    mode = startup_trading_mode_label(settings)
    syms = settings.symbols_list
    sym_preview = ",".join(syms[:16]) + ("…" if len(syms) > 16 else "")
    line_list: list[str] = [
        "Bot started (pre-canary startup contact).",
        f"timestamp_utc={ts}",
        f"DRY_RUN={str(settings.DRY_RUN).lower()}",
        f"trading_mode={mode}",
        f"alpaca_env={settings.ALPACA_ENV}",
        f"live_trading_enabled={str(settings.LIVE_TRADING_ENABLED).lower()}",
        f"can_submit_real_orders={str(settings.can_submit_real_orders).lower()}",
        f"active_symbols={sym_preview or '(none)'}",
        f"heartbeat_interval_s={float(settings.HEARTBEAT_INTERVAL_SECONDS):g}",
        f"canary_enabled={str(settings.RUN_LIVE_CANARY_ON_STARTUP).lower()}",
        f"kill_switch={'Latched' if kill_switch_latched else 'Unlatched'}",
        f"black_swan_monitor={'enabled' if settings.BLACK_SWAN_ENABLED else 'disabled'}",
        f"ml_filter={'enabled' if settings.ENABLE_ML_FILTER else 'disabled'}",
        f"kelly_sizing={'enabled' if settings.ENABLE_KELLY_SIZING else 'disabled'}",
        f"autotune={'enabled' if settings.ENABLE_AUTOTUNE else 'disabled'}",
    ]
    if settings.DRY_RUN:
        line_list.append("DRY RUN ENABLED — simulated fills only")
    elif settings.ALPACA_ENV.lower() == "live" and bool(settings.can_submit_real_orders):
        line_list.append("⚠️ LIVE ACCOUNT — REAL ORDERS MAY BE POSTED TO THE BROKER")
    elif str(settings.ALPACA_ENV).lower() == "paper":
        line_list.append("Paper brokerage — LIVE_TRADING_ENABLED gate applies")
    else:
        line_list.append("Sim / non-paper Alpaca endpoint — verify order gates before trusting mode")

    t = "🚀 Bot startup (pre-flight)"
    real_live_hazard = settings.ALPACA_ENV.lower() == "live" and bool(settings.can_submit_real_orders)
    c = 0xE67E22 if real_live_hazard else 0x3498DB
    return t, line_list, c


async def discord_first_contact_standalone(settings: Settings, *, kill_switch_latched: bool) -> bool:
    """Send the first startup embed before canary/orchestrator."""

    if not settings.ENABLE_DISCORD_BOT:
        return True
    title_fb, fb_lines, col = build_discord_first_contact_lines(
        settings,
        kill_switch_latched=kill_switch_latched,
    )
    ok = await post_discord_standalone_embed(settings, title=title_fb, lines=fb_lines, color=col)
    if ok:
        _LOG.info("event=discord_startup_first_contact_sent dry_run=%s", str(settings.DRY_RUN).lower())
    else:
        _LOG.warning(
            "event=discord_init_failed reason=first_contact_not_sent dry_run=%s",
            str(settings.DRY_RUN).lower(),
        )
    return ok


def simulated_fill_discord_spec(evt: dict[str, Any]) -> dict[str, Any]:
    """Queue payload for simulated fill Discord (OrderService sink)."""

    fb_lines = [
        "SIMULATED FILL",
        f"dry_run={evt.get('dry_run', True)}",
        f"symbol={evt.get('symbol')}",
        f"side={evt.get('side')}",
        f"qty={evt.get('qty')}",
        f"intended_limit_price={evt.get('limit_price')}",
        f"simulated_fill_price={evt.get('simulated_fill_price')}",
        f"timestamp={evt.get('timestamp')}",
        f"strategy={evt.get('strategy')}",
        f"reason={evt.get('reason')}",
    ]
    return {"title": "SIMULATED FILL", "lines": fb_lines, "color": 0xF1C40F}


def enqueue_discord_alert(q: asyncio.Queue | None, spec: dict[str, Any]) -> None:
    if q is None:
        return
    try:
        q.put_nowait(spec)
    except asyncio.QueueFull:
        _LOG.warning("event=discord_alert_dropped queue_full")


class DiscordCommandCenter:
    """Long-lived discord.py client with slash commands and outbound consumer."""

    def __init__(self, settings: Settings, callbacks: DiscordCallbacks) -> None:
        self._settings = settings
        self._callbacks = callbacks

    async def run(self, outbound_embeds: asyncio.Queue[dict[str, Any]]) -> None:
        await drive_discord_task(self._settings, self._callbacks, outbound_embeds=outbound_embeds)


async def drive_discord_task(
    settings: Settings,
    callbacks: DiscordCallbacks,
    *,
    outbound_embeds: asyncio.Queue[dict[str, Any]],
) -> None:
    if not settings.ENABLE_DISCORD_BOT:
        return
    tok = str(settings.DISCORD_BOT_TOKEN or "").strip()
    ch_raw = str(settings.DISCORD_CHANNEL_ID or "").strip()
    if not tok or not ch_raw:
        _LOG.warning(
            "event=discord_init_failed reason=missing_credentials",
        )
        return

    allowed = settings.discord_allowed_user_ids_set
    rate: dict[int, float] = {}
    cooldown = float(settings.DISCORD_COMMAND_RATE_LIMIT_SECONDS)

    def rate_ok(uid: int) -> bool:
        now = time.time()
        if now - rate.get(uid, 0.0) < cooldown:
            return False
        rate[uid] = now
        return True

    try:
        import discord  # noqa: WPS433
        from discord import app_commands  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("event=discord_init_failed reason=import_error err=%s", exc)
        return

    intents = discord.Intents.default()
    channel_id_int = int(ch_raw)

    class _Bot(discord.Client):
        def __init__(self) -> None:
            super().__init__(intents=intents)
            self.tree = app_commands.CommandTree(self)

        async def alert_consumer(self) -> None:
            while True:
                spec = await outbound_embeds.get()
                title = str(spec.get("title", "NOTICE"))[:256]
                lines = spec.get("lines") or []
                color = int(spec.get("color", 0x3498DB))
                try:
                    ch = self.get_channel(channel_id_int) or await self.fetch_channel(channel_id_int)
                    emb = discord.Embed(
                        title=title,
                        description="\n".join(str(x) for x in lines)[:3500],
                        color=color,
                    )
                    await ch.send(embed=emb)
                    _LOG.info("event=discord_alert_sent title=%s", title)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("event=discord_alert_failed title=%s err=%s", title, exc)

        async def setup_hook(self) -> None:
            async def _auth(ix: discord.Interaction) -> bool:
                uid = getattr(getattr(ix, "user", None), "id", None)
                chan = getattr(ix, "channel_id", None)
                if chan is None or int(chan) != channel_id_int:
                    _LOG.warning("event=discord_command_rejected reason=wrong_channel")
                    await ix.response.send_message("Wrong channel.", ephemeral=True)
                    return False
                if uid is None or int(uid) not in allowed:
                    _LOG.warning("event=discord_command_rejected unauthorized user_id=%s", uid)
                    await ix.response.send_message("Unauthorized.", ephemeral=True)
                    return False
                if not rate_ok(int(uid)):
                    await ix.response.send_message("Rate limited.", ephemeral=True)
                    return False
                return True

            @self.tree.command(name="status", description="Bot diagnostics")  # type: ignore[misc]
            async def slash_status(ix: discord.Interaction) -> None:
                if not await _auth(ix):
                    return
                _LOG.info("event=discord_command_received command=/status user_id=%s", ix.user.id)
                blob = "```" + textwrap.shorten(callbacks.status_text_fn(), 1750, placeholder="…") + "```"
                await ix.response.send_message(blob[:2000])

            @self.tree.command(name="kill", description="Emergency kill + flatten")  # type: ignore[misc]
            async def slash_kill(ix: discord.Interaction) -> None:
                if not await _auth(ix):
                    return
                _LOG.critical("event=discord_remote_kill user_id=%s", ix.user.id)
                await ix.response.send_message("KILL latched remotely — initiating flatten.")
                await callbacks.kill_fn()

            @self.tree.command(name="report", description="Today ET summary (no secrets)")  # type: ignore[misc]
            async def slash_report(ix: discord.Interaction) -> None:
                if not await _auth(ix):
                    return
                _LOG.info("event=discord_command_received command=/report user_id=%s", ix.user.id)
                blob = "```" + textwrap.shorten(callbacks.report_text_fn(), 1750, placeholder="…") + "```"
                await ix.response.send_message(blob[:2000])

            @self.tree.command(name="skip", description="Skip symbol remainder of ET day")  # type: ignore[misc]
            async def slash_skip(ix: discord.Interaction, symbol: str) -> None:
                if not await _auth(ix):
                    return
                sym_u = symbol.strip().upper()
                callbacks.skip_fn(sym_u)
                _LOG.warning("event=discord_skip_symbol symbol=%s user_id=%s", sym_u, ix.user.id)
                await ix.response.send_message(f"`{sym_u}` skipped remainder of today's ET session window.")

            await self.tree.sync()
            asyncio.create_task(self.alert_consumer())

    bot = _Bot()
    try:
        await bot.start(tok)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("event=discord_init_failed reason=bot_start err=%s", exc)


def startup_initialization_notification(
    *,
    settings: Settings,
    equity: float | None,
    buying_power: float | None,
    symbols_preview: str,
    kill_switch_latched: bool,
    heartbeat_active: bool,
    canary_status: str,
    ml_ready: bool,
    risk_mode_label: str = "normal",
) -> dict[str, Any]:
    """Rich embed payload for Discord (no secrets)."""

    title = "🚀 Dry Run Initialization" if settings.DRY_RUN else "🚀 Live Initialization"
    lines = [
        f"mode={'DRY_RUN' if settings.DRY_RUN else 'LIVE'}",
        f"alpaca_env={settings.ALPACA_ENV}",
        (
            f"equity_usd={float(equity):,.2f}"
            if equity is not None and str(equity) != ""
            else "equity_usd=n/a"
        ),
        (
            f"buying_power_usd={float(buying_power):,.2f}"
            if buying_power is not None
            else "buying_power_usd=n/a"
        ),
        f"active_symbols={symbols_preview}",
        f"risk_mode={risk_mode_label}",
        f"Kelly_sizing={'Enabled' if settings.ENABLE_KELLY_SIZING else 'Disabled'}",
        f"ML_signal_filter={'Ready' if ml_ready else 'Not_ready'} "
        f"(enabled={str(settings.ENABLE_ML_FILTER).lower()})",
        f"canary={canary_status}",
        f"Kill_switch={'Latched' if kill_switch_latched else 'Unlatched'}",
        f"Heartbeat={'Active' if heartbeat_active else 'Pending'}",
    ]
    return {"title": title, "lines": lines, "color": 0x3498DB}


def format_dynamic_params_digest(settings: Settings) -> str:
    rp = resolve_dynamic_params_path(settings)
    d = load_dynamic_params_file(rp)
    if not d:
        return "dynamic_params=n/a"
    try:
        parts = [
            f"rsi_entry={float(d['rsi_entry_threshold']):g}",
            f"rsi_exit={float(d['rsi_exit_threshold']):g}",
            f"adx={float(d['adx_threshold']):g}",
            f"atr_stop={float(d['atr_stop_multiplier']):g}",
            f"trail_atr={float(d['atr_trailing_multiplier']):g}",
            f"pick={str(d.get('selected_at',''))[:19]}",
        ]
        return "autotuned | " + " ".join(parts)
    except (KeyError, TypeError, ValueError):
        return "dynamic_params=invalid"


def format_report(orchestrator: Any) -> str:
    settings = orchestrator._settings
    db = orchestrator._database
    day = today_eastern().strftime("%Y-%m-%d")
    trades = db.get_completed_trades_for_calendar_day_et(
        trading_day_yyyy_mm_dd=day,
        exclude_canary=True,
    )
    summary = build_summary_from_db_rows(trades)
    positions = orchestrator._latest_positions or []
    open_rows = [p for p in positions if abs(float(getattr(p, "qty", 0) or 0)) >= 0.99]

    pf_raw = summary.get("profit_factor")
    pf_txt = "n/a" if pf_raw is None else (f"{float(pf_raw):.4f}" if pf_raw != float("inf") else "inf")
    mdd = summary.get("max_drawdown")
    mdd_txt = "n/a" if mdd is None else f"{float(mdd):.4f}"

    lines = [
        f"et_day={day}",
        f"closed_trades={int(summary.get('closed_trades', 0))}",
        f"win_rate_pct={summary.get('win_rate_pct', 'n/a')}",
        f"net_pnl={float(summary.get('net_pnl', 0.0)):.4f}",
        f"max_drawdown={mdd_txt}",
        f"profit_factor={pf_txt}",
        f"open_positions={len(open_rows)}",
        format_dynamic_params_digest(settings),
    ]
    return "\n".join(lines)


def format_status(orchestrator: Any) -> str:
    settings = orchestrator._settings
    ks = orchestrator._kill_switch
    thr = merge_strategy_thresholds(settings, dyn_path=resolve_dynamic_params_path(settings))
    blk = orchestrator._black_swan.triggered()
    stream = orchestrator._stream_health.all_ok
    hb = Path(settings.LOG_DIR) / "heartbeat.log"
    hb_txt = "n/a"
    try:
        if hb.is_file():
            hb_txt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(hb.stat().st_mtime))
    except OSError:
        pass

    day = today_eastern().isoformat()
    ml = getattr(orchestrator, "_ml_filter", None)
    ml_ok = bool(getattr(ml, "is_trained", lambda: False)())

    positions = orchestrator._latest_positions or []
    opn = len([p for p in positions if abs(float(getattr(p, "qty", 0) or 0)) >= 0.99])

    acc = getattr(orchestrator, "_latest_account", None)
    eq_txt = f"{float(acc.equity):.2f}" if acc is not None else "n/a"

    db = getattr(orchestrator, "_database", None)
    trades_today: list[Any] = []
    if db is not None and hasattr(db, "get_completed_trades_for_calendar_day_et"):
        try:
            trades_today = list(
                db.get_completed_trades_for_calendar_day_et(
                    trading_day_yyyy_mm_dd=day,
                    exclude_canary=True,
                )
                or [],
            )
        except TypeError:
            trades_today = []
    day_pnl = 0.0
    for r in trades_today:
        rv = r["realized_pnl"]
        if rv is not None:
            try:
                day_pnl += float(rv)
            except (TypeError, ValueError):
                continue

    dry = "true" if settings.DRY_RUN else "false"
    live_submit = "true" if settings.can_submit_real_orders else "false"
    mode = f"{settings.ALPACA_ENV}/{'live_submit' if live_submit == 'true' else 'no_submit'}/dry_run={dry}"

    corr = "on" if settings.CORRELATION_BREAKER_ENABLED else "off"
    bsw = "on" if settings.BLACK_SWAN_ENABLED else "off"

    return "\n".join(
        [
            f"et_day={day}",
            f"mode={mode}",
            f"DRY_RUN={dry}",
            f"KILL_SWITCH={'latched' if ks.is_latched() else 'ok'}",
            f"BLACK_SWAN_MONITOR={'TRIGGERED' if blk else f'armed({bsw})'}",
            f"CORRELATION_BREAKER={corr}",
            f"quotes_stream_ok={'true' if stream else 'false'}",
            f"OPEN_POSITIONS={opn}",
            f"equity={eq_txt}",
            f"daily_realized_pnl_et={day_pnl:.4f}",
            f"MAX_RISK_PER_TRADE_PCT_env={settings.MAX_RISK_PER_TRADE_PCT:.6f}",
            f"rsi_entry/exit/adx/atr_stop/trail_atr="
            f"{thr.rsi_oversold:g}/{thr.rsi_exit:g}/{thr.adx_range_max:g}/{thr.atr_stop_multiplier:g}/{thr.trail_atr_multiplier:g}",
            f"AUTOTUNE={settings.ENABLE_AUTOTUNE} ML={settings.ENABLE_ML_FILTER} KELLY={settings.ENABLE_KELLY_SIZING}",
            f"ML_trained={'yes' if ml_ok else 'no'} ML_threshold={settings.ML_FILTER_THRESHOLD:.4f}",
            f"heartbeat_log_mtime_utc={hb_txt}",
            format_dynamic_params_digest(settings),
            "secrets=N/A",
        ],
    )


__all__ = [
    "DiscordCallbacks",
    "DiscordCommandCenter",
    "build_discord_first_contact_lines",
    "discord_first_contact_standalone",
    "drive_discord_task",
    "enqueue_discord_alert",
    "format_dynamic_params_digest",
    "format_report",
    "format_status",
    "post_discord_standalone_embed",
    "simulated_fill_discord_spec",
    "startup_initialization_notification",
    "startup_trading_mode_label",
]

