"""ATR-based position sizing with hard clamps.

Sizing pipeline:
1. capital_base = settings.resolved_capital_base(account.equity)
2. effective_risk_pct = MAX_RISK_PER_TRADE_PCT * conviction_risk_multiplier *
   anti_martingale_multiplier (default 1)
3. risk_budget = capital_base * effective_risk_pct
4. stop_distance = ATR * ATR_STOP_MULTIPLIER
5. raw_shares = floor(risk_budget / stop_distance)
6. clamp by existing USD / BP / gross / bot-managed ceilings
7. integer shares unless ENABLE_FRACTIONAL
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config.constants import LOGGER_RISK
from config.settings import Settings
from core.account import AccountSnapshot, PositionSnapshot
from core.database import Database

from . import kelly_sizer
from .compliance import ComplianceAdapter
from .exposure import ExposureChecker

KellySizer = kelly_sizer.KellySizer
compute_kelly_risk_scaling = kelly_sizer.compute_kelly_risk_scaling
kelly_fraction_from_trade_stats = kelly_sizer.kelly_fraction_from_trade_stats


@dataclass(frozen=True)
class PositionSize:
    """Result of a sizing computation."""

    symbol: str
    shares: float  # 0 means "skip"
    notional: float
    entry_price: float
    stop_distance: float
    risk_budget: float
    capital_base: float
    conviction_risk_multiplier: float
    effective_risk_pct: float
    rationale: str
    skipped_reason: str | None = None
    risk_mode: str = "normal"
    anti_martingale_multiplier: float = 1.0
    sizing_mode: str = "flat_atr"
    kelly_win_rate: float | None = None
    kelly_avg_win: float | None = None
    kelly_avg_loss: float | None = None
    kelly_profit_factor: float | None = None
    kelly_fraction: float | None = None
    kelly_modified: float | None = None
    risk_pct_before_kelly_caps: float | None = None


class PositionSizer:
    """Compute trade sizes with strict risk caps."""

    def __init__(
        self,
        settings: Settings,
        compliance: ComplianceAdapter,
        exposure: ExposureChecker,
        database: Database | None = None,
    ) -> None:
        self._settings = settings
        self._compliance = compliance
        self._exposure = exposure
        self._database = database
        self._log = logging.getLogger(LOGGER_RISK)

    def size(
        self,
        *,
        symbol: str,
        entry_price: float,
        atr: float,
        account: AccountSnapshot,
        positions: list[PositionSnapshot],
        bot_managed_notional: float,
        conviction_risk_multiplier: float = 1.0,
        sizing_block_reason: str | None = None,
        anti_martingale_multiplier: float = 1.0,
        risk_mode: str = "normal",
        recent_trade_hint: str = "",
    ) -> PositionSize:
        if sizing_block_reason:
            return self._skip(
                symbol,
                sizing_block_reason,
                entry_price=entry_price,
                conviction_risk_multiplier=conviction_risk_multiplier,
                anti_martingale_multiplier=anti_martingale_multiplier,
                risk_mode=risk_mode,
            )
        if entry_price <= 0:
            return self._skip(
                symbol, "non_positive_entry_price", entry_price=entry_price
            )
        if atr <= 0:
            return self._skip(symbol, "non_positive_atr", entry_price=entry_price)
        if account.equity <= 0:
            return self._skip(symbol, "non_positive_equity", entry_price=entry_price)
        if conviction_risk_multiplier <= 0 or not math.isfinite(conviction_risk_multiplier):
            return self._skip(
                symbol, "invalid_conviction_multiplier",
                entry_price=entry_price,
                anti_martingale_multiplier=anti_martingale_multiplier,
                risk_mode=risk_mode,
            )

        am_mult = float(anti_martingale_multiplier)
        if am_mult <= 0 or not math.isfinite(am_mult) or am_mult > 1.0001:
            return self._skip(
                symbol,
                "invalid_anti_martingale_multiplier",
                entry_price=entry_price,
                conviction_risk_multiplier=conviction_risk_multiplier,
                anti_martingale_multiplier=am_mult,
                risk_mode=risk_mode,
            )

        capital_base = self._settings.resolved_capital_base(account.equity)
        if capital_base <= 0:
            return self._skip(
                symbol,
                "non_positive_capital_base",
                entry_price=entry_price,
                capital_base=capital_base,
            )

        base_risk_pct = (
            self._settings.MAX_RISK_PER_TRADE_PCT
            * float(conviction_risk_multiplier)
            * am_mult
        )
        sizing_mode = "flat_atr"
        k_stats: dict[str, float] = {}
        risk_pct_pre_kelly = float(base_risk_pct)
        fk = None
        mod_k = None
        k_mult = 1.0
        eff_risk_pct: float
        if self._settings.ENABLE_KELLY_SIZING and self._database is not None:
            ks = KellySizer(self._settings, self._database)
            kdec = ks.get_adjusted_risk_pct(symbol, float(base_risk_pct), {})
            eff_anchor = float(kdec.risk_pct)
            fk = float(kdec.kelly_fraction)
            mod_k = float(self._settings.KELLY_FRACTION) * max(0.0, fk)
            sizing_mode = "kelly_atr" if kdec.sizing_mode == "kelly_adjusted" else "flat_atr"
            k_mult = eff_anchor / float(base_risk_pct) if float(base_risk_pct) > 1e-15 else 1.0
            k_stats = {
                "win_rate": float(kdec.win_rate),
                "avg_win": float(kdec.avg_win),
                "avg_loss": float(kdec.avg_loss),
                "profit_factor": float(kdec.profit_factor),
                "full_kelly": fk,
                "modified_kelly": mod_k,
                "risk_mult_uncapped": float(k_mult),
                "sample_n": float(kdec.sample_n),
            }
            eff_risk_pct = float(eff_anchor)
        else:
            eff_risk_pct = float(base_risk_pct)
        risk_budget = capital_base * eff_risk_pct
        stop_distance = atr * self._settings.ATR_STOP_MULTIPLIER
        if stop_distance <= 0:
            return self._skip(
                symbol,
                "non_positive_stop_distance",
                entry_price=entry_price,
                capital_base=capital_base,
                conviction_risk_multiplier=conviction_risk_multiplier,
                effective_risk_pct=eff_risk_pct,
            )

        raw_shares = risk_budget / stop_distance

        usd_cap_shares = self._settings.MAX_EQUITY_USAGE_USD / entry_price
        bp = self._compliance.buying_power(account)
        bp_shares = bp / entry_price if bp > 0 else 0.0

        gross_cap_dollars = max(
            0.0,
            self._settings.MAX_GROSS_EXPOSURE_PCT * account.equity
            - sum(abs(p.market_value) for p in positions),
        )
        gross_cap_shares = gross_cap_dollars / entry_price if entry_price > 0 else 0.0

        remaining_bot = max(0.0, self._settings.MAX_EQUITY_USAGE_USD - bot_managed_notional)
        remaining_bot_shares = remaining_bot / entry_price if entry_price > 0 else 0.0

        candidates = [
            ("raw_atr", raw_shares),
            ("usd_cap", usd_cap_shares),
            ("buying_power", bp_shares),
            ("gross_exposure", gross_cap_shares),
            ("bot_managed_remaining", remaining_bot_shares),
        ]
        clamping_reason, clamped = min(candidates, key=lambda x: x[1])

        fractional_enabled = bool(self._settings.ENABLE_FRACTIONAL)
        min_shares = float(self._settings.MIN_SHARES)
        if fractional_enabled:
            min_shares = min(min_shares, 0.001)
        if not fractional_enabled:
            shares: float = float(math.floor(clamped))
        else:
            shares = max(0.0, math.floor(clamped * 1000.0) / 1000.0)

        if shares < min_shares:
            max_alloc = float(self._settings.max_dollars_per_trade)
            floored_shares = float(math.floor(clamped))
            if (not fractional_enabled) and (max_alloc / entry_price) < min_shares:
                explicit_reason = (
                    "SIZE_ZERO: price exceeds max allocation and fractional trading is disabled"
                )
            else:
                explicit_reason = (
                    f"SIZE_ZERO: clamped shares below MIN_SHARES={min_shares:.4f}"
                )
            self._log.warning(
                "event=size_zero symbol=%s code=SIZE_ZERO price=%.6f max_allocation_usd=%.6f "
                "raw_shares=%.8f clamped_shares=%.8f floored_shares=%.8f final_shares=%.8f "
                "fractional_enabled=%s min_shares=%.4f clamping_reason=%s",
                symbol,
                entry_price,
                max_alloc,
                raw_shares,
                clamped,
                floored_shares,
                shares,
                str(fractional_enabled).lower(),
                min_shares,
                clamping_reason,
                extra={"symbol": symbol},
            )
            self._log_sizing(
                symbol=symbol,
                capital_base=capital_base,
                risk_budget=risk_budget,
                atr=atr,
                stop_distance=stop_distance,
                raw_shares=raw_shares,
                clamping_reason=clamping_reason,
                final_shares=shares,
                outcome="skip",
                conviction_risk_multiplier=conviction_risk_multiplier,
                effective_risk_pct=eff_risk_pct,
                risk_mode=risk_mode,
                anti_martingale_multiplier=am_mult,
                recent_trade_hint=recent_trade_hint,
                sizing_mode=sizing_mode,
                risk_pct_before_kelly_caps=risk_pct_pre_kelly,
                risk_pct_after_kelly=float(eff_risk_pct),
                kelly_win_rate=k_stats.get("win_rate"),
                kelly_avg_win=k_stats.get("avg_win"),
                kelly_avg_loss=k_stats.get("avg_loss"),
                kelly_profit_factor=k_stats.get("profit_factor"),
                kelly_fraction=fk,
                kelly_modified=mod_k,
            )
            return self._skip(
                symbol,
                (
                    f"{explicit_reason}|price={entry_price:.6f}|cap={max_alloc:.6f}"
                    f"|raw_shares={raw_shares:.8f}|floored_shares={floored_shares:.8f}"
                    f"|fractional_enabled={str(fractional_enabled).lower()}"
                    f"|clamped_by={clamping_reason}"
                ),
                entry_price=entry_price,
                stop_distance=stop_distance,
                risk_budget=risk_budget,
                capital_base=capital_base,
                conviction_risk_multiplier=conviction_risk_multiplier,
                effective_risk_pct=eff_risk_pct,
                anti_martingale_multiplier=am_mult,
                risk_mode=risk_mode,
            )

        proposed_notional = shares * entry_price
        decision = self._exposure.check(
            account=account,
            positions=positions,
            proposed_notional=proposed_notional,
            bot_managed_notional=bot_managed_notional,
        )
        if not decision.allowed:
            self._log_sizing(
                symbol=symbol,
                capital_base=capital_base,
                risk_budget=risk_budget,
                atr=atr,
                stop_distance=stop_distance,
                raw_shares=raw_shares,
                clamping_reason=f"exposure:{decision.reason}",
                final_shares=0.0,
                outcome="skip",
                conviction_risk_multiplier=conviction_risk_multiplier,
                effective_risk_pct=eff_risk_pct,
                risk_mode=risk_mode,
                anti_martingale_multiplier=am_mult,
                recent_trade_hint=recent_trade_hint,
                sizing_mode=sizing_mode,
                risk_pct_before_kelly_caps=risk_pct_pre_kelly,
                risk_pct_after_kelly=float(eff_risk_pct),
                kelly_win_rate=k_stats.get("win_rate"),
                kelly_avg_win=k_stats.get("avg_win"),
                kelly_avg_loss=k_stats.get("avg_loss"),
                kelly_profit_factor=k_stats.get("profit_factor"),
                kelly_fraction=fk,
                kelly_modified=mod_k,
            )
            return self._skip(
                symbol,
                f"exposure_check:{decision.reason}",
                entry_price=entry_price,
                stop_distance=stop_distance,
                risk_budget=risk_budget,
                capital_base=capital_base,
                conviction_risk_multiplier=conviction_risk_multiplier,
                effective_risk_pct=eff_risk_pct,
                anti_martingale_multiplier=am_mult,
                risk_mode=risk_mode,
            )

        self._log_sizing(
            symbol=symbol,
            capital_base=capital_base,
            risk_budget=risk_budget,
            atr=atr,
            stop_distance=stop_distance,
            raw_shares=raw_shares,
            clamping_reason=clamping_reason,
            final_shares=shares,
            outcome="ok",
            conviction_risk_multiplier=conviction_risk_multiplier,
            effective_risk_pct=eff_risk_pct,
            risk_mode=risk_mode,
            anti_martingale_multiplier=am_mult,
            recent_trade_hint=recent_trade_hint,
            sizing_mode=sizing_mode,
            risk_pct_before_kelly_caps=risk_pct_pre_kelly,
            risk_pct_after_kelly=float(eff_risk_pct),
            kelly_win_rate=k_stats.get("win_rate"),
            kelly_avg_win=k_stats.get("avg_win"),
            kelly_avg_loss=k_stats.get("avg_loss"),
            kelly_profit_factor=k_stats.get("profit_factor"),
            kelly_fraction=fk,
            kelly_modified=mod_k,
            final_qty=shares,
        )
        if bool(self._settings.ENABLE_KELLY_SIZING) and sizing_mode == "kelly_atr":
            sn = int(float(k_stats.get("sample_n", 0.0)))
            pre = float(risk_pct_pre_kelly)
            mult = float(eff_risk_pct) / pre if pre > 1e-15 else 1.0
            wr_val = float(k_stats.get("win_rate", 0.0))
            aw = float(k_stats.get("avg_win") or 0.0)
            al = float(k_stats.get("avg_loss") or 0.0)
            pay = (aw / al) if al > 1e-12 else 0.0
            self._log.info(
                "event=kelly_sizing_applied symbol=%s enabled=true multiplier=%.8f sample_size=%s "
                "win_rate=%.6f payoff_ratio=%.6f final_quantity=%.4f fallback_reason=n_a",
                symbol,
                mult,
                sn,
                wr_val,
                pay,
                float(shares),
                extra={"symbol": symbol},
            )
        rationale = (
            f"mode={sizing_mode}"
            f"|clamped_by={clamping_reason}"
            f"|conv_mult={conviction_risk_multiplier:.4f}"
            f"|anti_mart={am_mult:.4f}"
            f"|eff_risk_pct={eff_risk_pct:.6f}"
            f"|risk_mode={risk_mode}"
            f"|recent={recent_trade_hint}"
        )
        return PositionSize(
            symbol=symbol,
            shares=shares,
            notional=proposed_notional,
            entry_price=entry_price,
            stop_distance=stop_distance,
            risk_budget=risk_budget,
            capital_base=capital_base,
            conviction_risk_multiplier=float(conviction_risk_multiplier),
            effective_risk_pct=float(eff_risk_pct),
            rationale=rationale,
            risk_mode=risk_mode,
            anti_martingale_multiplier=float(am_mult),
            sizing_mode=sizing_mode,
            kelly_win_rate=k_stats.get("win_rate"),
            kelly_avg_win=k_stats.get("avg_win"),
            kelly_avg_loss=k_stats.get("avg_loss"),
            kelly_profit_factor=k_stats.get("profit_factor"),
            kelly_fraction=fk,
            kelly_modified=mod_k,
            risk_pct_before_kelly_caps=float(risk_pct_pre_kelly),
        )

    def _log_sizing(
        self,
        *,
        symbol: str,
        capital_base: float,
        risk_budget: float,
        atr: float,
        stop_distance: float,
        raw_shares: float,
        clamping_reason: str,
        final_shares: float,
        outcome: str,
        conviction_risk_multiplier: float,
        effective_risk_pct: float,
        risk_mode: str = "normal",
        anti_martingale_multiplier: float = 1.0,
        recent_trade_hint: str = "",
        sizing_mode: str = "flat_atr",
        risk_pct_before_kelly_caps: float | None = None,
        risk_pct_after_kelly: float | None = None,
        kelly_win_rate: float | None = None,
        kelly_avg_win: float | None = None,
        kelly_avg_loss: float | None = None,
        kelly_profit_factor: float | None = None,
        kelly_fraction: float | None = None,
        kelly_modified: float | None = None,
        final_qty: float | None = None,
    ) -> None:
        self._log.info(
            "event=position_sizing outcome=%s sizing_mode=%s symbol=%s risk_mode=%s "
            "capital_base=%.4f risk_budget=%.6f eff_risk_pct=%.6f risk_pct_pre_kelly=%s risk_pct_post_kelly=%s "
            "win_rate=%s avg_win=%s avg_loss=%s profit_factor=%s kelly_frac=%s modified_kelly=%s "
            "anti_martingale_multiplier=%.4f conv_mult=%.4f recent_trade_outcomes=%s "
            "atr=%.6f stop_distance=%.6f raw_shares=%.4f clamping_reason=%s final_shares=%.4f final_qty=%s "
            "reason=%s",
            outcome,
            sizing_mode,
            symbol,
            risk_mode,
            capital_base,
            risk_budget,
            effective_risk_pct,
            f"{risk_pct_before_kelly_caps:.8f}" if risk_pct_before_kelly_caps is not None else "n_a",
            f"{risk_pct_after_kelly:.8f}" if risk_pct_after_kelly is not None else "n_a",
            f"{kelly_win_rate:.6f}" if kelly_win_rate is not None else "n_a",
            f"{kelly_avg_win:.6f}" if kelly_avg_win is not None else "n_a",
            f"{kelly_avg_loss:.6f}" if kelly_avg_loss is not None else "n_a",
            f"{kelly_profit_factor:.6f}" if kelly_profit_factor is not None else "n_a",
            f"{kelly_fraction:.6f}" if kelly_fraction is not None else "n_a",
            f"{kelly_modified:.6f}" if kelly_modified is not None else "n_a",
            anti_martingale_multiplier,
            conviction_risk_multiplier,
            recent_trade_hint or "n_a",
            atr,
            stop_distance,
            raw_shares,
            clamping_reason,
            final_shares,
            f"{final_qty:.4f}" if final_qty is not None else f"{final_shares:.4f}",
            outcome,
            extra={"symbol": symbol},
        )

    def _skip(
        self,
        symbol: str,
        reason: str,
        *,
        entry_price: float = 0.0,
        stop_distance: float = 0.0,
        risk_budget: float = 0.0,
        capital_base: float = 0.0,
        conviction_risk_multiplier: float = 1.0,
        effective_risk_pct: float = 0.0,
        anti_martingale_multiplier: float = 1.0,
        risk_mode: str = "normal",
    ) -> PositionSize:
        self._log.info(
            "Sizing skip for %s: %s",
            symbol,
            reason,
            extra={"symbol": symbol},
        )
        return PositionSize(
            symbol=symbol,
            shares=0.0,
            notional=0.0,
            entry_price=entry_price,
            stop_distance=stop_distance,
            risk_budget=risk_budget,
            capital_base=capital_base,
            conviction_risk_multiplier=float(conviction_risk_multiplier),
            effective_risk_pct=float(effective_risk_pct),
            rationale=reason,
            skipped_reason=reason,
            risk_mode=risk_mode,
            anti_martingale_multiplier=float(anti_martingale_multiplier),
        )
