"""Best-effort laptop resource probes (battery, CPU, memory, disk)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import Optional

import psutil

from config.constants import LOGGER_APP
from config.settings import Settings

_LOG = logging.getLogger(LOGGER_APP)


@dataclass(frozen=True)
class LocalResourceStatus:
    battery_percent: Optional[float]
    power_plugged: Optional[bool]
    cpu_percent: Optional[float]
    memory_percent: Optional[float]
    disk_free_gb: Optional[float]


class LocalResourceWarning:
    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        self.code = code

    def __str__(self) -> str:
        return self.code


def get_local_resource_status() -> LocalResourceStatus:
    """Collect machine metrics without raising."""

    pct_bat: Optional[float] = None
    plugged: Optional[bool] = None
    cpu_p: Optional[float] = None
    mem_pct: Optional[float] = None
    disk_gb: Optional[float] = None
    try:
        bat = psutil.sensors_battery()
        if bat is not None:
            pct_bat = float(bat.percent)
            plugged = bool(bat.power_plugged)
    except Exception:
        pct_bat = None
        plugged = None

    try:
        cpu_p = float(psutil.cpu_percent(interval=0.05))
        vm = psutil.virtual_memory()
        mem_pct = float(vm.percent)
    except Exception:
        cpu_p = None
        mem_pct = None

    try:
        du = shutil.disk_usage("/")
        disk_gb = float(du.free) / (1024.0**3)
    except Exception:
        try:
            du = shutil.disk_usage(".")
            disk_gb = float(du.free) / (1024.0**3)
        except Exception:
            disk_gb = None

    return LocalResourceStatus(
        battery_percent=pct_bat,
        power_plugged=plugged,
        cpu_percent=cpu_p,
        memory_percent=mem_pct,
        disk_free_gb=disk_gb,
    )


def check_local_resource_risk(settings: Settings) -> list[LocalResourceWarning]:
    """Operational warnings derived from probes + settings."""

    codes: list[str] = []
    st = get_local_resource_status()
    thr = float(settings.LOW_BATTERY_THRESHOLD_PCT)

    if st.battery_percent is not None:
        if settings.WARN_ON_LOW_BATTERY and st.battery_percent < thr:
            codes.append(f"battery_below_{settings.LOW_BATTERY_THRESHOLD_PCT}_pct:{st.battery_percent:.1f}")
        if st.power_plugged is False:
            codes.append("on_battery_unplugged")
    else:
        codes.append("battery_power_info_unavailable")

    if settings.REQUIRE_POWER_FOR_LOCAL_LIVE and settings.can_submit_real_orders:
        if st.power_plugged is False:
            codes.append("require_power_live_unplugged")
        if st.battery_percent is not None and st.battery_percent < thr:
            codes.append("require_power_live_low_battery")

    return [LocalResourceWarning(c) for c in codes]


def log_local_resource_check(settings: Settings, log: logging.Logger | None = None) -> LocalResourceStatus:
    """Structured ``event=local_resource_check`` line."""

    logger = log or _LOG
    st = get_local_resource_status()
    warns_objs = check_local_resource_risk(settings)
    warn_codes = [w.code for w in warns_objs]
    pct = "n_a" if st.battery_percent is None else f"{st.battery_percent:.1f}"
    plug = "n_a" if st.power_plugged is None else str(st.power_plugged).lower()
    cpu = "n_a" if st.cpu_percent is None else f"{st.cpu_percent:.1f}"
    mem = "n_a" if st.memory_percent is None else f"{st.memory_percent:.1f}"
    disk = "n_a" if st.disk_free_gb is None else f"{st.disk_free_gb:.2f}"
    warn_txt = ",".join(warn_codes) if warn_codes else ""
    benign_only = warn_codes == ["battery_power_info_unavailable"]
    lvl = logging.INFO if benign_only else (logging.WARNING if warn_codes else logging.INFO)
    logger.log(
        lvl,
        "event=local_resource_check battery_percent=%s power_plugged=%s cpu_percent=%s "
        "memory_percent=%s disk_free_gb=%s warnings=%s",
        pct,
        plug,
        cpu,
        mem,
        disk,
        warn_txt if warn_txt else "none",
    )
    return st


def log_startup_local_health(settings: Settings, log: logging.Logger) -> None:
    """Call once near process start."""

    log_local_resource_check(settings, log=log)


__all__ = [
    "LocalResourceStatus",
    "LocalResourceWarning",
    "check_local_resource_risk",
    "get_local_resource_status",
    "log_local_resource_check",
    "log_startup_local_health",
]
