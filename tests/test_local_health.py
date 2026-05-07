"""Laptop resource probes (psutil mocked; no hardware dependency in CI)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from config.constants import LOGGER_APP
from config.settings import Settings
from utils.local_health import check_local_resource_risk, get_local_resource_status


class _Batt:
    def __init__(self, percent: float, power_plugged: bool) -> None:
        self.percent = percent
        self.power_plugged = power_plugged


def _mk_settings(factory: Callable[..., Settings], **kwargs: object) -> Settings:
    return factory(**kwargs)


def _noop_fourMocks(mb: MagicMock, mc: MagicMock, mv: MagicMock, md: MagicMock) -> None:
    _ = (mb, mc, mv, md)


@patch("utils.local_health.psutil.sensors_battery", return_value=_Batt(95.0, True))
@patch("utils.local_health.psutil.cpu_percent", return_value=12.0)
@patch("utils.local_health.psutil.virtual_memory", return_value=MagicMock(percent=40.0))
@patch("utils.local_health.shutil.disk_usage", return_value=MagicMock(free=50 * 1024**3))
def test_plugged_in_safe_case(
    mock_disk: MagicMock,
    mock_vm: MagicMock,
    mock_cpu: MagicMock,
    mock_batt: MagicMock,
    make_settings_factory,
) -> None:
    _noop_fourMocks(mock_disk, mock_vm, mock_cpu, mock_batt)
    st = get_local_resource_status()
    assert st.battery_percent == 95.0
    assert st.power_plugged is True
    warns = check_local_resource_risk(_mk_settings(make_settings_factory))
    codes = [w.code for w in warns]
    assert "on_battery_unplugged" not in codes


@patch("utils.local_health.psutil.sensors_battery", return_value=_Batt(90.0, False))
@patch("utils.local_health.psutil.cpu_percent", return_value=12.0)
@patch("utils.local_health.psutil.virtual_memory", return_value=MagicMock(percent=40.0))
@patch("utils.local_health.shutil.disk_usage", return_value=MagicMock(free=50 * 1024**3))
def test_unplugged_emits_warning(
    mock_disk: MagicMock,
    mock_vm: MagicMock,
    mock_cpu: MagicMock,
    mock_batt: MagicMock,
    make_settings_factory,
) -> None:
    _noop_fourMocks(mock_disk, mock_vm, mock_cpu, mock_batt)
    codes = [
        w.code for w in check_local_resource_risk(
            _mk_settings(make_settings_factory, WARN_ON_LOW_BATTERY=False),
        )
    ]
    assert "on_battery_unplugged" in codes


@patch("utils.local_health.psutil.sensors_battery", return_value=_Batt(12.0, True))
@patch("utils.local_health.psutil.cpu_percent", return_value=12.0)
@patch("utils.local_health.psutil.virtual_memory", return_value=MagicMock(percent=40.0))
@patch("utils.local_health.shutil.disk_usage", return_value=MagicMock(free=50 * 1024**3))
def test_low_battery_emits_warning(
    mock_disk: MagicMock,
    mock_vm: MagicMock,
    mock_cpu: MagicMock,
    mock_batt: MagicMock,
    make_settings_factory,
) -> None:
    _noop_fourMocks(mock_disk, mock_vm, mock_cpu, mock_batt)
    codes = [
        w.code
        for w in check_local_resource_risk(
            _mk_settings(make_settings_factory, LOW_BATTERY_THRESHOLD_PCT=20),
        )
    ]
    assert any(c.startswith("battery_below_") for c in codes)


@patch("utils.local_health.psutil.sensors_battery", return_value=None)
@patch("utils.local_health.psutil.cpu_percent", return_value=12.0)
@patch("utils.local_health.psutil.virtual_memory", return_value=MagicMock(percent=40.0))
@patch("utils.local_health.shutil.disk_usage", return_value=MagicMock(free=50 * 1024**3))
def test_missing_battery_reports_unavailable_only(
    mock_disk: MagicMock,
    mock_vm: MagicMock,
    mock_cpu: MagicMock,
    mock_batt: MagicMock,
    make_settings_factory,
) -> None:
    _noop_fourMocks(mock_disk, mock_vm, mock_cpu, mock_batt)
    warns = check_local_resource_risk(_mk_settings(make_settings_factory))
    assert [w.code for w in warns] == ["battery_power_info_unavailable"]


def test_log_local_resource_emits_structured(
    caplog: pytest.LogCaptureFixture,
    make_settings_factory,
) -> None:
    """Integration-style log smoke (real psutil acceptable; level may vary by machine)."""

    import logging

    from utils.local_health import log_local_resource_check  # noqa: PLC0415

    settings = _mk_settings(make_settings_factory)
    tlog = logging.getLogger("local_health_integration_test")
    tlog.propagate = True

    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    try:
        with caplog.at_level(logging.INFO):
            log_local_resource_check(settings, log=tlog)
    finally:
        root.setLevel(old_level)

    assert any(
        r.getMessage().startswith("event=local_resource_check") for r in caplog.records
    )
