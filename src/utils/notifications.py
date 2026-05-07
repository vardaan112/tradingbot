"""Cross-platform user notifications (desktop / headless fallbacks)."""

from __future__ import annotations

import logging
import subprocess
import sys

from config.constants import LOGGER_APP

_LOG = logging.getLogger(LOGGER_APP)


def notify_user(title: str, message: str, severity: str = "warning") -> None:
    """Best-effort popup / OS notification; never raises."""

    try:
        try:
            from plyer import notification  # type: ignore[import-untyped]

            notification.notify(title=title[:80], message=message[:400], timeout=8)
            return
        except Exception:
            pass

        if sys.platform == "darwin":
            esc_t = title.replace("\\", "\\\\").replace('"', '\\"')
            esc_m = message.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{esc_m}" with title "{esc_t}"',
                ],
                capture_output=True,
                check=False,
                timeout=15,
            )
            return

        if sys.platform == "win32":
            try:
                import winsound

                winsound.Beep(880, 200)
            except Exception:
                pass
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(0, message[:1024], title[:128], 0x30)
            except Exception:
                pass
            return

        subprocess.run(
            [
                "notify-send",
                "-u",
                "critical" if severity == "critical" else "normal",
                title[:128],
                message[:512],
            ],
            capture_output=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("event=notification_failed title=%s severity=%s error=%s", title, severity, exc)


__all__ = ["notify_user"]
