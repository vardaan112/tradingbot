"""Launcher: run ``python main.py`` from the repository root.

The script lives at repo root named ``main.py`` for operator convenience.
Importing a sibling ``main`` module would resolve to this file; we load the
canonical entrypoint object from ``src/main.py`` explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    """Delegate to ``src/main.py`` (package ``main`` on ``PYTHONPATH=src``)."""

    root = Path(__file__).resolve().parent
    src_main = root / "src" / "main.py"
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    spec = importlib.util.spec_from_file_location("tradingbot_src_main", src_main)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load entrypoint module from {src_main}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
