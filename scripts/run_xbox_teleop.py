#!/usr/bin/env python3
"""Compatibility wrapper for the installed hil-pusht-teleop command."""

from __future__ import annotations

from hil_pusht_env.cli.teleop import main


if __name__ == "__main__":
    raise SystemExit(main())
