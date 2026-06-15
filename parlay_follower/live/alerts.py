"""Alert sinks. Console by default; extend with push (ntfy.sh, Pushover, etc.)."""
from __future__ import annotations

import sys
import time


def console_alert(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


def loud_console_alert(message: str) -> None:
    bar = "=" * 72
    print(f"\a\n{bar}\n{time.strftime('%H:%M:%S')}  {message}\n{bar}\n",
          file=sys.stdout, flush=True)
