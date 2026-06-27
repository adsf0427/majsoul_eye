"""Screen capture of the Mahjong Soul window (web client or native).

The user plays in their own browser/client (NOT Akagi's Playwright instance), so
we grab the OS window region rather than a Playwright page. Windows-first via
``pywin32`` + ``mss``; degrades to a full-monitor grab if the window can't be
located, and to a clear error if the libs are missing.

Returns frames as BGR ``numpy`` arrays (OpenCV convention).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

# Akagi's default Majsoul window-title pattern (settings.json window_selection).
DEFAULT_TITLE_PATTERN = r"雀魂|Majsoul|MahjongSoul|Jantama"


class ScreenGrabber:
    def __init__(self, title_pattern: str = DEFAULT_TITLE_PATTERN, monitor: int = 1):
        self.title_re = re.compile(title_pattern)
        self.monitor = monitor
        self._mss = None
        self._hwnd: Optional[int] = None

    # --- window location (Windows) -----------------------------------------

    def find_window(self) -> Optional[int]:
        try:
            import win32gui  # type: ignore
        except Exception:
            return None
        matches: list[int] = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and self.title_re.search(title):
                    matches.append(hwnd)

        win32gui.EnumWindows(_cb, None)
        self._hwnd = matches[0] if matches else None
        return self._hwnd

    def _client_rect(self) -> Optional[dict]:
        """Client-area rect of the located window, as an mss monitor dict."""
        if self._hwnd is None and self.find_window() is None:
            return None
        try:
            import win32gui  # type: ignore
            left, top, right, bottom = win32gui.GetClientRect(self._hwnd)
            x, y = win32gui.ClientToScreen(self._hwnd, (left, top))
            w, h = right - left, bottom - top
            if w <= 0 or h <= 0:
                return None
            return {"left": x, "top": y, "width": w, "height": h}
        except Exception:
            return None

    # --- grab ---------------------------------------------------------------

    def _ensure_mss(self):
        if self._mss is None:
            import mss  # type: ignore  (raises ImportError with a clear message if absent)
            self._mss = mss.mss()
        return self._mss

    def grab(self) -> Optional[np.ndarray]:
        """Grab the Majsoul window client area (or full monitor) as BGR ndarray."""
        sct = self._ensure_mss()
        region = self._client_rect() or sct.monitors[self.monitor]
        shot = sct.grab(region)
        # mss returns BGRA; drop alpha -> BGR (OpenCV convention)
        return np.asarray(shot)[:, :, :3]

    def close(self) -> None:
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None
