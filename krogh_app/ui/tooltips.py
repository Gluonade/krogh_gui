"""Tooltip widget helper extracted from the main GUI file."""

from __future__ import annotations

import tkinter as tk


class ToolTip:
    def __init__(self, widget, text_provider, delay_ms=350):
        self.widget = widget
        self.text_provider = text_provider
        self.delay_ms = delay_ms
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_schedule(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except (RuntimeError, tk.TclError):
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        text = self.text_provider() if callable(self.text_provider) else self.text_provider
        if not text or self._window is not None:
            return
        try:
            self._window = tk.Toplevel(self.widget)
            self._window.wm_overrideredirect(True)
            self._window.attributes("-topmost", True)
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._window.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                self._window,
                text=text,
                justify="left",
                wraplength=420,
                relief="solid",
                borderwidth=1,
                background="#fffde8",
                padx=8,
                pady=6,
            )
            label.pack()
        except (RuntimeError, tk.TclError):
            self._window = None

    def _hide(self, _event=None):
        self._cancel_schedule()
        if self._window is not None:
            try:
                self._window.destroy()
            except (RuntimeError, tk.TclError):
                pass
            self._window = None

    def destroy(self):
        self._hide()


__all__ = ["ToolTip"]
