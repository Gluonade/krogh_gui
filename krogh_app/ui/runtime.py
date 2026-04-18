"""Runtime/status helpers for the Krogh GUI application."""

from __future__ import annotations

import threading
import tkinter as tk


class UIRuntimeCoordinator:
    """Keeps Tkinter thread-safe UI updates in one reusable place."""

    def call_on_ui_thread(self, gui, callback, *args, **kwargs) -> None:
        if getattr(gui, "_is_closing", False):
            return
        try:
            if threading.get_ident() == getattr(gui, "_main_thread_ident", None):
                callback(*args, **kwargs)
            else:
                gui.after(0, lambda: (not getattr(gui, "_is_closing", False)) and callback(*args, **kwargs))
        except (RuntimeError, tk.TclError):
            pass

    def append_output(self, gui, text) -> None:
        gui.output.insert("end", f"{text}\n")
        gui.output.see("end")

    def append_output_async(self, gui, text) -> None:
        self.call_on_ui_thread(gui, self.append_output, gui, text)

    def set_status(self, gui, text) -> None:
        if hasattr(gui, "status_var"):
            gui.status_var.set(text)

    def set_status_async(self, gui, text) -> None:
        self.call_on_ui_thread(gui, self.set_status, gui, text)

    def set_progress_running(self, gui, running: bool) -> None:
        if not hasattr(gui, "progress"):
            return
        if running:
            gui.progress.start(12)
        else:
            gui.progress.stop()


__all__ = ["UIRuntimeCoordinator"]
