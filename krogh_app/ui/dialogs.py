"""Reusable simple dialog helpers for the Tkinter UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def show_scrolled_text_dialog(parent, title: str, content: str, button_text: str = "Close", geometry: str = "860x620"):
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.geometry(geometry)

    frame = ttk.Frame(dlg)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    text = tk.Text(frame, wrap="word")
    text.insert("1.0", content)
    text.config(state="disabled")
    text.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    scrollbar.pack(side="right", fill="y")
    text.config(yscrollcommand=scrollbar.set)

    ttk.Button(dlg, text=button_text, command=dlg.destroy).pack(pady=(0, 10))
    return dlg


__all__ = ["show_scrolled_text_dialog"]
