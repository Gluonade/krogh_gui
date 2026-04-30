"""Window-control helpers for the Krogh GUI application."""

from __future__ import annotations


class UIControlCoordinator:
    """Handles language refreshes and small control-state updates."""

    _STATE_LABELS = {
        "en": {
            "normoxia": "normoxia",
            "intermediate_oxygenation": "intermediate oxygenation",
            "low_oxygenation_approaching_critical": "low tissue oxygen approaching critical values",
            "hypoxia": "hypoxia",
            "profound_hypoxia": "profound tissue hypoxia",
        },
        "de": {
            "normoxia": "Normoxie",
            "intermediate_oxygenation": "intermediaere Oxygenierung",
            "low_oxygenation_approaching_critical": "niedrige Gewebeoxygenierung nahe kritischer Werte",
            "hypoxia": "Hypoxie",
            "profound_hypoxia": "ausgepraegte Gewebehypoxie",
        },
    }

    def format_oxygenation_state_label(self, language, state_name):
        language_key = "de" if str(language).lower() in {"de", "deutsch", "german"} else "en"
        labels = self._STATE_LABELS.get(language_key, self._STATE_LABELS["en"])
        return labels.get(state_name, str(state_name).replace("_", " "))

    def apply_language_selection(self, gui) -> None:
        selected_code = gui.language_name_to_code.get(gui.language_display_var.get(), gui.language_code)
        if selected_code == gui.language_code:
            return
        state = gui._capture_ui_state()
        gui.language_code = selected_code
        for child in gui.winfo_children():
            child.destroy()
        gui._build_ui()
        gui._restore_ui_state(state)

    def set_series_param_display(self, gui, field_key) -> None:
        gui.series_param_key = field_key
        gui.series_param_var.set(gui.series_param_key_to_display.get(field_key, field_key))

    def set_series_param2_display(self, gui, field_key) -> None:
        gui.series_param2_key = field_key
        gui.series_param2_var.set(gui.series_param_key_to_display.get(field_key, field_key))

    def set_publication_layout_display(self, gui, layout_key) -> None:
        gui.publication_layout_key = layout_key if layout_key in {"a4", "wide"} else "wide"
        if hasattr(gui, "publication_layout_key_to_display"):
            gui.publication_layout_var.set(
                gui.publication_layout_key_to_display.get(gui.publication_layout_key, gui.publication_layout_key)
            )
        else:
            gui.publication_layout_var.set(gui.publication_layout_key)

    def on_publication_layout_selected(self, gui) -> None:
        if hasattr(gui, "publication_layout_display_to_key"):
            gui.publication_layout_key = gui.publication_layout_display_to_key.get(
                gui.publication_layout_var.get(),
                "wide",
            )

    def toggle_series_dimension_inputs(self, gui) -> None:
        is_2d = gui.mode_var.get() == "series" and gui.series_dimension_var.get() == "2d"
        combo_state = "readonly" if is_2d else "disabled"
        entry_state = "normal" if is_2d else "disabled"
        if hasattr(gui, "series_param2_combo"):
            gui.series_param2_combo.config(state=combo_state)
        if hasattr(gui, "series_plot_mode_combo"):
            gui.series_plot_mode_combo.config(state=combo_state)
        if hasattr(gui, "publication_layout_combo"):
            layout_state = "readonly" if gui.mode_var.get() == "series" and gui.publication_mode_var.get() else "disabled"
            gui.publication_layout_combo.config(state=layout_state)
        for key in ("secondary_start_value", "secondary_end_value", "secondary_step_size"):
            entry = getattr(gui, "series_entries_by_key", {}).get(key)
            if entry is not None:
                entry.config(state=entry_state)


__all__ = ["UIControlCoordinator"]
