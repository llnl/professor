"""PyInstaller entry point for the Professor Dash application."""

from __future__ import annotations

import multiprocessing


def main() -> None:
    """Start the Dash application with frozen multiprocessing support."""
    multiprocessing.freeze_support()

    from professor.gui.dash_gui import main as dash_main

    dash_main()


if __name__ == "__main__":
    main()
