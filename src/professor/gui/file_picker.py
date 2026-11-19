"""Native file picker support for the Dash GUI."""

from __future__ import annotations

import multiprocessing
import sys
from multiprocessing.connection import Connection


def _select_file_path() -> str:
    """Request a configuration filename via a native file picker."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    if sys.platform == 'darwin':
        root.geometry("1x1+500+300") 
        root.update()
    else:
        root.withdraw()
        root.attributes("-topmost", True)
        root.focus_force()

    filetypes = [
        ("Config (*.yml, *.yaml)", "*.yml *.yaml"),
        ("Archive (*.zip, *.tar, *.tar.gz)", "*.zip *.tar *.tar.gz"),
        ("All", "*"),
    ]
    file_path = filedialog.askopenfilename(parent=root, title="Select Target Model", filetypes=filetypes)
    root.destroy()
    return file_path


def _get_file_path_worker(connection: Connection) -> None:
    try:
        connection.send((True, _select_file_path()))
    except Exception as exc:  # noqa: BLE001
        connection.send((False, str(exc)))
    finally:
        connection.close()


def get_file_path() -> str:
    """Open the Tk file picker in a separate process and return the chosen path.

    macOS requires Tkinter to run on the main thread, but Dash callbacks may run
    on a worker thread. The separate process gives the picker its own main thread.
    """
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_get_file_path_worker, args=(child_connection,))

    try:
        process.start()
    except BaseException:
        parent_connection.close()
        child_connection.close()
        raise

    child_connection.close()
    try:
        try:
            result = parent_connection.recv()
        except EOFError:
            result = None
    finally:
        parent_connection.close()
        process.join()
        process.close()

    if result is None:
        raise RuntimeError("The file picker process exited without returning a path.")

    success, value = result
    if not success:
        raise RuntimeError(f"Could not open the file picker: {value}")
    return value
