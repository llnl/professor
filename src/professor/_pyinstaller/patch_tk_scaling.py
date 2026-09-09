#!/usr/bin/env python3
"""Enable Tk scaling 1.0 in a PyInstaller onedir bundle."""

from pathlib import Path
import argparse
import re
import sys


SCALING_BLOCK = """# Set a consistent Tk scale for the frozen application.
if {[llength [info commands tk]]} {
    tk scaling 1.0
}
"""


def patch_bundle(root: Path) -> bool:
    tk_script = root / "_internal" / "_tk_data" / "tk.tcl"

    if not tk_script.is_file():
        raise FileNotFoundError(
            f"Could not find bundled Tk script: {tk_script}\nPass the root directory of the PyInstaller onedir bundle."
        )

    content = tk_script.read_text(encoding="utf-8")

    if "tk scaling 1.0" in content:
        return False

    match = re.search(r"(?m)^package require -exact Tk[^\n]*$", content)
    if match is None:
        raise RuntimeError(f"Unexpected Tk script format: {tk_script}")

    insertion = "\n\n" + SCALING_BLOCK.rstrip("\n")
    content = content[: match.end()] + insertion + content[match.end() :]
    tk_script.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch a PyInstaller bundle to set Tk scaling to 1.0.")
    parser.add_argument("bundle_root", type=Path, help="PyInstaller onedir root")
    args = parser.parse_args()

    try:
        changed = patch_bundle(args.bundle_root.resolve())
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if changed:
        print("Patched bundled Tk scaling to 1.0.")
    else:
        print("Tk scaling patch already present; no changes made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
