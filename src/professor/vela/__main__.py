# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
import sys

from professor import _name
from professor.vela import __version__
from professor.vela.constants import LOG_LEVELS


def parse_args() -> argparse.Namespace:
    print(_name)
    parser = argparse.ArgumentParser(
        prog="prof-vis",
        description=f"[v{__version__}] Run the prof-vis GUI with the specified model config",  # noqa E501
    )

    parser.add_argument(
        "cfgpath",
        help="the relative path of the model's yaml config",
    )

    parser.add_argument(
        "-c",
        "--cmap",
        default=None,
        help="""
        initial colormap (any matplotlib colormap)
        {default: magma}
        """,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--theme",
        default=None,
        choices=["light", "dark", "system"],
        help="specify which theme to use",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="info logging in console (key steps, inference time, fps)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug logging in console (more verbose than --info)",
    )
    parser.add_argument("-v", "--version", action="version", version=f"v{__version__}")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)
    print(f"[v{__version__}] Loading the GUI...this may take a minute")
    """
    This is not an ideal place for an import, but this import is not trivial. The user
    saves a lot of time when they improperly run `vela` (such as with no args)
    """
    from professor.vela.vela import Vela

    log = "warn"
    if args.debug:
        log = "debug"
    elif args.info:
        log = "info"

    format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    logging.basicConfig(format=format, datefmt="%H:%M:%S", level=LOG_LEVELS[log])

    Vela(args.cfgpath, colormap=args.cmap, theme=args.theme)


if __name__ == "__main__":
    sys.exit(main())
