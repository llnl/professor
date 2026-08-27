# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dataset paths used by the mltrainer integration tests."""

import socket
from typing import Optional


def dataset_path(dataset_name: str, hostname: Optional[str] = None) -> str:
    """Return the appropriate shared filesystem path for a test dataset."""
    hostname = (hostname or socket.gethostname()).split(".", maxsplit=1)[0].lower()
    filesystem = "lustre5" if hostname.startswith("tuolumne") else "lustre1"
    return f"/p/{filesystem}/jekel1/data/{dataset_name}"
