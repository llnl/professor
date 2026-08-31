# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from tests.mltrainer_integration_paths import dataset_path


def test_dataset_path_uses_lustre1_on_tioga():
    assert dataset_path("dataset", hostname="tioga") == "/p/lustre1/jekel1/data/dataset"


def test_dataset_path_uses_lustre5_on_tuolumne_nodes():
    assert dataset_path("dataset", hostname="tuolumne1046") == "/p/lustre5/jekel1/data/dataset"
    assert dataset_path("dataset", hostname="tuolumne.llnl.gov") == "/p/lustre5/jekel1/data/dataset"
