# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# from datetime import datetime
from typing import OrderedDict

from omegaconf import OmegaConf
from professor.vela.utils import get_object_reference, initialize_function_calls_in_parameters

import logging

LOGGER = logging.getLogger(__name__)


def test_get_object_reference_top_level_module():
    target = "collections.OrderedDict"
    func = get_object_reference(target)
    res = func()
    assert isinstance(res, OrderedDict)


def test_get_object_reference_nested_module():
    target = "urllib.parse.urljoin"
    func = get_object_reference(target)
    res = func("http://www.cwi.nl/%7Eguido/Python.html", "FAQ.html")
    expected = "http://www.cwi.nl/%7Eguido/FAQ.html"
    assert res == expected


def test_initialize_function_calls_in_parameters_no_args():
    example_params_yaml = """
    example_dict:
      invocation:
        target: collections.OrderedDict
    """
    d = OmegaConf.create(example_params_yaml)
    ans = initialize_function_calls_in_parameters(d)
    assert "example_dict" in ans and isinstance(ans["example_dict"], OrderedDict)


def test_initialize_function_calls_in_parameters_with_args():
    example_params_yaml = """
    example_dict:
      invocation:
        target: urllib.parse.urljoin
        params:
          base: http://www.cwi.nl/%7Eguido/Python.html
          url: FAQ.html
    """
    d = OmegaConf.create(example_params_yaml)
    ans = "http://www.cwi.nl/%7Eguido/FAQ.html"
    expected = {"example_dict": ans}
    assert initialize_function_calls_in_parameters(d) == expected


def test_initialize_function_calls_in_parameters_mixed():
    example_params_yaml = """
    example_dict:
      invocation:
        target: urllib.parse.urljoin
        params:
          base: http://www.cwi.nl/%7Eguido/Python.html
          url: FAQ.html
    normal_arg: 2
    """
    d = OmegaConf.create(example_params_yaml)
    call_target_ans = "http://www.cwi.nl/%7Eguido/FAQ.html"
    expected = {"example_dict": call_target_ans, "normal_arg": 2}
    assert initialize_function_calls_in_parameters(d) == expected


# def test_initialize_function_calls_in_parameters_empty():
#     example_params_yaml = """
#     example_dict:
#       object: urllib.parse.urljoin
#       base: http://www.cwi.nl/%7Eguido/Python.html
#       url: FAQ.html
#     first_layer:
#       object: socket.gethostname
#     """
#     d = OmegaConf.create(example_params_yaml)
#     ans = "http://www.cwi.nl/%7Eguido/FAQ.html"
#     res = initialize_function_calls_in_parameters(d)
#     assert res["example_dict"] == ans
#     assert isinstance(res["first_layer"], str)
