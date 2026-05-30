# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import sys
from collections.abc import Mapping
import logging
import importlib.util
from importlib import import_module
from typing import Any
from professor.vela.config import Keys

logger: logging.Logger = logging.getLogger(__name__)


def get_object_reference(target: str) -> Any:
    """
    Returns a reference to an object based on the python package location provided. The
    returned object can then be instantiated.

    Example:

    ```
    target = "collections.OrderedDict"
    func = get_object_reference(target)
    func()
    ```

    :param target: Dot separated path of the object like `package.module.Object`
    :return: Reference to the object
    """
    target_list = target.rsplit(".", 1)
    module_name, object_name = target_list[0], target_list[1]
    logger.debug(f"module_name: {module_name}")
    logger.debug(f"object_name: {object_name}")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        logger.debug(f"Could not find: {module_name}\nException: {e}")
        logger.debug("Attempting relative import!")
        file_path = module_name.replace(".", "/") + ".py"
        logger.debug(f"type: {type(file_path)}")
        logger.debug(f"Trying this file to import: {file_path}")
        spec = importlib.util.spec_from_file_location(
            module_name,
            location=file_path,
        )
        # these two function can soft-fail which is annoying
        # check if spec is none, if none return error
        # as this will not import the module
        if spec is None:
            raise ImportError("Can not find {file_path}")
        logger.debug(f"Found spec: {spec}")
        logger.debug(f"type(spec): {type(spec)}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return getattr(module, object_name)


def initialize_function_calls_in_parameters(params: dict[str, Any]) -> dict[str, Any]:
    res = {}
    for p in params:
        curr = params[p]
        if isinstance(curr, Mapping) and Keys.INVOCATION.value in curr:
            obj = curr[Keys.INVOCATION.value]
            target = obj[Keys.TARGET.value]
            func = get_object_reference(target)
            if Keys.PARAMS.value in obj and Keys.PARAMS.value is not None:
                res[p] = func(**curr[Keys.INVOCATION.value][Keys.PARAMS.value])
            else:
                res[p] = func()
        else:
            res[p] = curr
    return res
