# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import importlib
import pkgutil
from functools import partial
from inspect import signature
from typing import Any, Callable

import robolab.core.utils.file_utils as file_utils
import robolab.core.utils.params_utils as params_utils


def get_callable_info(func: Callable) -> tuple[str, dict[str, Any]]:
    """
    Extract function name from a callable function, and return the function name and pre-filled arguments.
    """
    if isinstance(func, partial):
        func_name = func.func.__name__
        params = func.keywords
    else:
        func_name = func.__name__
        params = func.__dict__

    return func_name, params

def func_as_str(func: Callable) -> str:
    """
    Convert a func function to a string.
    """
    func_name, params = get_callable_info(func)
    param_str = ', '.join(f"{k}={v}" for k, v in params.items())
    return f"{func_name}({param_str})"

def load_callable_from_module(module_name: str | Any, name: str) -> Callable:
    """
    Load a callable (function or class) from a given module.

    Args:
        module_name (str): The module to import.
        name (str): The name of the callable object (function or class).

    Returns:
        Callable: The loaded callable object.

    Raises:
        ImportError: If the module or object cannot be found.
        TypeError: If the object is not callable.
    """
    try:
        if isinstance(module_name, str):
            module = importlib.import_module(module_name)
        else:
            module = module_name
        obj = getattr(module, name)
        if not callable(obj):
            raise TypeError(f"'{name}' in module '{module_name}' is not callable.")
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot load '{name}' from module '{module_name}': {e}")


def search_function_in_module(
    module, function_name: str, filename: str = None, function_params: dict = None
) -> Callable:
    """
    The function that searches through the module (folder) and looks for the appropriate function.
    If the function is not found in the appropriate folder, it raises a ValueError.

    Example usage:
        import module
        callable_function = search_function_in_module(module, "function_name", myclass.py)
        output = callable_function(**params)

    Args:
        module (imported module): _description_
        function_name (str): name of the function.
        filename (str, optional): The python file that contains the function. If none is provided, it will search through the entire folder to find the function with a matching name.Defaults to None.
        function_params (dict, optional): optional params to the function. if provided, it will return a partial callable function.

    Raises:
        ValueError: _description_

    Returns:
        Callable: the callable policy function.
    """
    func = None
    if filename is not None:
        if filename.endswith(".py"):
            filename = file_utils.get_filename_without_extension(filename)
        func = load_callable_from_module(
            module.__name__ + "." + filename, function_name
        )
    else:
        for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
            func = load_callable_from_module(
                module.__name__ + "." + modname, function_name
            )
            if func is not None:
                break
    if func is None:
        raise ValueError(
            f"No suitable function [function_name: '{function_name}', filename: '{filename}', module: {module.__name__}] found!"
        )

    if function_params is not None and isinstance(function_params, dict):
        func = partial(func, **function_params)

    return func


def load_callable_from_dict(
    config: dict, prefill=False
) -> tuple[Callable, list, dict, str]:
    """
    Dynamically load either a function or a class from a module based on the given configuration.

    Args:
        config (dict): A dictionary with the following structure:
            {
                "module": "module_name",
                "function": "function_name",
                "class": "ClassName",
                "args": [arg1, arg2],  # Optional for classes
                "kwargs": {"param1": value1, "param2": value2}  # Optional for both
            }
        prefill (bool): if True, prefills the class initialization or function with args and kwargs.

    Returns:
        tuple[Callable, list, dict, str]:
            If type is "function", returns the callable function along with any args, kwargs, type
            If type is "class", returns an instantiated class object along with any args, kwargs, type
    """
    module_name = config.get("module")
    args = config.get("args", [])
    kwargs = config.get("kwargs", {})

    params_utils.check_required_params_available(config, ["module"])
    callable_type = params_utils.check_one_of_required_params_available(
        config, ["function", "class"]
    )
    callable_name = config.get(callable_type)
    fcn = load_callable_from_module(module_name, callable_name)

    if prefill:
        fcn = prefill_callable(fcn, args, kwargs)

    return fcn, args, kwargs, callable_type


def prefill_callable(fcn: Callable | Any, args: list = None, kwargs: dict = None):
    if kwargs or args:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        fcn = partial(fcn, *args, **kwargs)
    return fcn


def verify_callable_args_supplied(func: Callable, params: dict) -> tuple[bool, str]:
    """
    Verify if all arguments are supplied to a callable function.
    """
    p = partial(func, **params)
    sig = signature(p.func)
    try:
        sig.bind_partial(*p.args, **(p.keywords or {}))
        return True, None
    except TypeError as e:
        message = f"Error: {e}. Function {func.__name__} expects {list(sig.parameters.keys())} but got {p.args} and {p.keywords}."
        return False, message