# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import inspect


def check_required_params_available(params: dict, required_params: list):
    """
    Check that required parameters are given.

    Args:
        params (dict): _description_
        required_params (list): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    missing = [param for param in required_params if param not in params.keys()]
    if missing:
        raise ValueError(
            f"Missing required parameters: '{', '.join(missing)}' from param dict: {params}"
        )
    return True


def check_one_of_required_params_available(params: dict, one_of: list) -> str:
    """
    Check that exactly one of the required parameters or parameter groups is provided,
    and return which one was provided.

    - If an element in `one_of` is a string, that parameter must be present.
    - If an element is a list/tuple, all parameters in that group must be present together.

    Usage:
    params = {'a': 1, 'b': 2}
    one_of = ['x', ['a', 'b'], 'y']

    result = check_one_of_required_params_available(params, one_of)
    print(result)  # Output: ['a', 'b']
    """
    available = []

    for item in one_of:
        if isinstance(item, str):
            if item in params:
                available.append(item)
        elif isinstance(item, (list, tuple)):
            if all(param in params for param in item):
                available.append(item)
        else:
            raise TypeError(
                f"Invalid type in one_of: {type(item)}. Must be str, list, or tuple."
            )

    if len(available) == 0:
        raise ValueError(
            f"Missing required parameters: one of {one_of} must be provided! Param dict: {params}"
        )
    elif len(available) > 1:
        raise ValueError(
            f"Multiple parameters provided: only one of {one_of} must be provided! Provided: {available}"
        )

    return available[0]  # Return the single parameter or group that was provided


def filter_valid_params(func, param: dict = None) -> tuple[dict | dict]:
    """
    Filters a dictionary of parameters, keeping only those that are valid for the given function.
    Raises a warning if there are extra parameters that the function does not accept.

    Args:
        func (_type_): The function to check parameters against.

        param (dict, optional): Dictionary of parameters to filter.

    Returns:
        tuple:
            - valid_params (dict): Parameters accepted by the function.
            - invalid_params (dict): Parameters not accepted by the function.
    """
    if param is None:
        param = {}

    # Get valid parameter names for the function
    valid_keys = inspect.signature(func).parameters.keys()

    # Find invalid keys
    valid_params = {k: param[k] for k in valid_keys if k in param}
    invalid_params = {k: param[k] for k in param if k not in valid_keys}

    # Return only valid parameters
    return valid_params, invalid_params
