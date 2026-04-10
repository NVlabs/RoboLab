# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Decorators for function eval types (atomic, composite, etc.)
def atomic(fn):
    """Decorator to mark a function as an atomic condition that cannot be decomposed further."""
    fn.type = "atomic"
    return fn

def composite(fn):
    """Decorator to mark a function as a composite condition composed of multiple sub-conditions."""
    fn.type = "composite"
    return fn