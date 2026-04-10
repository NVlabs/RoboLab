# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import ABC, abstractmethod


class InferenceClient(ABC):
    @abstractmethod
    def __init__(self, args) -> None:
        """
        Initializes the client.
        """
        pass

    @abstractmethod
    def infer(self, obs, instruction, *, env_id=0) -> dict:
        """
        Does inference on observation and returns the final processed
        dictionary used to do inference.
        """

        pass

    @abstractmethod
    def reset(self):
        """
        Resets the client to start a new episode.
        """
        pass
