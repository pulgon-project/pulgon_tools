# Copyright 2023-2026 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import argparse
from typing import Optional, Sequence, Tuple

import numpy as np


class RawDescriptionDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve epilog formatting while showing argument default values."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_string = action.help or ""
        defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
        has_default = (
            action.default is not None
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in help_string
        )
        should_show_default = (
            action.option_strings or action.nargs in defaulting_nargs
        )
        if has_default and should_show_default:
            help_string += " (default: %(default)s)"
        return help_string


def parse_sequence(value: object, name: str) -> Sequence:
    """Validate that an argparse value is a sequence."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    return value


def parse_symbols(
    value: object, expected_count: Optional[int] = None
) -> Tuple[str, ...]:
    """Validate a sequence of element symbols."""
    symbols = parse_sequence(value, "--symbol")
    if not symbols or not all(isinstance(symbol, str) for symbol in symbols):
        raise ValueError("--symbol must contain element symbols.")
    if expected_count is not None and len(symbols) != expected_count:
        raise ValueError(
            f"--symbol must contain exactly {expected_count} element symbols."
        )
    return tuple(symbols)


def parse_positive_int_pair(value: object, name: str) -> Tuple[int, int]:
    """Validate two non-negative integers with a nonzero sum."""
    numbers = parse_sequence(value, name)
    if len(numbers) != 2 or not all(
        isinstance(number, int) for number in numbers
    ):
        raise ValueError(f"{name} must be two integers.")
    first, second = numbers
    if first < 0 or second < 0 or first + second <= 0:
        raise ValueError(f"{name} values must be non-negative and non-zero.")
    return first, second


def parse_motif_groups(
    values: Sequence[float], default: Sequence[Sequence[float]]
) -> np.ndarray:
    """Return motif coordinates grouped as rows of [r, phi, z]."""
    if values is None:
        return np.array(default, dtype=float)
    if len(values) % 3 != 0:
        raise ValueError(
            "--motif expects coordinates in groups of three: r phi z."
        )
    return np.array(values, dtype=float).reshape(-1, 3)
