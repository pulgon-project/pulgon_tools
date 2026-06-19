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
