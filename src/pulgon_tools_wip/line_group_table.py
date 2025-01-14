import logging
import re

import numpy as np
from ipdb import set_trace


def get_family_Num_from_sym_symbol(trans_sym, rota_sym):
    """
    Determine the family number based on translational and rotational symmetry symbols.

    This function maps the translational symmetry (`trans_sym`) and rotational
    symmetry (`rota_sym`) symbols to a family number using a predefined family table
    (`family_map`).

    Args:
        trans_sym (str): The translational symmetry symbol.
        rota_sym (str): The rotational symmetry symbol.

    Returns:
        int or None: The line group family number,
        or None if the symbols cannot be identified.

    Raises:
        None: This function does not raise any exceptions directly, but logs an
        error message if the symbols cannot be identified.
    """

    family_map = {
        "T": {
            "S": 2,
            "C": {"h": 3, "v": 6, "digit": 1},
            "D": {"d": 9, "h": 11},
        },
        "T'": {
            "S": 10,
            "C": {"h": 12, "digit": 7},
        },
        "TN": {
            "TQ": {
                "C": 1,
                "D": 5,
                "S": 4,
            },
            "T2n": {
                "C": {"h": 4, "v": 8},
                "D": 13,
            },
        },
    }

    def log_error():
        logging.error("Can't identify this symbol %s %s", trans_sym, rota_sym)
        return None

    def resolve_value(sym_map, key, suffix=None):
        """Helper to resolve the value from the symbol map."""
        value = sym_map.get(key)
        if isinstance(value, dict) and suffix:
            return value.get(suffix)
        return value

    if trans_sym.startswith("T'"):
        sym_map = family_map["T'"]
    elif trans_sym == "T":
        sym_map = family_map["T"]
    elif trans_sym.startswith("T"):
        sym_map = family_map["TN"]
        num_trans = float(re.findall(r"T(.*?)\(", trans_sym)[0])
        num_rots = float(re.findall(r"\d+", rota_sym)[0])
        sym_map = (
            sym_map["T2n"]
            if np.isclose(num_trans, 2 * num_rots, 1e-4)
            else sym_map["TQ"]
        )

        if rota_sym[0] == "S" and not np.isclose(num_trans, num_rots, 1e-4):
            return log_error()

    else:
        return log_error()

    # Resolve the value based on rota_sym
    key = rota_sym[0]
    suffix = rota_sym[-1] if len(rota_sym) > 1 else None
    value = resolve_value(sym_map, key, suffix)

    if value is not None:
        return value
    if key == "C" and suffix.isdigit() and "digit" in sym_map.get(key, {}):
        return sym_map[key]["digit"]

    return log_error()
