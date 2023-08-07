import os
import sys

import pytest

sys.path.append(os.getcwd())
from main import Cn, U, dimino, sigmaV


@pytest.fixture(name="generators")
def fixture_point_group_generators():
    return [Cn(6), sigmaV(), U()]


def test_dimino(generators):
    assert len(dimino(generators)) == 24
