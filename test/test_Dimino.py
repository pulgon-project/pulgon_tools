import pytest

from src.mypkg.generate_structures import Cn, U, dimino, sigmaV


@pytest.fixture(name="generators")
def fixture_point_group_generators():
    return [Cn(6), sigmaV(), U()]


def test_dimino(generators):
    assert len(dimino(generators)) == 24
