import pytest

from autoeis import julia_helpers


def test_init_julia():
    Main = julia_helpers.init_julia()
    assert Main.seval("1+1") == 2


def test_import_julia_modules():
    Main = julia_helpers.init_julia()

    # Ensure installed modules can be imported
    ec = julia_helpers.import_package("EquivalentCircuits", Main)
    assert hasattr(ec, "circuit_evolution")

    # Throw error for non-existent module
    with pytest.raises(Exception):
        julia_helpers.import_package("NonExistentModule", Main)


def test_import_backend():
    Main = julia_helpers.init_julia()
    ec = julia_helpers.import_backend(Main)
    assert hasattr(ec, "circuit_evolution")
