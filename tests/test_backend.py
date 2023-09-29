import pytest

from autoeis import julia_helpers


def test_init_julia():
    Main = julia_helpers.init_julia()
    assert Main.eval("1+1") == 2


def test_import_julia_modules():
    Main = julia_helpers.init_julia()

    # Ensure installed modules can be imported
    julia_helpers.import_package(Main, "EquivalentCircuits")

    # Throw error for non-existent module
    with pytest.raises(Exception):
        julia_helpers.import_package(Main, "NonExistentModule")
