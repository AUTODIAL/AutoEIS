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

    # Throw error for non-existent module if error=True
    with pytest.raises(Exception):
        julia_helpers.import_package("NonExistentModule", Main, error=True)
    # Otherwise, return None
    ref = julia_helpers.import_package("NonExistentModule", Main, error=False)
    assert ref is None


def test_import_backend():
    # Import backend with Julia runtime as argument
    Main = julia_helpers.init_julia()
    ec = julia_helpers.import_backend(Main)
    assert hasattr(ec, "circuit_evolution")
    # Import backend without Julia runtime as argument
    ec = julia_helpers.import_backend()
    assert hasattr(ec, "circuit_evolution")
