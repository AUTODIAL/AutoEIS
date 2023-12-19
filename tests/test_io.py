import os

import autoeis as ae


def test_get_assets_path():
    path = ae.io.get_assets_path()
    assert os.path.exists(path)


def test_load_test_dataset():
    Z, freq = ae.io.load_test_dataset()
    assert Z.shape == freq.shape
    assert len(Z) == len(freq) > 0
