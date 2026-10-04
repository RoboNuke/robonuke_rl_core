"""Registers the `gpu` marker for tests that need Isaac Sim."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs Isaac Sim and a GPU; run with `pytest -m gpu`"
    )
