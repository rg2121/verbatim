"""Setup every test shares."""

import pytest


@pytest.fixture(autouse=True)
def no_default_reranker(monkeypatch):
    """The suite is offline, so front ends search without the reranker's model.

    A test of the default turns it back on itself.
    """
    from verbatim import config

    monkeypatch.setattr(config, "RERANKER", "off")


@pytest.fixture(autouse=True)
def no_default_repair(monkeypatch):
    """Scripted models answer once per call, so front ends run without the repair.

    A test of the default turns it back on itself.
    """
    from verbatim import config

    monkeypatch.setattr(config, "REFLECT", False)
