"""Setup every test shares."""

import pytest


@pytest.fixture(autouse=True)
def no_default_reranker(monkeypatch):
    """The suite is offline, so front ends search without the reranker's model.

    A test of the default turns it back on itself.
    """
    from verbatim import config

    monkeypatch.setattr(config, "RERANKER", "off")
