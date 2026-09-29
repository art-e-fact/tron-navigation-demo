import pytest

try:
    from artefacts_toolkit.config import get_artefacts_params
except ImportError:
    get_artefacts_params = None


@pytest.fixture
def artefacts_params() -> dict:
    """The scenario's params in artefacts.yaml; {} when run with plain pytest."""
    if get_artefacts_params is None:
        return {}
    try:
        return get_artefacts_params()
    except Exception:
        return {}
