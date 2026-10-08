# Presence of this file puts the repo root on sys.path, so degradation/schade.py
# can resolve `from utils import ...` when pytest collects degradation/.
import pytest


@pytest.fixture
def scaler():
    """One pinned price transform, shared by every test that builds an observation.

    Pinned to the 2024-25 market price cap, which is the cap the training split ends
    under. Tests that care about a different cap build their own PriceScaler.
    """
    from rl.features import PriceScaler

    return PriceScaler(17500.0)
