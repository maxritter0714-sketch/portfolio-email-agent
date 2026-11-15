import pytest

import market


@pytest.fixture(autouse=True)
def empty_price_caches():
    """Every test starts without prices cached by an earlier test."""
    market.clear_price_caches()
