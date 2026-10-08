import pytest
from src.data_processing import DEFAULT_DATA, load_dataset


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(DEFAULT_DATA)
