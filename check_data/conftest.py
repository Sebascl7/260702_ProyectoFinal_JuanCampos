"""check_data/conftest.py — Fixtures para los tests de validación de datos."""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def df_train():
    """Carga el conjunto de entrenamiento para los tests."""
    path = Path("data/attrition_train.csv")
    if not path.exists():
        pytest.skip("data/attrition_train.csv no encontrado — ejecuta los pasos 1-3 primero")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def df_test():
    """Carga el conjunto de test para los tests."""
    path = Path("data/attrition_test.csv")
    if not path.exists():
        pytest.skip("data/attrition_test.csv no encontrado — ejecuta los pasos 1-3 primero")
    return pd.read_csv(path)
