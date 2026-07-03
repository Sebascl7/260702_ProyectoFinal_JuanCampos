"""
check_data/test_data.py — Validación automática del dataset con pytest.

Tests de schema, integridad, rangos y distribuciones para el dataset
de Abandono de Empleados.

Ejecutar: pytest check_data/test_data.py -v
"""
import pandas as pd
import numpy as np

FEATURES = ["edad", "ingreso_mensual", "distancia_casa", "anios_empresa",
            "satisfaccion_laboral", "nivel_cargo", "horas_extra", "num_empresas_previas"]
TARGET   = "abandono"
ALL_COLS = FEATURES + [TARGET]


def test_columnas_requeridas(df_train):
    """Todas las columnas requeridas deben estar presentes."""
    for col in ALL_COLS:
        assert col in df_train.columns, f"Columna faltante: {col}"


def test_sin_nulos_train(df_train):
    """El conjunto de entrenamiento no debe tener nulos."""
    nulos = df_train.isnull().sum().sum()
    assert nulos == 0, f"Se encontraron {nulos} nulos en el set de entrenamiento"


def test_sin_nulos_test(df_test):
    """El conjunto de test no debe tener nulos."""
    nulos = df_test.isnull().sum().sum()
    assert nulos == 0, f"Se encontraron {nulos} nulos en el set de test"


def test_target_es_binario(df_train):
    """El target 'abandono' debe ser binario (0 o 1)."""
    valores = set(df_train[TARGET].unique())
    assert valores.issubset({0, 1}), f"Target no binario: {valores}"


def test_tasa_abandono_razonable(df_train):
    """La tasa de abandono debe estar entre 5% y 50%."""
    tasa = df_train[TARGET].mean()
    assert 0.05 <= tasa <= 0.50, f"Tasa de abandono fuera de rango: {tasa:.2%}"


def test_edad_en_rango(df_train):
    """La edad debe estar entre 0 y 1 (normalizada) o 18 y 65."""
    assert df_train["edad"].between(0, 1).all() or df_train["edad"].between(18, 65).all(), \
        "Valores de edad fuera de rango"


def test_satisfaccion_laboral_categorias(df_train):
    """satisfaccion_laboral debe tener valores en {1, 2, 3, 4}."""
    valores = set(df_train["satisfaccion_laboral"].unique())
    assert valores.issubset({1, 2, 3, 4}), f"Categorías inesperadas en satisfaccion_laboral: {valores}"


def test_nivel_cargo_categorias(df_train):
    """nivel_cargo debe tener valores en {1, 2, 3, 4, 5}."""
    valores = set(df_train["nivel_cargo"].unique())
    assert valores.issubset({1, 2, 3, 4, 5}), f"Categorías inesperadas en nivel_cargo: {valores}"


def test_horas_extra_binario(df_train):
    """horas_extra debe ser binario."""
    valores = set(df_train["horas_extra"].unique())
    assert valores.issubset({0, 1}), f"horas_extra no binario: {valores}"


def test_train_test_sin_solapamiento(df_train, df_test):
    """Train y test no deben tener filas duplicadas exactas."""
    merged = pd.merge(df_train, df_test, how="inner")
    assert len(merged) == 0, f"{len(merged)} filas solapadas entre train y test"


def test_tamanio_minimo_train(df_train):
    """El set de entrenamiento debe tener al menos 1000 filas."""
    assert len(df_train) >= 1000, f"Set de entrenamiento muy pequeño: {len(df_train)}"


def test_distribucion_similar_train_test(df_train, df_test):
    """La tasa de abandono en train y test debe ser similar (±5%)."""
    tasa_train = df_train[TARGET].mean()
    tasa_test  = df_test[TARGET].mean()
    assert abs(tasa_train - tasa_test) < 0.05, \
        f"Distribución muy diferente: train={tasa_train:.2%} test={tasa_test:.2%}"
