"""
data/run.py — Etapa 1: Descarga/generación del dataset de abandono de empleados.

Dataset basado en: IBM HR Analytics Employee Attrition & Performance (Kaggle)
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Genera un dataset sintético realista con las mismas variables que el dataset
de attrition real de IBM, adaptado a español.

Ejecutar: python data/run.py
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DOWNLOAD | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N        = 12000
SEED     = 42


def generar_dataset_attrition(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Genera dataset sintético de abandono de empleados (12,000 empleados).

    Variables inspiradas en el dataset IBM HR Analytics de Kaggle.
    """
    rng = np.random.default_rng(seed)

    edad                 = rng.integers(18, 61, n)
    ingreso_mensual      = rng.normal(6500, 4500, n).clip(1000, 25000)
    distancia_casa       = rng.integers(1, 30, n)
    anios_empresa        = rng.integers(0, 40, n)
    satisfaccion_laboral = rng.choice([1, 2, 3, 4], n, p=[0.15, 0.25, 0.35, 0.25])
    nivel_cargo          = rng.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.30, 0.20, 0.12, 0.08])
    horas_extra          = rng.choice([0, 1], n, p=[0.70, 0.30])
    num_empresas_previas = rng.integers(0, 10, n)

    # Target correlacionado con las variables (lógica de negocio realista)
    # Ajustado para ~16-20% abandono (similar al dataset IBM HR de Kaggle)
    score_abandono = (
        -0.02 * edad
        - 0.0001 * ingreso_mensual
        + 0.05 * distancia_casa
        - 0.03 * anios_empresa
        - 0.25 * satisfaccion_laboral
        - 0.20 * nivel_cargo
        + 1.20 * horas_extra
        + 0.20 * num_empresas_previas
        + rng.normal(0, 0.8, n)
    )
    prob_abandono = 1 / (1 + np.exp(-score_abandono))
    abandono      = (rng.uniform(0, 1, n) < prob_abandono).astype(int)

    df = pd.DataFrame({
        "edad":                 edad,
        "ingreso_mensual":      ingreso_mensual.round(2),
        "distancia_casa":       distancia_casa,
        "anios_empresa":        anios_empresa,
        "satisfaccion_laboral": satisfaccion_laboral,
        "nivel_cargo":          nivel_cargo,
        "horas_extra":          horas_extra,
        "num_empresas_previas": num_empresas_previas,
        "abandono":             abandono,
    })

    # Introducir nulos realistas (~2% por columna)
    for col in ["ingreso_mensual", "distancia_casa", "anios_empresa"]:
        idx = rng.choice(n, size=int(n * 0.02), replace=False)
        df.loc[idx, col] = np.nan

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Descarga o genera el dataset de attrition")
    parser.add_argument("--output",  default="data/attrition_data_raw.csv")
    parser.add_argument("--n_rows",  type=int, default=N)
    parser.add_argument("--seed",    type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    log.info("Generando dataset de abandono de empleados (%d filas)...", args.n_rows)
    df = generar_dataset_attrition(n=args.n_rows, seed=args.seed)

    output = Path(args.output)
    df.to_csv(output, index=False)

    log.info("Dataset guardado: %s", output)
    log.info("Shape: %d filas x %d columnas", *df.shape)
    log.info("Tasa de abandono: %.1f%%", df['abandono'].mean() * 100)
    log.info("Nulos: %d", df.isnull().sum().sum())
