"""serve/app.py — API REST del modelo de predicción de abandono de empleados."""
import logging, os, pickle
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | API | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES = ["edad", "ingreso_mensual", "distancia_casa", "anios_empresa",
            "satisfaccion_laboral", "nivel_cargo", "horas_extra", "num_empresas_previas"]
UMBRAL, MODEL_PATH = 0.50, Path("artifacts/modelo_attrition.pkl")
modelo = None


class SolicitudEmpleado(BaseModel):
    edad:                 int   = Field(..., ge=18, le=65, description="Edad del empleado")
    ingreso_mensual:      float = Field(..., gt=0, description="Ingreso mensual en USD")
    distancia_casa:       int   = Field(..., ge=1, le=30, description="Distancia al trabajo en km")
    anios_empresa:        int   = Field(..., ge=0, le=40, description="Años en la empresa")
    satisfaccion_laboral: int   = Field(..., ge=1, le=4, description="1=Baja, 2=Media, 3=Alta, 4=Muy Alta")
    nivel_cargo:          int   = Field(..., ge=1, le=5, description="1=Junior, 2=Mid, 3=Senior, 4=Lead, 5=Director")
    horas_extra:          int   = Field(..., ge=0, le=1, description="0=No, 1=Sí")
    num_empresas_previas: int   = Field(..., ge=0, le=10, description="Número de empleos anteriores")
    model_config = {"json_schema_extra": {"example": {
        "edad": 35, "ingreso_mensual": 6500.0, "distancia_casa": 9,
        "anios_empresa": 5, "satisfaccion_laboral": 3, "nivel_cargo": 2,
        "horas_extra": 1, "num_empresas_previas": 1}}}


class PrediccionEmpleado(BaseModel):
    probabilidad_abandono: float
    decision:              str
    score:                 float
    modelo:                str


class HealthResponse(BaseModel):
    status: str
    modelo: str
    version: str


def cargar_modelo():
    global modelo
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            modelo = mlflow.sklearn.load_model(
                f"models:/{os.getenv('MODEL_NAME','EmployeeAttritionModel')}/{os.getenv('MODEL_STAGE','Production')}")
            log.info("Modelo desde MLflow Registry: %s", type(modelo).__name__)
            return
        except Exception as e:
            log.warning("MLflow Registry no disponible: %s", e)
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
        log.info("Modelo desde: %s (%s)", MODEL_PATH, type(modelo).__name__)
    else:
        raise FileNotFoundError("Ejecuta: python random_forest/run.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(title="API Predicción Abandono de Empleados — MLOps Proyecto Final",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Info"])
def root():
    return {"api": "Predicción Abandono de Empleados", "version": "1.0.0",
            "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(status="ok", modelo=type(modelo).__name__, version="1.0.0")


@app.post("/predict", response_model=PrediccionEmpleado, tags=["Prediccion"])
def predict(solicitud: SolicitudEmpleado):
    """Evalúa riesgo de abandono. decision: RETENER | RIESGO_ABANDONO"""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        df   = pd.DataFrame([solicitud.model_dump()])
        prob = float(modelo.predict_proba(df[FEATURES])[0][1])
        return PrediccionEmpleado(
            probabilidad_abandono=round(prob, 4),
            decision="RIESGO_ABANDONO" if prob >= UMBRAL else "RETENER",
            score=round(prob, 4),
            modelo=type(modelo).__name__,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
