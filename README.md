# 🏢 Pipeline MLOps End-to-End — Predicción de Abandono de Empleados

## Proyecto Final — Sebastian Campos

Pipeline completo de MLOps para predecir el **abandono de empleados** (Employee Attrition), inspirado en el dataset [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) de Kaggle.

---

## 📋 Descripción del Caso de Uso

El abandono de empleados es un problema crítico para las empresas. Este proyecto implementa un flujo **end-to-end de MLOps** que:

1. **Genera/descarga** un dataset de recursos humanos
2. **Preprocesa** los datos (imputación, normalización)
3. **Segrega** en train/test con estratificación
4. **Valida** la calidad de los datos con pytest
5. **Entrena** un modelo Random Forest con GridSearchCV + MLflow
6. **Evalúa** el modelo con quality gates (AUC, F1)
7. **Detecta drift** entre datos de referencia y producción
8. **Sirve** el modelo como API REST con FastAPI + Docker

---

## 🗂️ Estructura del Proyecto

```
ProyectoFinal_SebastianCampos/
├── .github/workflows/ml_pipeline.yml   ← CI/CD con GitHub Actions
├── .gitignore
├── Makefile                            ← Automatización de tareas
├── README.md
├── config.yaml                         ← Configuración centralizada
├── requirements.txt                    ← Dependencias Python
├── main.py                             ← Orquestador del pipeline
├── data/run.py                         ← Etapa 1: Generación de datos
├── preprocess/run.py                   ← Etapa 2: Limpieza
├── segregate/run.py                    ← Etapa 3: Split train/test
├── check_data/                         ← Etapa 4: Validación
│   ├── conftest.py
│   └── test_data.py
├── random_forest/run.py                ← Etapa 5: Entrenamiento
├── evaluate/run.py                     ← Etapa 6: Evaluación
├── drift/run.py                        ← Etapa 7: Detección de drift
├── serve/                              ← API REST
│   ├── __init__.py
│   ├── app.py
│   └── Dockerfile
├── artifacts/                          ← Modelo y métricas (generado)
└── reportes/                           ← Reportes de drift (generado)
```

---

## 📊 Variables del Dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `edad` | int | Edad del empleado (18-60) |
| `ingreso_mensual` | float | Ingreso mensual en USD |
| `distancia_casa` | int | Distancia al trabajo en km |
| `anios_empresa` | int | Años en la empresa |
| `satisfaccion_laboral` | int | 1=Baja, 2=Media, 3=Alta, 4=Muy Alta |
| `nivel_cargo` | int | 1=Junior a 5=Director |
| `horas_extra` | int | 0=No, 1=Sí |
| `num_empresas_previas` | int | Cantidad de empleos anteriores |
| **`abandono`** | int | **TARGET**: 0=Se queda, 1=Abandona |

---

## 🚀 Ejecución Rápida

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline completo (7 etapas)
python main.py

# 4. Servir API (después del pipeline)
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

---

## 🔧 Tecnologías Utilizadas

- **Python 3.10+** — Lenguaje principal
- **scikit-learn** — Modelo Random Forest + GridSearchCV
- **MLflow** — Tracking de experimentos y registro de modelos
- **pandas / numpy** — Procesamiento de datos
- **pytest** — Validación automática de datos
- **FastAPI** — API REST de producción
- **Docker** — Contenedorización
- **GitHub Actions** — CI/CD automatizado
- **Evidently** — Detección de drift

---

## 📡 API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Información de la API |
| GET | `/health` | Health check del modelo |
| POST | `/predict` | Predicción de abandono |
| GET | `/docs` | Documentación Swagger |

### Ejemplo de predicción:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"edad":35,"ingreso_mensual":6500,"distancia_casa":9,"anios_empresa":5,"satisfaccion_laboral":3,"nivel_cargo":2,"horas_extra":1,"num_empresas_previas":1}'
```

---

## 👤 Autor

**Sebastian Campos** — Proyecto Final MLOps
