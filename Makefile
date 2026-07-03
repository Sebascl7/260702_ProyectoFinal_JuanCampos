# Makefile — Pipeline MLOps End-to-End — Abandono de Empleados
# Proyecto Final: Sebastian Campos
.PHONY: all install lint pipeline etapas serve docker smoke clean help

install:
	pip install -r requirements.txt

all: install lint pipeline serve
	@echo "Pipeline completo OK"

lint:
	flake8 . --max-line-length=100 --exclude=.git,__pycache__,mlruns,artifacts,data,reportes

# Pipeline completo
pipeline:
	python main.py

# Etapas individuales
etapa1: ; python data/run.py
etapa2: ; python preprocess/run.py
etapa3: ; python segregate/run.py
etapa4: ; pytest check_data/test_data.py -v --tb=short
etapa5: ; python random_forest/run.py
etapa6: ; python evaluate/run.py
etapa7: ; python drift/run.py

# Servidor API
serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t attrition-api:local -f serve/Dockerfile .

docker-run:
	docker run -p 8000:8000 --name attrition-api attrition-api:local

docker-stop:
	docker stop attrition-api && docker rm attrition-api

# Smoke tests
smoke:
	curl -sf http://localhost:8000/health | python3 -m json.tool
	@echo ""
	curl -X POST http://localhost:8000/predict \
	  -H 'Content-Type: application/json' \
	  -d '{"edad":35,"ingreso_mensual":6500,"distancia_casa":9,"anios_empresa":5,"satisfaccion_laboral":3,"nivel_cargo":2,"horas_extra":1,"num_empresas_previas":1}' \
	  | python3 -m json.tool

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf data/ artifacts/ reportes/ mlruns/ __pycache__ pipeline_run.log
	find . -name "*.pyc" -delete
	@echo "Limpieza completada"

help:
	@echo ""
	@echo "=== Pipeline MLOps End-to-End — Abandono de Empleados ==="
	@echo "  make install    — instalar dependencias"
	@echo "  make pipeline   — ejecutar las 7 etapas en orden"
	@echo "  make etapa1-7   — ejecutar una etapa específica"
	@echo "  make serve      — levantar API FastAPI en puerto 8000"
	@echo "  make docker-build — construir imagen Docker"
	@echo "  make docker-run   — ejecutar contenedor"
	@echo "  make smoke      — test rápido de los endpoints"
	@echo "  make mlflow-ui  — UI de MLflow en puerto 5000"
	@echo "  make clean      — limpiar artefactos"
	@echo ""
