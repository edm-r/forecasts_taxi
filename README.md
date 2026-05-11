# TP Taxi NYC 2023

Pipeline industriel de prédiction du `tip_pct` sur les courses Yellow Taxi NYC 2023, réalisé pour le TP Python Avancé du Master 1 IABD.

## Objectif

Construire une chaîne complète:
- ingestion des Parquets TLC 2023,
- pipeline modulaire avec patterns POO,
- optimisation et parallélisation,
- feature engineering avancé,
- entraînement tracké avec MLflow,
- exposition via API FastAPI,
- monitoring Prometheus + Grafana.

La cible prédite est:

```text
tip_pct = tip_amount / fare_amount
```

## Arborescence utile

```text
tp_taxi/
├── data/raw/                 # parquets bruts 2023
├── data/processed/           # sorties pipeline
├── src/core/                 # metaclasses, descriptors, decorators, context, config
├── src/pipeline/             # loaders, strategies, observers, orchestrator
├── src/optim/                # vectorisation, benchmarks, parallélisme, async, numba
├── src/features/             # temporal, encoding, imputation, windowing
├── src/models/               # training, registry helpers, feature prep
├── src/api/                  # FastAPI + schémas Pydantic
├── scripts/                  # scripts de démo et d'exploitation
├── monitoring/               # Prometheus / Grafana
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── RAPPORT.md
```

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Jeux de données

Les 12 mois 2023 sont attendus dans `data/raw/`.

Téléchargement asynchrone:

```bash
.venv/bin/python scripts/download_year.py --year 2023 --months 1 2 3 4 5 6 7 8 9 10 11 12
```

## Validation du dépôt

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Partie 1: démo loader et pipeline

Démo `LoaderFactory`:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_loader.py data/raw/yellow_tripdata_2023-01.parquet
```

Script d'intégration partie 1:

```bash
PYTHONPATH=src .venv/bin/python scripts/part1_pipeline.py
```

Sortie attendue:
- `data/processed/yellow_tripdata_2023-01_part1.parquet`
- `pipeline.log`

## Entraînement local

Exemple avec échantillonnage:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_tip_model.py \
  --tracking-uri file:./mlruns \
  --sample-rows-per-file 50000
```

`--sample-rows-per-file 0` signifie "pas d'échantillonnage", mais sur une machine limitée en RAM ce mode peut provoquer un OOM avec l'implémentation actuelle.

## Stack MLOps complète

Lancer la stack:

```bash
docker compose up -d --build
```

Services exposés:
- MLflow: `http://127.0.0.1:5001`
- API FastAPI: `http://127.0.0.1:8001`
- Swagger: `http://127.0.0.1:8001/docs`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## Entraînement vers le MLflow Docker

```bash
docker compose run --rm trainer \
  python scripts/train_tip_model.py \
  --tracking-uri http://mlflow:5000 \
  --sample-rows-per-file 1000 \
  --experiment-name tip_prediction_smoke \
  --run-name smoke_v1
```

Promotion d'un run vers `Production`:

```bash
docker compose run --rm trainer \
  python scripts/promote_model.py \
  --tracking-uri http://mlflow:5000 \
  --run-id <RUN_ID> \
  --model-name TipPredictor \
  --stage Production
```

Redémarrage de l'API:

```bash
docker compose restart api
curl http://127.0.0.1:8001/health
```

## Tests API

Prédiction unitaire:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "passenger_count": 1,
    "trip_distance": 1.1,
    "pickup_datetime": "2023-01-01T00:55:08",
    "pu_location_id": 43,
    "do_location_id": 237,
    "payment_type": 1
  }'
```

Métriques Prometheus:

```bash
curl http://127.0.0.1:8001/metrics | rg 'http_requests_total|http_request_duration_seconds'
```

## Limitations connues

- Le vrai entraînement `full-data` sur les 12 mois n'est pas tenable en mémoire avec l'implémentation actuelle, qui concatène les mois avant le split. Sur cette machine, le mode `--sample-rows-per-file 0` finit en OOM.
- La partie DVC est préparée par `scripts/init_dvc.sh`, mais nécessite une initialisation DVC effective sur la machine cible.
- Les embeddings de l'exercice 3.6 n'ont pas été implémentés faute de dépendance DL dédiée dans l'environnement actuel.
