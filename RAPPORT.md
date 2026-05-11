# RAPPORT — TP Python Avancé

## 1. Objectif du projet

Le projet vise à construire un pipeline industriel de prédiction du pourboire relatif (`tip_pct`) sur les courses Yellow Taxi NYC 2023, depuis l'ingestion des Parquets bruts jusqu'au service REST monitoré.

La variable cible est:

```text
tip_pct = tip_amount / fare_amount
```

Le fil directeur retenu est:
- architecture modulaire réutilisable,
- optimisation chiffrée avant parallélisation,
- séparation stricte entre préparation train/serving,
- exposition d'un modèle versionné via MLflow + FastAPI.

## 2. Partie 1 — POO avancée et patterns

### 2.1 Architecture mise en place

- `PipelineStepMeta` valide la présence de `name` et du hook `_run(self, df: pd.DataFrame) -> pd.DataFrame`.
- `BasePipelineStep.run()` centralise le cycle d'exécution et les notifications `before_run` / `after_run`.
- `ModelConfig` repose sur des descripteurs typés (`Positive`, `OneOf`, `TypedAttr`, `BoundedFloat`).
- `Config` fournit un singleton en lecture seule sur `config.yaml`.
- `LoaderFactory` repose sur un registre + décorateur `@register_loader`.
- `FeatureEngineer` injecte dynamiquement une stratégie d'encodage.
- `PipelineSubject`, `LoggingObserver`, `MetricsObserver`, `AlertObserver` assurent le monitoring sans couplage fort.
- `Pipeline` combine mixins (`LoggableMixin`, `SerializableMixin`, `ValidatableMixin`) et composition d'étapes.

### 2.2 Validation d'intégration

Le script `scripts/part1_pipeline.py` a été exécuté sur `yellow_tripdata_2023-01.parquet`.

Résultats observés:
- fichier source: `3_066_766` lignes,
- sortie persistée: `data/processed/yellow_tripdata_2023-01_part1.parquet`,
- journal d'exécution: `pipeline.log`,
- colonne `tip_pct` bien générée.

## 3. Partie 2 — Optimisation et parallélisme

### 3.1 Démarche

Le pipeline a d'abord été profilé en version naïve, puis réécrit en version vectorisée avant d'introduire multiprocessing, threading ciblé et Numba.

Modules produits:
- `src/optim/vectorize.py`
- `src/optim/benchmarks.py`
- `src/optim/parallel.py`
- `src/optim/async_io.py`
- `src/optim/jit_kernels.py`

### 3.2 Benchmark significatif

Sur un échantillon réel de `50_000` lignes de janvier:

| Fonction | Temps observé |
|---|---:|
| `compute_features_naive` | ~138 ms |
| `compute_features_vectorized` | ~4.5 ms |

Le facteur d'accélération observé est de l'ordre de `30x` sur cet échantillon, avec un code bien plus scalable sur gros volumes.

### 3.3 Conclusions

- la vectorisation apporte le gain principal,
- `multiprocessing` est pertinent pour traiter les mois en parallèle,
- `threading` n'aide pas sur CPU-bound mais reste utile pour l'I/O,
- Numba devient intéressant pour les noyaux non trivialement vectorisables.

## 4. Partie 3 — Pandas avancé et feature engineering

### 4.1 Enrichissements réalisés

- optimisation mémoire et conversion de dtypes,
- calculs par `MultiIndex` / `pivot_table`,
- window functions (`rolling`, `expanding`, `rank`, lags),
- features temporelles dans `TemporalFeaturizer`,
- target encoding bayésien et cross-fitted,
- imputation médiane, KNN, MICE, multiple imputation,
- time-series engineering agrégé à l'heure.

### 4.2 Choix de design

Le point clé a été d'utiliser un chemin commun de préparation des features entre entraînement et serving. Cela limite la dérive entre le monde notebook / entraînement et le monde API / production.

Ce rôle est centralisé dans `src/models/features.py`.

### 4.3 Limite notable

L'exercice 3.6 côté embeddings profonds n'a pas été implémenté faute d'environnement PyTorch/Keras dédié dans cette version du projet.

## 5. Partie 4 — Industrialisation et MLOps

### 5.1 MLflow

Le training est encapsulé dans `src/models/train.py`:
- tracking des paramètres,
- logging des métriques (`rmse_val`, `mae_val`, `residual_std`),
- logging du modèle et des artefacts,
- helpers de promotion vers le registry.

Un point technique important a été corrigé: les artefacts MLflow doivent être stockés via une racine proxifiée (`mlflow-artifacts:/`) pour que les modèles rechargés depuis `runs:/.../model` et `models:/.../Production` soient réellement exploitables.

### 5.2 API

L'API FastAPI expose:
- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /metrics`

Les entrées/sorties sont validées par Pydantic v2, avec rejet explicite des courses incohérentes (`trip_distance < 0.1`, etc.).

### 5.3 Monitoring

La stack `docker-compose` orchestre:
- `api`
- `mlflow`
- `postgres`
- `prometheus`
- `grafana`

`prometheus-fastapi-instrumentator` expose automatiquement les métriques HTTP. Un dashboard Grafana minimal a été provisionné.

## 6. Validation bout-en-bout

### 6.1 Stack opérationnelle

Validation effectuée:
- `postgres` healthy,
- `mlflow` healthy,
- `api` healthy,
- `prometheus` up,
- `grafana` up.

### 6.2 Modèle servi

Un modèle a été entraîné puis promu dans le Model Registry MLflow.

Exemple de réponse API observée:

```json
{
  "tip_pct_predicted": 0.3007620537480158,
  "confidence_interval": [0.20276205374801579, 0.39876205374801577],
  "model_version": "TipPredictor/Production"
}
```

La target Prometheus `http://api:8000/metrics` remonte bien en `up`.

## 7. Limites et discussion

### 7.1 Mémoire

Le principal verrou de cette machine concerne le vrai entraînement `full-data` sur les 12 mois:
- dans Docker: conteneur `OOMKilled`,
- sur le host: processus Python tué par le système.

Cause: l'implémentation actuelle concatène tous les mois en mémoire avant nettoyage et split, ce qui provoque un pic mémoire trop élevé.

### 7.2 Conséquence

Le pipeline MLOps complet est validé de bout en bout, mais le run “12 mois sans échantillonnage” n'est pas soutenable sur cette machine sans refonte out-of-core / streaming.

### 7.3 Pistes d'amélioration

- lecture mois par mois avec agrégation incrémentale,
- matérialisation intermédiaire dans `data/features/`,
- apprentissage sur chunks,
- vraie intégration DVC initialisée et trackée dans Git,
- ajout d'embeddings zone avec PyTorch/Keras,
- CI/CD GitHub Actions.

## 8. Conclusion

Le projet remplit l'objectif principal du TP: montrer un pipeline complet, modulaire, testé et industrialisé autour des données NYC Taxi 2023.

Les points les plus solides sont:
- l'architecture POO et patterns,
- l'instrumentation et les tests,
- la chaîne MLflow → registry → API → Prometheus/Grafana.

La principale limite restante est purement opérationnelle: l'entraînement full-data ne tient pas en mémoire avec la stratégie actuelle de chargement.
