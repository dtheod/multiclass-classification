# Multiclass Classification Pipeline

Small end-to-end project that processes bug/issue data, builds features, trains an XGBoost multiclass classifier and exposes a FastAPI prediction endpoint.

Key modules and entrypoints
- Data processing flow: [`src.process.process_data`](src/process.py) — implementation: [src/process.py](src/process.py)
  - Core tasks: [`src.process.load_data`](src/process.py), [`src.process.data_selection`](src/process.py), [`src.process.component_name_fix`](src/process.py), [`src.process.assignee_fix`](src/process.py), [`src.process.product_name_fix`](src/process.py), [`src.process.final_selection`](src/process.py)
- Feature engineering flow: [`src.feature_engineer.feature_data`](src/feature_engineer.py) — implementation: [src/feature_engineer.py](src/feature_engineer.py)
  - Feature tasks: [`src.feature_engineer.create_features`](src/feature_engineer.py), [`src.feature_engineer.one_hot_encoding`](src/feature_engineer.py), [`src.feature_engineer.feature_variance`](src/feature_engineer.py), [`src.feature_engineer.principal_components`](src/feature_engineer.py), [`src.feature_engineer.oversampling`](src/feature_engineer.py), [`src.feature_engineer.model_input`](src/feature_engineer.py)
- Training: [`src.train_model.train_model`](src/train_model.py) — implementation: [src/train_model.py](src/train_model.py)
  - Helpers: [`src.train_model.label_encoding`](src/train_model.py), [`src.train_model.stratified_split`](src/train_model.py), [`src.train_model.get_objective`](src/train_model.py), [`src.train_model.optimize`](src/train_model.py), [`src.train_model.predict`](src/train_model.py)
- API (FastAPI): [app/main.py](app/main.py)
  - Pipeline steps: `DataSelection`, `CustomFeature`, `OneHot`, `FeatureVariance`, `PCA_Components`, `XGBoosting`, `ReverseEncoder` — see [app/main.py](app/main.py)
  - Test client: [app/serve.py](app/serve.py)

What it does
- Cleans and normalises raw dataset columns, groups rare products/components into generic buckets.
- Builds aggregated features (counts, uniques) and persists them for serving.
- Encodes categorical variables via one-hot, applies variance threshold and PCA for dimensionality reduction.
- Trains an XGBoost multiclass model via hyperparameter search (Hyperopt), saves final model and artifacts to `models/`.
- Exposes a prediction endpoint (/predict) through a sklearn-style pipeline wrapped in FastAPI.

Algorithms & libraries
- Data orchestration: Prefect flows (`src/process.py`, `src/feature_engineer.py`)
- Feature engineering: OneHotEncoder, VarianceThreshold, PCA, SMOTE (imbalanced sampling) — see [`src.feature_engineer`](src/feature_engineer.py)
- Model: XGBoost classifier (`xgboost.XGBClassifier`) with hyperparameter optimization using Hyperopt — see [`src.train_model.get_objective`](src/train_model.py) and [`src.train_model.optimize`](src/train_model.py)
- Serving: FastAPI + sklearn Pipeline wrappers — see [app/main.py](app/main.py)
- Misc: fuzzy matching via `fuzzywuzzy` in assignee cleaning (`src/process.py`)

Quick start (development)
1. Install dependencies and hooks:
   - Use Poetry (project configured): `make install` (calls `poetry install`) — see [Makefile](Makefile)
2. Persist Prefect flow outputs (optional):
   - `export PREFECT__FLOWS__CHECKPOINTING=true`
3. Run processing + feature flows:
   - `python src/feature_main.py` — entry: [`src.feature_main.main`](src/feature_main.py)
     - This runs both processing (`src/process.process_data`) and feature engineering (`src/feature_engineer.feature_data`).
4. Train the model:
   - `python src/train_model.py` — entry: [`src.train_model.train_model`](src/train_model.py)
   - Artifacts saved into `models/` (label encoder, one-hot encoder, PCA, XGBoost model, feature summaries).
5. Run tests:
   - `make test` (`pytest --no-header -v`) — tests live in [tests/test_process.py](tests/test_process.py)
6. Run API (locally / in Docker):
   - Local (requires the saved `models/`): run FastAPI app defined in [app/main.py](app/main.py) with Uvicorn.
   - Docker:
     - Build: `docker build -t testliodocker ./` — Dockerfile: [Dockerfile](Dockerfile)
     - Run: `docker run -d --name testliodocker -p 80:80 testliodocker`
     - Test endpoint: see example client [app/serve.py](app/serve.py)

Files of interest
- Processing & features: [src/process.py](src/process.py), [src/feature_engineer.py](src/feature_engineer.py)
- Training: [src/train_model.py](src/train_model.py)
- Flow launcher: [src/feature_main.py](src/feature_main.py)
- API: [app/main.py](app/main.py), [app/serve.py](app/serve.py)
- Tests: [tests/test_process.py](tests/test_process.py)
- Notebook: exploratory analysis: [notebooks/Exploration.ipynb](notebooks/Exploration.ipynb)
- Configs: [config/main.yaml](config/main.yaml)
- Docker: [Dockerfile](Dockerfile)
- Makefile: [Makefile](Makefile)

Notes & tips
- Paths for saved artifacts are controlled by Hydra config (`config/main.yaml`) — inspect and adjust before running flows.
- The Prefect flows persist intermediate CSVs under `data/processed/` and `data/features_data/` when checkpointing is enabled.
- The FastAPI app expects model artifacts under `/models` inside the container — the Dockerfile copies `./models` into the image.
