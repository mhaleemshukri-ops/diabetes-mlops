# Diabetes MLOps Project

This is our MLOps assignment using the diabetes dataset from sklearn.

We trained two models and used FastAPI, Docker and GitHub Actions for the application and testing.

## Models

- v0.1: Linear Regression (RMSE: 53.85)
- v0.2: Ridge Regression (RMSE: 53.63)

We used an 80/20 train-test split. Both models are saved in the models folder.

## Running the project

Install the requirements:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Run the API:

```bash
uvicorn app:app --reload
```

API documentation: http://127.0.0.1:8000/docs

## Docker

Build the image:

```bash
docker build -t diabetes-mlops .
```

Run with v0.2:

```bash
docker run -p 8000:8000 -e MODEL_VERSION=v0.2 diabetes-mlops
```

For v0.1, change `MODEL_VERSION` to `v0.1`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

GitHub Actions is used for CI testing.

Model changes are documented in CHANGELOG.md.
