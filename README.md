# Plan: ML Microservice — FastAPI + Docker + Kubernetes

A learning project to build, deploy, and serve a machine learning model using FastAPI, Docker, and Kubernetes (minikube).

## Project Goals

- Train and save an XGBoost digit classifier
- Serve predictions via a FastAPI REST API
- Containerize the service with Docker
- Deploy and scale using Kubernetes (minikube)

## Key Decisions

- **Model artifact strategy:** Train locally, bake into Docker image at build time
- **Local Kubernetes:** minikube
- **Compose file:** docker-compose.yml included for local testing
- **Python version:** Local 3.14 (deferred if issues arise), Docker uses 3.11-slim
- **Input validation:** Permissive `list[float]`, no strict 64-feature length check

## Target Project Structure

```
ml-microservice/
├── app/
│   ├── train.py          (XGBoost model training script)
│   └── serve.py          (FastAPI inference app)
├── artifacts/
│   ├── model.json        (XGBoost model — gitignored)
│   └── scaler.pkl        (StandardScaler — gitignored)
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── main.py               (placeholder)
```

## Phase 1: Dependencies & File Scaffolding

✅ **Status: Complete**

- Create `requirements.txt` with core packages: xgboost, ucimlrepo, scikit-learn, numpy, fastapi, uvicorn[standard], pydantic, joblib
- Create `artifacts/` directory (empty, gitignored)
- Clear `main.py` boilerplate
- Create `.gitignore` for artifacts/, venv/, __pycache__

## Phase 2: Fix Training Script (`app/train.py`)

✅ **Status: Complete**

**Issue:** The `StandardScaler` is fit during training but never saved. Without it, inference preprocessing won't match training preprocessing.

**Changes:**
- Add `import joblib`
- Update `preprocess_data()` to return the scaler as 7th value
- Update `main()` to unpack 7 values from `preprocess_data()`
- Save scaler to `artifacts/scaler.pkl` using `joblib.dump()`
- Update `save_model()` default path to `"artifacts/model.json"`

**Result:**
- Running `python app/train.py` saves both `artifacts/model.json` and `artifacts/scaler.pkl`

## Phase 3: FastAPI Serving App (`app/serve.py`)

⏳ **Status: Pending — Implementation guidance provided**

**Create `app/serve.py` with:**

- **Startup:** Load model and scaler once using FastAPI lifespan context manager
  - `model = xgb.Booster()`
  - `model.load_model("artifacts/model.json")`
  - `scaler = joblib.load("artifacts/scaler.pkl")`

- **Pydantic Models:**
  - `PredictRequest`: `features: list[float]`
  - `PredictionResponse`: `prediction: int`
  - `HealthResponse`: `status: str`

- **Endpoints:**
  - `GET /health` → `{"status": "ok"}`
  - `POST /predict` → scales input, wraps in `xgb.DMatrix`, predicts digit (0-9)

**Testing locally:**
```bash
# Terminal 1: Start app
uvicorn app.serve:app --reload --port 8000

# Terminal 2: Health check
curl http://localhost:8000/health
# {"status":"ok"}

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0]*64}'
# {"prediction": 0}

# Interactive API docs
# Open http://localhost:8000/docs
```

## Phase 4: Docker

⏳ **Status: Pending**

**Dockerfile:**
- Base: `python:3.11-slim`
- Install dependencies from `requirements.txt`
- Copy `artifacts/` and `app/` into image
- Expose port 8000
- Entrypoint: `uvicorn app.serve:app --host 0.0.0.0 --port 8000`

**`.dockerignore`:**
- `venv/`, `.idea/`, `__pycache__`, `*.pyc`

**`docker-compose.yml`:**
- Build image (tag: `ml-microservice`)
- Map `8000:8000`

**Build & run locally:**
```bash
# Build
docker build -t ml-microservice .

# Run
docker compose up

# Test
curl http://localhost:8000/health
```

## Phase 5: Kubernetes (minikube)

⏳ **Status: Pending**

**`k8s/deployment.yaml`:**
- Deployment with 2 replicas
- Image: `ml-microservice:latest`
- Port: 8000
- Image pull policy: `Never` (use local image)

**`k8s/service.yaml`:**
- Service type: `NodePort`
- Port: 8000
- Selector matches Deployment pods

**Deploy & access:**
```bash
# Start minikube
minikube start

# Share Docker daemon (builds inside minikube)
eval $(minikube docker-env)

# Build image (now available inside minikube)
docker build -t ml-microservice .

# Deploy
kubectl apply -f k8s/

# Get service URL
minikube service ml-microservice-service --url

# Test
curl http://<minikube-service-url>/health
```

## Verification Checklist

### Phase 1
- [ ] `requirements.txt` created with all dependencies
- [ ] `artifacts/` directory exists
- [ ] `.gitignore` includes artifacts/

### Phase 2
- [ ] `python app/train.py` runs without errors
- [ ] `artifacts/model.json` created
- [ ] `artifacts/scaler.pkl` created

### Phase 3
- [ ] `app/serve.py` created
- [ ] `uvicorn app.serve:app --reload` starts successfully
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `POST /predict` with 64 floats returns `{"prediction": int}`

### Phase 4
- [ ] `Dockerfile` created
- [ ] `.dockerignore` created
- [ ] `docker-compose.yml` created
- [ ] `docker build -t ml-microservice .` succeeds
- [ ] `docker compose up` starts the service
- [ ] Endpoints accessible via `curl localhost:8000/*`

### Phase 5
- [ ] `k8s/deployment.yaml` created
- [ ] `k8s/service.yaml` created
- [ ] `minikube start` succeeds
- [ ] `kubectl apply -f k8s/` deploys pods
- [ ] `minikube service ml-microservice-service --url` returns accessible URL
- [ ] Endpoints accessible via the minikube service URL

## Dataset

**UCI Machine Learning Repository — Optical Recognition of Handwritten Digits**
- ID: 80
- Samples: ~1,797
- Features: 64 (8×8 pixel intensity values)
- Classes: 10 (digits 0-9)
- Train/Val/Test split: 70% / 15% / 15%

## Model

**XGBoost Classifier**
- Objective: `multi:softmax` (10-class classification)
- Max depth: 6
- Learning rate: 0.1
- Boost rounds: 200 (with early stopping, window 20)
- Typical accuracy: ~97% on test set

## Quick Start (End-to-End)

```bash
# Phase 2: Train model
python app/train.py

# Phase 3: Serve locally
uvicorn app.serve:app --reload

# Phase 4: Docker
docker build -t ml-microservice .
docker compose up

# Phase 5: Kubernetes
minikube start
eval $(minikube docker-env)
docker build -t ml-microservice .
kubectl apply -f k8s/
minikube service ml-microservice-service --url
```

## Notes

- Model and scaler are generated by training; excluded from version control via `.gitignore`
- Artifacts must be baked into the Docker image at build time (no runtime downloads)
- FastAPI lifespan context ensures efficient resource loading
- Error handling for missing artifacts should be added in production
- Multi-stage Docker builds and model versioning strategies can be explored later
