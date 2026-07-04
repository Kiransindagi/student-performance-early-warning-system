from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
from .schemas import StudentRecord, PredictionResult, BatchPredictionResult, ModelMetadataResponse
from .dependencies import load_artifacts, get_model, get_metadata
from .services import predict_single, predict_batch
from .config import settings

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    # Cleanup if needed

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse("app/static/index.html")

@app.get("/health")
def health_check(model=Depends(get_model), meta=Depends(get_metadata)):
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": meta.get('model_name', 'unknown')
    }

@app.get("/model/info", response_model=ModelMetadataResponse)
def model_info(meta=Depends(get_metadata)):
    return ModelMetadataResponse(
        model_name=meta.get("model_name"),
        model_type=meta.get("model_type"),
        experiment_track=meta.get("experiment_track"),
        target_definition=meta.get("target_definition"),
        probability_threshold=meta.get("probability_decision_threshold"),
        included_feature_count=len(meta.get("included_features", [])),
        excluded_prior_grade_features=meta.get("excluded_features", []),
        evaluation_summary=meta.get("test_metrics", {})
    )

@app.post("/predict", response_model=PredictionResult)
def predict(record: StudentRecord, model=Depends(get_model), meta=Depends(get_metadata)):
    return predict_single(record, model, meta)

@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch_endpoint(records: List[StudentRecord], model=Depends(get_model), meta=Depends(get_metadata)):
    return predict_batch(records, model, meta)
