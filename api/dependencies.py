import json
import joblib
import os
from .config import settings
from fastapi import HTTPException

# Global cache
model_pipeline = None
model_metadata = None

def load_artifacts():
    global model_pipeline, model_metadata
    
    if not os.path.exists(settings.MODEL_PATH):
        raise RuntimeError(f"Model artifact not found at {settings.MODEL_PATH}")
    if not os.path.exists(settings.METADATA_PATH):
        raise RuntimeError(f"Metadata artifact not found at {settings.METADATA_PATH}")
        
    try:
        model_pipeline = joblib.load(settings.MODEL_PATH)
        with open(settings.METADATA_PATH, 'r') as f:
            model_metadata = json.load(f)
            
        # Verify metadata
        required_keys = ['probability_decision_threshold', 'excluded_features', 'included_features']
        for k in required_keys:
            if k not in model_metadata:
                raise RuntimeError(f"Metadata missing required key: {k}")
                
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {str(e)}")

def get_model():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model_pipeline

def get_metadata():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Metadata is not loaded")
    return model_metadata
