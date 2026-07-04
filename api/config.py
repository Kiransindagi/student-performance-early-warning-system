import os

class Settings:
    APP_NAME: str = "Student Academic Early Warning System"
    API_V1_STR: str = "/api/v1"
    
    # Artifact Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/early_warning_model.pkl")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "models/model_metadata.json")
    
settings = Settings()
