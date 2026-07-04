from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class StudentRecord(BaseModel):
    age: int = Field(..., ge=10, le=30, description="Student's age")
    sex: str = Field(..., description="Student's sex (e.g., 'F', 'M')")
    address: str = Field(..., description="Student's home address type ('U' - urban or 'R' - rural)")
    family_size: str = Field(..., description="Family size ('LE3' - less or equal to 3 or 'GT3' - greater than 3)")
    parents_cohabitation: str = Field(..., description="Parent's cohabitation status ('T' - living together or 'A' - apart)")
    mother_education: int = Field(..., ge=0, le=4, description="Mother's education (0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)")
    father_education: int = Field(..., ge=0, le=4, description="Father's education (0 to 4)")
    mother_job: str = Field(..., description="Mother's job ('teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')")
    father_job: str = Field(..., description="Father's job")
    study_time: int = Field(..., ge=1, le=4, description="Weekly study time (1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (1-3, else 4)")
    school_support: str = Field(..., description="Extra educational support (yes or no)")
    family_support: str = Field(..., description="Family educational support (yes or no)")
    paid_classes: str = Field(..., description="Extra paid classes within the course subject (yes or no)")
    extra_activities: str = Field(..., description="Extra-curricular activities (yes or no)")
    internet_access: str = Field(..., description="Internet access at home (yes or no)")
    romantic_relationship: str = Field(..., description="With a romantic relationship (yes or no)")
    free_time: int = Field(..., ge=1, le=5, description="Free time after school (from 1 - very low to 5 - very high)")
    going_out: int = Field(..., ge=1, le=5, description="Going out with friends (from 1 - very low to 5 - very high)")
    weekday_alcohol: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (from 1 - very low to 5 - very high)")
    weekend_alcohol: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (from 1 - very low to 5 - very high)")
    health_status: int = Field(..., ge=1, le=5, description="Current health status (from 1 - very bad to 5 - very good)")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (from 0 to 93)")

    model_config = {
        "extra": "forbid"
    }

class PredictionResult(BaseModel):
    predicted_class: int
    risk_probability: float
    probability_percentage: str
    decision_threshold: float
    risk_tier: str
    model_version: str
    decision_message: str

class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    total_records: int
    flagged_at_risk: int
    risk_rate: str

class ModelMetadataResponse(BaseModel):
    model_name: str
    model_type: str
    experiment_track: str
    target_definition: str
    probability_threshold: float
    included_feature_count: int
    excluded_prior_grade_features: List[str]
    evaluation_summary: Dict[str, Any]
