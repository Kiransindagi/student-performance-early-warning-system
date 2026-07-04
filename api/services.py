import pandas as pd
from typing import List
from .schemas import StudentRecord, PredictionResult, BatchPredictionResult

def determine_risk_tier(probability: float, threshold: float) -> str:
    """
    Risk Tiers are presentation aids.
    Low: below threshold * 0.8
    Moderate: between threshold * 0.8 and threshold * 1.2
    High: above threshold * 1.2
    """
    if probability < threshold * 0.8:
        return "Low"
    elif probability <= threshold * 1.2:
        return "Moderate"
    else:
        return "High"

def generate_decision_message(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "The model indicates elevated academic risk based on the available early-warning features."
    return "The predicted risk is below the current intervention threshold."

def predict_single(record: StudentRecord, model, metadata) -> PredictionResult:
    # Convert to DataFrame matching pipeline features
    df = pd.DataFrame([record.model_dump()])
    
    # Ensure columns match included_features exactly in order
    features = metadata['included_features']
    df = df[features]
    
    threshold = metadata['probability_decision_threshold']
    
    # Get probability of class 1 (At Risk)
    proba = model.predict_proba(df)[0][1]
    
    pred_class = 1 if proba >= threshold else 0
    tier = determine_risk_tier(proba, threshold)
    msg = generate_decision_message(proba, threshold)
    
    return PredictionResult(
        predicted_class=pred_class,
        risk_probability=proba,
        probability_percentage=f"{proba * 100:.1f}%",
        decision_threshold=threshold,
        risk_tier=tier,
        model_version=metadata.get('model_name', 'unknown'),
        decision_message=msg
    )

def predict_batch(records: List[StudentRecord], model, metadata) -> BatchPredictionResult:
    results = [predict_single(r, model, metadata) for r in records]
    
    total = len(results)
    flagged = sum(1 for r in results if r.predicted_class == 1)
    
    return BatchPredictionResult(
        predictions=results,
        total_records=total,
        flagged_at_risk=flagged,
        risk_rate=f"{(flagged / total * 100):.1f}%" if total > 0 else "0%"
    )
