import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.inspection import permutation_importance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.ingest import load_data
from src.data.validate import create_target
from src.features.preprocess import build_preprocessor, prepare_data
from src.models.train import get_models, cross_validate_models, tune_logistic_regression, tune_random_forest

print("Loading Data...")
df = load_data("data/raw/student-mat.csv")
df = create_target(df, threshold=10)

# Track A: Early Warning (no grades)
X_ew, y_ew = prepare_data(df, include_prior_grades=False)
X_train_ew, X_test_ew, y_train, y_test = train_test_split(X_ew, y_ew, test_size=0.2, random_state=42, stratify=y_ew)

# Track B: Performance (with grades)
X_pf, y_pf = prepare_data(df, include_prior_grades=True)
X_train_pf, X_test_pf, _, _ = train_test_split(X_pf, y_pf, test_size=0.2, random_state=42, stratify=y_pf)

# 1. & 4. Cross Validation Model Comparison (Early Warning)
print("Running CV for Early Warning Models...")
prep_ew = build_preprocessor(include_prior_grades=False)
models_ew = get_models(prep_ew)
cv_results_ew = cross_validate_models(models_ew, X_train_ew, y_train)
cv_results_ew.to_csv("reports/model_comparison.csv", index=False)
print(cv_results_ew)

# 5. Controlled Hyperparameter Tuning
print("Tuning Logistic Regression...")
best_lr, best_lr_params = tune_logistic_regression(X_train_ew, y_train, prep_ew)
print("Best LR params:", best_lr_params)

# 6. Threshold Optimization
print("Optimizing Threshold...")
# Get out-of-fold predictions
y_oof_proba = cross_val_predict(best_lr, X_train_ew, y_train, cv=5, method="predict_proba")[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_oof_proba)

threshold_results = []
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    y_pred_t = (y_oof_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_t).ravel()
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    threshold_results.append({
        'threshold': t,
        'precision': p,
        'recall': r,
        'f1': f1,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
    })
df_thresholds = pd.DataFrame(threshold_results)
df_thresholds.to_csv("reports/threshold_analysis.csv", index=False)

# Select threshold that gives at least 0.75 recall (identifying 75% of at-risk students)
candidate = df_thresholds[df_thresholds['recall'] >= 0.75].sort_values('precision', ascending=False)
best_threshold = candidate.iloc[0]['threshold'] if not candidate.empty else 0.5
print(f"Selected Threshold: {best_threshold}")

# 7. Final Held-Out Evaluation
print("Evaluating on Test Set...")
best_lr.fit(X_train_ew, y_train)
y_test_proba = best_lr.predict_proba(X_test_ew)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_metrics = {
    'roc_auc': roc_auc_score(y_test, y_test_proba),
    'pr_auc': average_precision_score(y_test, y_test_proba),
    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
    'f1': f1_score(y_test, y_test_pred),
    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
    'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
    'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
}

# 8. Compare with Performance Model
print("Running Performance Model Comparison...")
prep_pf = build_preprocessor(include_prior_grades=True)
models_pf = get_models(prep_pf)
cv_results_pf = cross_validate_models(models_pf, X_train_pf, y_train)
best_lr_pf = models_pf["LogisticRegression"]
best_lr_pf.fit(X_train_pf, y_train)
y_test_proba_pf = best_lr_pf.predict_proba(X_test_pf)[:, 1]
pf_roc = roc_auc_score(y_test, y_test_proba_pf)
pf_pr = average_precision_score(y_test, y_test_proba_pf)

# 9. Feature Importance
print("Calculating Permutation Importance...")
importances = permutation_importance(best_lr, X_test_ew, y_test, n_repeats=10, random_state=42, scoring='average_precision')
feature_importances = pd.DataFrame({'feature': X_test_ew.columns, 'importance': importances.importances_mean}).sort_values('importance', ascending=False)
print("Top features:")
print(feature_importances.head(5))

# 10. Fairness Audit
print("Running Fairness Audit...")
fairness_results = []
for group_col in ['sex', 'address']:
    for group_val in X_test_ew[group_col].unique():
        idx = X_test_ew[group_col] == group_val
        y_g = y_test[idx]
        y_p = y_test_pred[idx]
        tn, fp, fn, tp = confusion_matrix(y_g, y_p, labels=[0,1]).ravel()
        fairness_results.append({
            'group_feature': group_col,
            'group_value': group_val,
            'sample_count': len(y_g),
            'positive_prevalence': np.mean(y_g),
            'accuracy': (tp + tn) / len(y_g) if len(y_g) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': f1_score(y_g, y_p, zero_division=0),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
df_fairness = pd.DataFrame(fairness_results)
df_fairness.to_csv("reports/fairness_metrics.csv", index=False)
with open("reports/fairness_report.md", "w") as f:
    f.write("# Fairness Audit\n\n")
    f.write(df_fairness.to_csv(index=False))

# 11. Artifact Management
print("Saving Artifacts...")
os.makedirs("models", exist_ok=True)
joblib.dump(best_lr, "models/early_warning_model.pkl")

metadata = {
    "model_name": "early_warning_model",
    "model_type": "Pipeline with LogisticRegression",
    "experiment_track": "Early Warning",
    "included_features": list(X_train_ew.columns),
    "excluded_features": ["grade_period1", "grade_period2", "final_grade"],
    "target_definition": "final_grade < 10",
    "risk_threshold": 10,
    "probability_decision_threshold": float(best_threshold),
    "random_seed": 42,
    "test_metrics": test_metrics,
    "performance_model_comparison": {
        "ew_roc_auc": test_metrics['roc_auc'],
        "pf_roc_auc": float(pf_roc),
        "ew_pr_auc": test_metrics['pr_auc'],
        "pf_pr_auc": float(pf_pr)
    }
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done. Sprint 2 Execution Completed.")
