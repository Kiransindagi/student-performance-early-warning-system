import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score

def get_models(preprocessor):
    return {
        "Dummy": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DummyClassifier(strategy='prior', random_state=42))]),
        "LogisticRegression": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))]),
        "RandomForest": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))]),
        "HistGradientBoosting": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))])
    }

def cross_validate_models(models, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1'
    }
    
    results = []
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=1, return_train_score=False)
        results.append({
            'Model': name,
            'mean_roc_auc': np.mean(cv_results['test_roc_auc']),
            'std_roc_auc': np.std(cv_results['test_roc_auc']),
            'mean_pr_auc': np.mean(cv_results['test_pr_auc']),
            'std_pr_auc': np.std(cv_results['test_pr_auc']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'mean_f1': np.mean(cv_results['test_f1']),
        })
    return pd.DataFrame(results)

def tune_logistic_regression(X, y, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__class_weight': ['balanced', None]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=8, cv=skf, scoring='average_precision', random_state=42, n_jobs=1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

def tune_random_forest(X, y, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=20, cv=skf, scoring='average_precision', random_state=42, n_jobs=1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
