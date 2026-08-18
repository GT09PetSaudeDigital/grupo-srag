from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42


def build_models(random_state: int = RANDOM_STATE):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def build_gradient_boosting_sample_weight(y_train: pd.Series) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y_train)
