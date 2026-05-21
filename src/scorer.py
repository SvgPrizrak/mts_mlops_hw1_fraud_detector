import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

MODEL_PATH = "./models/my_catboost.cbm"
MODEL_THRESHOLD = 0.98


def load_model() -> CatBoostClassifier:
    """
    Загружает обученную CatBoost-модель.
    """
    logger.info("Importing pretrained model...")

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    logger.info("Pretrained model imported successfully")

    return model


def make_predictions(
    model: CatBoostClassifier,
    processed_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Делает inference.

    Возвращает:
    - scores: вероятности положительного класса;
    - predictions: бинарные предсказания 0/1.
    """
    scores = model.predict_proba(processed_df)[:, 1]
    predictions = (scores > MODEL_THRESHOLD).astype(int)

    logger.info("Prediction complete")

    return scores, predictions


def get_feature_importances(
    model: CatBoostClassifier,
    feature_names: list[str],
    top_n: int = 5,
) -> dict[str, float]:
    """
    Возвращает top-N feature importances.
    """
    importances = model.get_feature_importance()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).head(top_n)

    result = {
        row["feature"]: float(row["importance"])
        for _, row in importance_df.iterrows()
    }

    logger.info("Feature importances calculated")

    return result