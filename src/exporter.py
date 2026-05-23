import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_output_dir(output_dir: str) -> None:
    """
    Создаёт output-директорию, если её нет.
    """
    os.makedirs(output_dir, exist_ok=True)


def save_submission(
    original_df: pd.DataFrame,
    predictions,
    output_path: str,
) -> None:
    """
    Сохраняет файл в формате sample_submission.csv.

    Текущий формат:
    - index
    - prediction

    Если в Kaggle sample_submission.csv другие колонки,
    здесь нужно заменить названия колонок.
    """
    submission = pd.DataFrame(
        {
            "index": original_df.index,
            "prediction": predictions,
        }
    )

    submission.to_csv(output_path, index=False)

    logger.info("Submission saved to: %s", output_path)


def save_feature_importances(
    feature_importances: dict[str, float],
    output_path: str,
) -> None:
    """
    Сохраняет top-5 feature importances в JSON.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            feature_importances,
            file,
            ensure_ascii=False,
            indent=4,
        )

    logger.info("Feature importances saved to: %s", output_path)
