import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_input_file(file_path: str) -> pd.DataFrame:
    """
    Загружает входной CSV-файл для скоринга.
    """
    logger.info("Loading input file: %s", file_path)

    df = pd.read_csv(file_path)

    logger.info("Input file loaded. Shape: %s", df.shape)

    return df