import logging
import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

sys.path.append(os.path.abspath("./src"))

from exporter import (ensure_output_dir, save_feature_importances,
                      save_submission)
from loader import load_input_file
from preprocessing import load_train_data, run_preproc
from reports import save_score_density_plot
from scorer import get_feature_importances, load_model, make_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/service.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class ProcessingService:
    """
    Основной сервис обработки файлов.
    """

    def __init__(self):
        logger.info("Initializing ProcessingService...")

        self.input_dir = Path("/app/input")
        self.output_dir = Path("/app/output")

        ensure_output_dir(str(self.output_dir))

        self.train = load_train_data()
        self.model = load_model()

        logger.info("Service initialized")

    def process_single_file(self, file_path: str) -> None:
        """
        Полный inference pipeline:
        1. Загрузка входного файла
        2. Препроцессинг
        3. Скоринг
        4. Сохранение sample_submission.csv
        5. Сохранение feature_importances_top5.json
        6. Сохранение prediction_score_density.png
        """
        try:
            file_path = Path(file_path)

            if file_path.name.startswith("."):
                logger.info("Skipping hidden file: %s", file_path)
                return

            if file_path.suffix.lower() != ".csv":
                logger.info("Skipping non-csv file: %s", file_path)
                return

            logger.info("Processing file: %s", file_path)

            original_df = load_input_file(str(file_path))

            input_df = original_df.drop(
                columns=["name_1", "name_2", "street", "post_code"],
                errors="ignore",
            )

            logger.info("Starting preprocessing")
            processed_df = run_preproc(
                train=self.train,
                input_df=input_df,
            )

            logger.info("Making prediction")
            scores, predictions = make_predictions(
                model=self.model,
                processed_df=processed_df,
            )

            submission_path = self.output_dir / "sample_submission.csv"
            feature_importances_path = self.output_dir / "feature_importances_top5.json"
            density_plot_path = self.output_dir / "prediction_score_density.png"

            logger.info("Preparing submission file")
            save_submission(
                original_df=original_df,
                predictions=predictions,
                output_path=str(submission_path),
            )

            logger.info("Preparing feature importances file")
            feature_importances = get_feature_importances(
                model=self.model,
                feature_names=list(processed_df.columns),
                top_n=5,
            )

            save_feature_importances(
                feature_importances=feature_importances,
                output_path=str(feature_importances_path),
            )

            logger.info("Preparing score density plot")
            save_score_density_plot(
                scores=scores,
                output_path=str(density_plot_path),
            )

            logger.info("Processing completed successfully")

        except Exception as error:
            logger.error(
                "Error processing file %s: %s",
                file_path,
                error,
                exc_info=True,
            )


class FileHandler(FileSystemEventHandler):
    """
    Обработчик событий в папке input.
    """

    def __init__(self, service: ProcessingService):
        self.service = service

    def on_created(self, event):
        """
        Срабатывает, когда в папке input появляется новый файл.
        """
        if event.is_directory:
            return

        if event.src_path.endswith(".csv"):
            logger.info("New file detected: %s", event.src_path)

            # Пауза нужна, чтобы файл успел полностью скопироваться.
            time.sleep(1)

            self.service.process_single_file(event.src_path)

    def on_moved(self, event):
        """
        Срабатывает, если файл был перемещён в папку input.
        """
        if event.is_directory:
            return

        if event.dest_path.endswith(".csv"):
            logger.info("File moved to input directory: %s", event.dest_path)

            # Пауза нужна, чтобы файл успел полностью переместиться.
            time.sleep(1)

            self.service.process_single_file(event.dest_path)


def process_existing_files(service: ProcessingService) -> None:
    """
    Обрабатывает CSV-файлы, которые уже лежат в input до запуска контейнера.
    """
    logger.info("Checking existing files in input directory")

    for file_path in service.input_dir.glob("*.csv"):
        logger.info("Existing file found: %s", file_path)
        service.process_single_file(str(file_path))


if __name__ == "__main__":
    logger.info("Starting ML scoring service...")

    service = ProcessingService()

    process_existing_files(service)

    observer = Observer()
    observer.schedule(
        FileHandler(service),
        path=str(service.input_dir),
        recursive=False,
    )

    observer.start()

    logger.info("File observer started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        observer.stop()

    observer.join()
