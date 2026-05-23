import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_score_density_plot(scores, output_path: str) -> None:
    """
    Сохраняет график плотности распределения predicted scores.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, density=True)
    plt.title("Density distribution of predicted scores")
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info("Score density plot saved to: %s", output_path)
