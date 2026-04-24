"""Training and calibration routines."""

"""Training entry points for ABSA models."""

from .train_sentiment import train_and_evaluate_sentiment

__all__ = ["train_and_evaluate_sentiment"]
