from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import models
from sklearn.metrics import classification_report

from .config import config


class PhishingClassifier:
    def __init__(self, model_path: Path) -> None:
        self.model = models.load_model(model_path)
        self.labels = config["model"]["efficientnet"]["labels"]

    def classify_single_image(self, img: tf.Tensor) -> str:
        """Predict class label for provided image

        Args:
            img (tf.Tensor): Image as tensor

        Returns:
            str: predicted class label
        """
        img = np.expand_dims(img, axis=0)
        probabilities = self.model.predict(img)
        return self._get_class_label(probabilities)[0]

    def classify_multiple_images(self, ds: tf.data.Dataset) -> list[str]:
        """Predict class for each image in dataset.

        Args:
            ds (tf.data.Dataset): dataset with images

        Returns:
            list[str]: list of predicted class names
        """
        scores = self.model.predict(ds)
        return self._get_class_label(scores)

    def evaluate_predictions(self, labels: list[str], predictions: list[str]) -> dict:
        """Evaluate performance of classifier

        Args:
            labels (list[str]): list of labels
            predictions (list[str]): list of predicted labels

        Returns:
            dict: dictionary with evaluation results
        """
        return classification_report(labels, predictions, output_dict=True)

    @staticmethod
    def _get_labels_from_dataset(ds: tf.data.Dataset) -> list[str]:
        """Get actual labels for each image in dataset.

        Args:
            ds (tf.data.Dataset): dataset with images

        Returns:
            list[str]: list of image labels
        """
        labels = []
        for _, batch_class_ids in ds:
            for class_ids in batch_class_ids:
                labels.append(ds.class_names[np.argmax(class_ids)])
        return labels

    def _get_class_label(self, pred_proba: np.array) -> list[str]:
        """Get class label for model predicted probabilities

        Args:
            pred_proba (np.array): predicted probabilities of each class

        Returns:
            list[str]: predicted labels
        """
        pred_labels = []
        pred_classes = np.argmax(pred_proba, axis=1)
        for pred_class in pred_classes:
            pred_labels.append(self.labels[pred_class])
        return pred_labels
