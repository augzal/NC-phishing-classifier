from pathlib import Path
from typing import Tuple

from keras import utils

from .classifier import PhishingClassifier
from .config import config
from .utils import preprocess_image, read_image


class Pipeline:
    def __init__(self, model_path: Path):
        self.cls = PhishingClassifier(model_path)

    def classify_image(self, file_content: bytes) -> str:
        """Predict class of the image.

        Args:
            file_content (bytes): Uploaded image file

        Returns:
            str: predicted class of the image
        """
        image = read_image(file_content)
        processed_image = preprocess_image(image)
        return self.cls.classify_single_image(processed_image)

    def classify_images(self, folder_path: Path) -> Tuple[list[str], list[str]]:
        """Classify all the images within the provided directory. Directory should contain
          folders with images, folder names will be used to determine image class name.
        Args:
            folder_path (Path): Path to the folder with images.

        Returns:
            Tuple[list[str], list[str]]: tuple lists of labels and predicted labels
        """
        model_config = config["model"]["efficientnet"]
        ds = utils.image_dataset_from_directory(
            directory=folder_path,
            labels="inferred",
            label_mode="categorical",
            batch_size=model_config["batch_size"],
            image_size=(model_config["img_h"], model_config["img_w"]),
            shuffle=False,
        )
        predicted_labels = self.cls.classify_multiple_images(ds)
        labels = self.cls._get_labels_from_dataset(ds)
        return labels, predicted_labels

    def evaluate_classification(self, folder_path: Path) -> dict:
        """Classify images and evaluate model performance.

        Args:
            folder_path (Path): Path to the folder with images.

        Returns:
            dict: dictionary with predicted labels and classification report
        """
        labels, predicted_labels = self.classify_images(folder_path)
        classification_report = self.cls.evaluate_predictions(labels, predicted_labels)
        return {
            "predicted_labels": predicted_labels,
            "classification_report": classification_report,
        }
