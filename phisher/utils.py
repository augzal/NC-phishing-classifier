import io
from typing import Tuple

from keras.src.backend import tensorflow as tf_backend
from keras.src.utils import image_utils
from PIL import Image
import pandas as pd


def preprocess_image(image: Image, image_shape: Tuple[int, int] = (256, 256)) -> Image:
    """Preprocess image before feeding to classification model.

    Args:
        image (Image): Image to process
        image_shape (Tuple[int, int], optional):image dimentions. Defaults to (256, 256).

    Returns:
        Image: preprocessed image
    """
    resized_img = image_utils.smart_resize(
        image,
        image_shape,
        interpolation="bilinear",
        data_format=None,
        backend_module=tf_backend,
    )
    return resized_img


def read_image(uploaded_file: bytes) -> Image:
    """Read image from bytes

    Args:
        uploaded_file (bytes): Uploaded file as bytes

    Returns:
        Image: read image
    """
    return Image.open(io.BytesIO(uploaded_file))


def save_csv(
    image_paths: list[str],
    labels: list[str],
    predicted_labels: list[str],
    csv_path: str,
):
    """Write predictions to csv

    Args:
        file_paths (list[str]): Paths of images which are classified
        labels (list[str]): Labels of images
        predicted_labels (list[str]): Predicted labels of images
        csv_path (str): Path where file should be saved
    """
    df = pd.DataFrame(
        {
            "file_paths": image_paths,
            "labels": labels,
            "predicted_labels": predicted_labels,
        }
    )
    df.to_csv(csv_path)
