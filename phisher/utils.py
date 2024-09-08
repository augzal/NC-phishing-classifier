import io
from typing import Tuple

from keras.src.backend import tensorflow as tf_backend
from keras.src.utils import image_utils
from PIL import Image


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
