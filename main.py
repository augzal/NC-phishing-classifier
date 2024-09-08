from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile

from phisher.config import config
from phisher.pipeline import Pipeline

app = FastAPI()
pipeline = Pipeline(config["model"]["efficientnet"]["model_path"])


@app.post("/classify_image")
async def classify_image(file: UploadFile) -> str:
    """Classify a single image and return a predicted class name.

    Args:
        file (UploadFile): Image to be classified

    Returns:
        str: class name
    """
    file_content = await file.read()
    return pipeline.classify_image(file_content)


@app.post("/evaluate_classification")
def read_root(folder_path: Path = "data/test_data/") -> dict:
    """predict and evaluate model. Provided directory should contain folders with images. Folder names should correspond to image class names.

    Args:
        folder_path (Path, optional): Path to the folder with images. Defaults to "data/test_data/".

    Returns:
        dict: Dictionary with predicted labels and evaluation results
    """
    return pipeline.evaluate_classification(folder_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
