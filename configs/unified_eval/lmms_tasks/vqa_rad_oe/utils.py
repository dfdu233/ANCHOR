"""Local lmms-eval adapter for the frozen official VQA-RAD OE manifest."""

from pathlib import Path
import re
import string

from PIL import Image


IMAGE_ROOT = Path("/home/dbw/datasets/public/vqa_rad_hf/test_images")


def _normalize(text: str) -> str:
    text = str(text).lower().replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.findall(r"\w+", text))


def doc_to_visual(doc):
    path = IMAGE_ROOT / doc["img_name"]
    if not path.is_file():
        raise FileNotFoundError(f"VQA-RAD image is missing: {path}")
    with Image.open(path) as image:
        return [image.convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Keep the frozen common-protocol prompt exactly equal to the source question.
    return str(doc["question"])


def process_results(doc, results):
    if len(results) != 1:
        raise ValueError(f"Expected one generated response, received {len(results)}")
    prediction = str(results[0])
    return {"normalized_exact": float(_normalize(prediction) == _normalize(doc["answer"]))}

