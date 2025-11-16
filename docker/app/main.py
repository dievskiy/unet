"""FastAPI service for running UNet ONNX inference."""

from __future__ import annotations

import io
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError


def _parse_tuple(raw_value: str, separator: str = ",") -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_value.split(separator))


def _parse_color_palette(raw_value: str) -> np.ndarray:
    colors: List[List[int]] = []
    for color_triplet in raw_value.split(";"):
        if not color_triplet:
            continue
        rgb = [int(channel.strip()) for channel in color_triplet.split(",")]
        if len(rgb) != 3:
            raise ValueError("Each CLASS_COLORS entry must list exactly three comma separated integers.")
        colors.append(rgb)
    if not colors:
        raise ValueError("CLASS_COLORS must include at least one color.")
    return np.asarray(colors, dtype=np.uint8)


MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model_unet.onnx")
TARGET_SIZE = _parse_tuple(os.getenv("TARGET_SIZE", "256,256"))
CLASS_COLORS = _parse_color_palette(os.getenv("CLASS_COLORS", "0,0,0;0,255,0;255,0,255"))

if len(TARGET_SIZE) != 2:
    raise ValueError("TARGET_SIZE must contain two integers in the format 'height,width'.")


class UnetOnnxRunner:
    def __init__(self, model_path: str, target_hw: Tuple[int, int], class_colors: np.ndarray):
        providers = os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(";")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.target_hw = target_hw
        self.class_colors = class_colors

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        resized = pil_image.resize((self.target_hw[1], self.target_hw[0]), Image.BILINEAR)
        np_image = np.asarray(resized, dtype=np.float32) / 255.0
        np_image = np.transpose(np_image, (2, 0, 1))
        return np.expand_dims(np_image, axis=0)

    def _mask_to_image(self, mask: np.ndarray) -> Image.Image:
        colored = self.class_colors[mask]
        return Image.fromarray(colored)

    def predict(self, pil_image: Image.Image) -> Image.Image:
        model_input = self._preprocess(pil_image)
        outputs = self.session.run([self.output_name], {self.input_name: model_input})
        logits = outputs[0]
        mask = np.argmax(logits, axis=1)[0].astype(np.uint8)
        return self._mask_to_image(mask)


app = FastAPI(title="UNet Segmentation Service")


@app.on_event("startup")
def _load_runner():
    app.state.runner = UnetOnnxRunner(MODEL_PATH, TARGET_SIZE, CLASS_COLORS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_class=StreamingResponse)
async def infer(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{file.content_type}'.")
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    mask_image = app.state.runner.predict(pil_image)

    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
