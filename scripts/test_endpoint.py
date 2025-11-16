#!/usr/bin/env python3
"""Send a dataset image through the deployed SageMaker UNet endpoint.

The script mirrors the same resize + normalization steps used during training,
invokes the Triton-backed endpoint with a JSON inference request, and saves the
resulting segmentation mask.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize

from src.config import UnetConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-name",
        help="Name of the SageMaker endpoint (alternatively pass --invoke-url)",
    )
    parser.add_argument(
        "--invoke-url",
        help="HTTPS URL output by the CDK stack; region/endpoint are derived from it",
    )
    parser.add_argument(
        "--region",
        help="AWS region override. Defaults to the region inferred from credentials or the invoke URL.",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the training config JSON. Defaults to ./config.json",
    )
    parser.add_argument(
        "--image-path",
        help="Specific image to test. Defaults to the first *.jpg under config.data_input_dir",
    )
    parser.add_argument(
        "--input-name",
        default="input",
        help="Triton input tensor name inside the ONNX graph (default: input)",
    )
    parser.add_argument(
        "--output-mask",
        default="artifacts/prediction.png",
        help="Where to save the rendered segmentation mask",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> UnetConfig:
    cfg_file = Path(config_path)
    if cfg_file.exists():
        return UnetConfig.from_json(str(cfg_file))
    return UnetConfig()


def _pick_image(image_path: Optional[str], dataset_dir: str) -> Path:
    if image_path:
        return Path(image_path)

    candidates = sorted(Path(dataset_dir).glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files found under {dataset_dir}. Provide --image-path explicitly.")
    return candidates[0]


def _preprocess_image(image_path: Path, target_size: Tuple[int, int]) -> torch.Tensor:
    resize = Resize(target_size)
    tensor = read_image(str(image_path), mode=ImageReadMode.RGB).float()
    tensor = resize(tensor)
    tensor /= 255.0
    return tensor.unsqueeze(0).contiguous()


def _build_triton_payload(batch: torch.Tensor, input_name: str) -> bytes:
    array = batch.numpy()
    payload = {
        "inputs": [
            {
                "name": input_name,
                "shape": list(array.shape),
                "datatype": "FP32",
                "data": array.flatten().tolist(),
            }
        ]
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _invoke_endpoint(endpoint_name: str, payload: bytes, *, region: Optional[str]) -> bytes:
    client = boto3.client("sagemaker-runtime", region_name=region)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    return response["Body"].read()


def _parse_triton_output(body: bytes) -> torch.Tensor:
    message = json.loads(body.decode("utf-8"))
    if "outputs" not in message or not message["outputs"]:
        raise RuntimeError(f"Unexpected response from endpoint: {message}")

    output_block = message["outputs"][0]
    shape = tuple(output_block["shape"])
    logits = torch.tensor(output_block["data"], dtype=torch.float32).reshape(shape)
    return logits


def _save_mask(mask: torch.Tensor, destination: Path, num_classes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    max_class = max(num_classes - 1, 1)
    scale = 255 // max_class
    arr = (mask.cpu().numpy().astype("uint8") * scale).clip(0, 255)
    Image.fromarray(arr, mode="L").save(destination)


def _infer_endpoint_and_region(endpoint_name: Optional[str], invoke_url: Optional[str]) -> Tuple[str, Optional[str]]:
    if endpoint_name:
        return endpoint_name, None

    if not invoke_url:
        raise ValueError("Either --endpoint-name or --invoke-url must be provided.")

    parsed = urlparse(invoke_url)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2 or path_parts[0] != "endpoints":
        raise ValueError(f"Invoke URL does not look like a SageMaker Runtime URL: {invoke_url}")

    endpoint = path_parts[1]
    match = re.search(r"runtime\.sagemaker\.([a-z0-9-]+)\.amazonaws\.com", parsed.netloc)
    region = match.group(1) if match else None
    return endpoint, region


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config)

    endpoint_name, region_hint = _infer_endpoint_and_region(args.endpoint_name, args.invoke_url)
    region = args.region or region_hint

    image_path = _pick_image(args.image_path, config.data_input_dir)
    batch = _preprocess_image(image_path, config.target_size)
    payload = _build_triton_payload(batch, args.input_name)

    response = _invoke_endpoint(endpoint_name, payload, region=region)
    logits = _parse_triton_output(response)
    mask = torch.argmax(logits, dim=1).squeeze(0)

    output_path = Path(args.output_mask)
    _save_mask(mask, output_path, config.num_classes)

    print(f"Invoked endpoint '{endpoint_name}' in region '{region or 'default'}'.")
    print(f"Input image: {image_path}")
    print(f"Saved prediction mask to: {output_path}")
    print("Class distribution:", torch.bincount(mask.flatten()).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
