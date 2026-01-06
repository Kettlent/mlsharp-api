import uuid
from pathlib import Path

import numpy as np
import torch

from sharp.api.model import get_model
from sharp.cli.predict import predict_image
from sharp.cli.render import render_gaussians
from sharp.utils import io
from sharp.utils.gaussians import save_ply, SceneMetaData

OUTPUT_ROOT = Path("/tmp/sharp_outputs")


def run_inference(image_bytes: bytes, filename: str) -> Path:
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input image
    image_path = job_dir / filename
    image_path.write_bytes(image_bytes)

    # Load image
    image, _, f_px = io.load_rgb(image_path)
    height, width = image.shape[:2]

    predictor, device = get_model()

    # Predict gaussians
    gaussians = predict_image(
        predictor=predictor,
        image=image,
        f_px=f_px,
        device=device,
    )

    # Save PLY
    ply_path = job_dir / "scene.ply"
    save_ply(gaussians, f_px, (height, width), ply_path)

    # Render videos
    metadata = SceneMetaData(
        focal_length_px=f_px,
        resolution_px=(width, height),
        color_space="linearRGB",
    )

    render_path = job_dir / "render.mp4"
    render_gaussians(
        gaussians=gaussians,
        metadata=metadata,
        output_path=render_path,
    )

    return job_dir
