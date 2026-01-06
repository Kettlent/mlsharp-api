import io
import zipfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from sharp.api.model import load_model
from sharp.api.inference import run_inference

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SHARP REST API", version="1.0")


@app.on_event("startup")
def startup():
    load_model()  # loads once on startup


@app.get("/health")
def health():
    return {"status": "ok", "cuda": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()

    job_dir = run_inference(image_bytes, file.filename)

    # Zip results
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(job_dir / "scene.ply", "scene.ply")
        z.write(job_dir / "render.mp4", "render.mp4")
        z.write(job_dir / "render.depth.mp4", "render.depth.mp4")

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=sharp_output.zip"},
    )
