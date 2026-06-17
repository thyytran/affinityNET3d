"""FastAPI inference server for protein-ligand binding affinity prediction.

Exposes a /predict endpoint that accepts a protein structure (PDB) and a
ligand (SDF/MOL2), extracts a point cloud from the complex, and returns the
predicted binding affinity from a trained PointNet / Point Transformer model.

Run with:
    serve-affinity
    # or
    uvicorn pointcloud_affinity.api:app --reload
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# These imports assume the package layout from setup.py. Adjust if your
# extraction / model code lives elsewhere.
from pointcloud_affinity.data.extract import complex_to_point_cloud
from pointcloud_affinity.models import build_model

logger = logging.getLogger("pointcloud_affinity.api")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration (override via environment variables)
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best.pt")
MODEL_NAME = os.environ.get("MODEL_NAME", "point_transformer")
DEVICE = os.environ.get(
    "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
POCKET_CUTOFF = float(os.environ.get("POCKET_CUTOFF", "8.0"))  # angstroms
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Holds loaded artifacts; populated on startup.
state: dict = {}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictionResponse(BaseModel):
    affinity: float = Field(..., description="Predicted binding affinity (pKd/pKi).")
    model: str = Field(..., description="Model architecture used for inference.")
    num_points: int = Field(..., description="Points in the extracted cloud.")
    units: str = Field(default="pK", description="Affinity units (-log10 Kd/Ki).")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# --------------------------------------------------------------------------- #
# Lifespan: load model once at startup
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model '%s' from %s on %s", MODEL_NAME, MODEL_PATH, DEVICE)
    if not Path(MODEL_PATH).exists():
        # Don't hard-crash; /healthz will report model_loaded=False so the
        # container can still start and surface a clear error on /predict.
        logger.error("Checkpoint not found at %s", MODEL_PATH)
    else:
        model = build_model(MODEL_NAME)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Support both raw state_dict and {"model_state": ...} checkpoints.
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        state["model"] = model
        logger.info("Model loaded successfully.")
    yield
    state.clear()


app = FastAPI(
    title="Point Cloud Binding Affinity",
    description="Protein-ligand binding affinity prediction from 3D point clouds.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _save_upload(upload: UploadFile, suffix: str) -> str:
    """Stream an upload to a temp file, enforcing a size cap."""
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename} exceeds {MAX_UPLOAD_BYTES} byte limit.",
        )
    if not data:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _run_inference(protein_path: str, ligand_path: str) -> tuple[float, int]:
    """Extract a point cloud and run the model. Returns (affinity, num_points)."""
    model = state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MODEL_PATH and server logs.",
        )

    coords, features = complex_to_point_cloud(
        protein_path, ligand_path, cutoff=POCKET_CUTOFF
    )
    num_points = int(coords.shape[0])
    if num_points == 0:
        raise HTTPException(
            status_code=422,
            detail="No atoms found within the pocket cutoff; check inputs.",
        )

    # (1, N, 3) coords + (1, N, F) features, batch size 1.
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    feats_t = torch.as_tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        pred = model(coords_t, feats_t)
    return float(pred.squeeze().item()), num_points


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="model" in state,
        device=DEVICE,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    protein: UploadFile = File(..., description="Protein structure (.pdb)."),
    ligand: UploadFile = File(..., description="Ligand structure (.sdf or .mol2)."),
) -> PredictionResponse:
    protein_suffix = Path(protein.filename or "protein.pdb").suffix or ".pdb"
    ligand_suffix = Path(ligand.filename or "ligand.sdf").suffix or ".sdf"

    protein_path = await _save_upload(protein, protein_suffix)
    ligand_path = await _save_upload(ligand, ligand_suffix)

    try:
        affinity, num_points = _run_inference(protein_path, ligand_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface extraction/model errors
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        for p in (protein_path, ligand_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return PredictionResponse(
        affinity=round(affinity, 4),
        model=MODEL_NAME,
        num_points=num_points,
    )


# --------------------------------------------------------------------------- #
# Entry point for the `serve-affinity` console script
# --------------------------------------------------------------------------- #
def main() -> None:
    import uvicorn

    uvicorn.run(
        "pointcloud_affinity.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=bool(os.environ.get("RELOAD", "")),
    )


if __name__ == "__main__":
    main()