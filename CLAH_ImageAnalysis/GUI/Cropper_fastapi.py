import os
import cv2
import numpy as np
import isx
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import text_dict
import io

app = FastAPI()

file_tag = text_dict()["file_tag"]

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the repo root directory
PROJECT_ROOT = paths.get_directory_of_repo_from_file()

# Get the TSX output directory
TSX_OUTPUT_DIR = PROJECT_ROOT / "frontend4WA" / "Cropper" / "dist"

# Mount static files
app.mount(
    "/dist",
    StaticFiles(directory=str(TSX_OUTPUT_DIR)),
    name="dist",
)
app.mount(
    "/assets",
    StaticFiles(directory=str(TSX_OUTPUT_DIR / "assets")),
    name="assets",
)
app.mount(
    "/docs",
    StaticFiles(directory=str(PROJECT_ROOT / "docs")),
    name="docs",
)


class CropCoordinates(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


@app.get("/api/list_isxd_files/")
def list_isxd_files(directory: str):
    try:
        files = [
            f for f in os.listdir(directory) if f.endswith(".isxd") and "CNMF" not in f
        ]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/load_isxd/")
def load_isxd(file_path: str):
    try:
        movie = isx.Movie.read(file_path)
        return {"total_frames": movie.timing.num_samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/get_frame/")
def get_frame(file_path: str, frame_idx: int):
    try:
        movie = isx.Movie.read(file_path)
        frame = movie.get_frame_data(frame_idx)
        # Normalize and convert to RGB
        normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_rgb = cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2RGB)
        _, buffer = cv2.imencode(file_tag["JPG"], frame_rgb)

        # Create a BytesIO object and write the image data to it
        img_io = io.BytesIO(buffer.tobytes())

        # Return the image as a streaming response with proper content type
        return StreamingResponse(img_io, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export_crop_coords/")
def export_crop_coords(file_path: str, coords: CropCoordinates):
    try:
        crop_coords = [(coords.x1, coords.y1), (coords.x2, coords.y2)]
        json_fname = os.path.join(os.path.dirname(file_path), "crop_dims.json")
        with open(json_fname, "w") as f:
            json.dump(crop_coords, f)
        return {"message": "Crop coordinates exported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/list_directory/")
def list_directory(directory: str):
    try:
        entries = os.listdir(directory)
        entries.sort()
        if directory == "/":
            entries = [entry for entry in entries if entry in ["home", "mnt"]]

        directories = [
            entry for entry in entries if os.path.isdir(os.path.join(directory, entry))
        ]
        files = [
            entry for entry in entries if os.path.isfile(os.path.join(directory, entry))
        ]
        files = [
            entry
            for entry in files
            if entry.endswith(file_tag["ISXD"])
            and (file_tag["CNMFE"] not in entry or file_tag["CNMFE2"] not in entry)
        ]

        return {"directories": directories, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    return FileResponse(str(TSX_OUTPUT_DIR / "index.html"))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")


if __name__ == "__main__":
    main()
