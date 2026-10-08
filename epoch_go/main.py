import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from epoch_go.agents.crew import crew_runner
from epoch_go.config import settings
from epoch_go.core.project_manager import project_manager
from epoch_go.core.websocket_manager import ws_manager
from epoch_go.tools.dynamic_server import DynamicServerManager

app = FastAPI(
    title="EpochGo",
    description="Autonomous Multi-Agent AutoML Platform by Epochlypse Research",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    ollama_ok = False
    models = []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if res.status_code == 200:
                ollama_ok = True
                models = [m["name"] for m in res.json().get("models", [])]
    except Exception:
        pass

    return {
        "status": "healthy",
        "ollama_connected": ollama_ok,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "active_model": settings.OLLAMA_MODEL,
        "available_models": models,
        "projects_count": len(project_manager.list_projects()),
    }


@app.get("/api/projects")
def list_projects():
    return project_manager.list_projects()


@app.post("/api/projects")
async def create_project(
    name: str = Form(...),
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    description: Optional[str] = Form(""),
    auto_run: bool = Form(True),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are currently supported")

    content = await file.read()
    metadata = project_manager.create_project(
        name=name,
        csv_file_bytes=content,
        csv_filename=file.filename,
        target_column=target_column,
        task_type=task_type,
        description=description or "",
    )

    project_id = metadata["id"]

    if auto_run:
        # Launch CrewAI pipeline in background task
        asyncio.create_task(crew_runner.run_pipeline(project_id))

    return metadata


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    meta = project_manager.get_metadata(project_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Project not found")

    p_dir = project_manager.get_project_dir(project_id)
    meta["has_preprocessing_report"] = (p_dir / "preprocessing.md").exists()
    meta["has_preprocessing_script"] = (p_dir / "preprocessing.py").exists()
    meta["has_model_report"] = (p_dir / "model_building.md").exists()
    meta["has_model_script"] = (p_dir / "model_building.py").exists()
    meta["has_model_artifact"] = (p_dir / "model.joblib").exists()
    meta["has_serve_script"] = (p_dir / "serve.py").exists()
    return meta


@app.post("/api/projects/{project_id}/run")
async def run_pipeline(project_id: str):
    meta = project_manager.get_metadata(project_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Project not found")

    current_status = meta.get("status")
    if current_status in ["preprocessing", "model_building", "serving"]:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is already actively executing (stage: {current_status}). Duplicate runs are locked.",
        )

    asyncio.create_task(crew_runner.run_pipeline(project_id))
    return {"message": "CrewAI pipeline started", "project_id": project_id}


@app.get("/api/projects/{project_id}/files/{filename}")
def get_file_content(project_id: str, filename: str):
    allowed_files = [
        "preprocessing.md",
        "model_building.md",
        "preprocessing.py",
        "model_building.py",
        "serve.py",
        "metadata.json",
    ]
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Access denied to requested file")

    content = project_manager.get_file_content(project_id, filename)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    return {"filename": filename, "content": content}


@app.get("/api/projects/{project_id}/download/{filename}")
def download_file(project_id: str, filename: str):
    allowed_files = [
        "raw_data.csv",
        "cleaned_data.csv",
        "model.joblib",
        "preprocessor.joblib",
        "preprocessing.py",
        "model_building.py",
        "serve.py",
        "preprocessing.md",
        "model_building.md",
    ]
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Access denied to requested download")

    path = project_manager.get_file_path(project_id, filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    return FileResponse(
        path=path,
        filename=filename,
        media_type="application/octet-stream",
    )


@app.get("/api/projects/{project_id}/schema")
def get_project_schema(project_id: str):
    p_dir = project_manager.get_project_dir(project_id)
    if not p_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    return DynamicServerManager.inspect_model_and_preprocessor(p_dir)


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


@app.post("/api/projects/{project_id}/predict")
def predict_endpoint(project_id: str, payload: PredictRequest):
    p_dir = project_manager.get_project_dir(project_id)
    if not p_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        results = DynamicServerManager.predict(p_dir, payload.data)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    success = project_manager.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully", "id": project_id}


@app.websocket("/ws/projects/{project_id}")
async def websocket_project(websocket: WebSocket, project_id: str):
    await ws_manager.connect_project(project_id, websocket)
    meta = project_manager.get_metadata(project_id)
    if meta:
        current_status = meta.get("status", "created")
        step_map = {"created": 0, "preprocessing": 1, "model_building": 2, "serving": 3, "completed": 4, "failed": -1}
        try:
            await websocket.send_json({
                "type": "status",
                "status": current_status,
                "step": step_map.get(current_status, 0),
                "best_model": meta.get("best_model"),
                "logs": meta.get("logs", []),
                "error": meta.get("error"),
            })
        except Exception:
            pass

    try:
        while True:
            # Keep-alive ping/pong
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect_project(project_id, websocket)
    except Exception:
        ws_manager.disconnect_project(project_id, websocket)


# Serve React UI static assets if built, else serve embedded modern dashboard
UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"
if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(UI_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = UI_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(UI_DIST / "index.html")
else:
    # High-fidelity embedded SPA dashboard for immediate out-of-the-box usage
    @app.get("/", response_class=HTMLResponse)
    async def serve_embedded_ui():
        embedded_ui_path = Path(__file__).resolve().parent / "templates" / "index.html"
        if embedded_ui_path.exists():
            with open(embedded_ui_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>EpochGo is running.</h1>"
