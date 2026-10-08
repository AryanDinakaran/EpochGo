import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from epoch_go.config import settings


class ProjectManager:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.PROJECTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "-", text)
        return text.strip("-")

    def create_project(
        self,
        name: str,
        csv_file_bytes: bytes,
        csv_filename: str,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,  # "classification", "regression", or None (auto)
        description: str = "",
    ) -> Dict[str, Any]:
        slug = self._slugify(name)
        short_id = uuid.uuid4().hex[:6]
        project_id = f"{slug}-{short_id}" if slug else f"project-{short_id}"
        project_dir = self.base_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        raw_csv_path = project_dir / "raw_data.csv"
        with open(raw_csv_path, "wb") as f:
            f.write(csv_file_bytes)

        # Quick inspection of the dataset
        df = pd.read_csv(raw_csv_path)
        columns = df.columns.tolist()

        # Auto-detect target column if not provided
        if not target_column or target_column not in columns:
            target_column = self._infer_target_column(df)

        # Auto-detect task type if not provided
        if not task_type:
            task_type = self._infer_task_type(df, target_column)

        metadata = {
            "id": project_id,
            "name": name,
            "description": description,
            "original_filename": csv_filename,
            "target_column": target_column,
            "task_type": task_type,
            "rows": len(df),
            "columns": columns,
            "num_columns": len(columns),
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "logs": [],
            "benchmark_results": [],
            "best_model": None,
            "serve_port": None,
            "error": None,
        }

        self.save_metadata(project_id, metadata)
        return metadata

    def _infer_target_column(self, df: pd.DataFrame) -> str:
        candidates = ["target", "label", "class", "churn", "price", "outcome", "survived", "status"]
        col_lower_map = {col.lower(): col for col in df.columns}
        for cand in candidates:
            if cand in col_lower_map:
                return col_lower_map[cand]
        # Fallback to the last column
        return df.columns[-1]

    def _infer_task_type(self, df: pd.DataFrame, target_col: str) -> str:
        if target_col not in df.columns:
            return "classification"
        target_series = df[target_col].dropna()
        # If string / categorical or low unique count compared to rows, it's classification
        num_unique = target_series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)

        if not is_numeric or num_unique <= 10 or (num_unique < len(target_series) * 0.05):
            return "classification"
        return "regression"

    def get_project_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id

    def get_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self.get_project_dir(project_id) / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_metadata(self, project_id: str, metadata: Dict[str, Any]) -> None:
        metadata["updated_at"] = datetime.utcnow().isoformat()
        meta_path = self.get_project_dir(project_id) / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def update_status(
        self,
        project_id: str,
        status: str,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = self.get_metadata(project_id)
        if meta:
            meta["status"] = status
            if error:
                meta["error"] = error
            if extra:
                meta.update(extra)
            self.save_metadata(project_id, meta)

    def append_log(self, project_id: str, message: str, agent: str = "System") -> None:
        meta = self.get_metadata(project_id)
        if meta:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "message": message,
            }
            if "logs" not in meta:
                meta["logs"] = []
            meta["logs"].append(entry)
            self.save_metadata(project_id, meta)

    def list_projects(self) -> List[Dict[str, Any]]:
        projects = []
        if not self.base_dir.exists():
            return projects

        for item in sorted(self.base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if item.is_dir() and (item / "metadata.json").exists():
                try:
                    with open(item / "metadata.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        # Add quick artifact availability flags
                        meta["has_preprocessing_report"] = (item / "preprocessing.md").exists()
                        meta["has_preprocessing_script"] = (item / "preprocessing.py").exists()
                        meta["has_model_report"] = (item / "model_building.md").exists()
                        meta["has_model_script"] = (item / "model_building.py").exists()
                        meta["has_model_artifact"] = (item / "model.joblib").exists()
                        meta["has_serve_script"] = (item / "serve.py").exists()
                        projects.append(meta)
                except Exception:
                    continue
        return projects

    def get_file_content(self, project_id: str, filename: str) -> Optional[str]:
        file_path = self.get_project_dir(project_id) / filename
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def get_file_path(self, project_id: str, filename: str) -> Optional[Path]:
        file_path = self.get_project_dir(project_id) / filename
        if file_path.exists():
            return file_path
        return None

    def delete_project(self, project_id: str) -> bool:
        project_dir = self.get_project_dir(project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)
            return True
        return False


project_manager = ProjectManager()
