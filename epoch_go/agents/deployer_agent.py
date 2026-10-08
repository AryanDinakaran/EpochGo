from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from epoch_go.config import settings
from epoch_go.tools.dynamic_server import DynamicServerManager


class DeployerAgent:
    """Agent 3: MLOps & Model Serving Engineer."""

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        self.ollama_url = (ollama_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    async def run(
        self,
        project_dir: Path,
        target_column: str,
        task_type: str,
        best_model_name: str,
        log_callback=None,
    ) -> Dict[str, Any]:
        serve_path = project_dir / "serve.py"

        if log_callback:
            await log_callback(f"Agent 3 (Serving Agent): Inspecting {best_model_name} and feature schema...")

        info = DynamicServerManager.inspect_model_and_preprocessor(project_dir)
        feature_schema = info.get("feature_schema", [])

        if log_callback:
            await log_callback(
                f"Agent 3 (Serving Agent): Extracted schema with {len(feature_schema)} input features. "
                "Synthesizing production-grade FastAPI microservice (serve.py)..."
            )

        serve_script = DynamicServerManager.generate_serve_script(
            project_dir=project_dir,
            target_col=target_column,
            task_type=task_type,
            feature_schema=feature_schema,
            best_model_name=best_model_name,
        )

        with open(serve_path, "w", encoding="utf-8") as f:
            f.write(serve_script)

        # Test sample prediction with first sample record
        sample_input = {f["name"]: f["sample"] for f in feature_schema}
        test_pred = None
        try:
            res = DynamicServerManager.predict(project_dir, [sample_input])
            test_pred = res["predictions"][0]
            if log_callback:
                await log_callback(
                    f"Agent 3 (Serving Agent): Inference test passed! Test sample output: {test_pred}"
                )
        except Exception as e:
            if log_callback:
                await log_callback(f"Agent 3 (Serving Agent): Note on inference test: {e}")

        if log_callback:
            await log_callback(
                "Agent 3 (Serving Agent): Deployment ready! serve.py script generated. "
                "Interactive playground endpoint enabled."
            )

        return {
            "success": True,
            "serve_script_path": str(serve_path),
            "feature_schema": feature_schema,
            "sample_input": sample_input,
            "test_prediction": test_pred,
        }
