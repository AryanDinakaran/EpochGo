import asyncio
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from epoch_go.agents.deployer_agent import DeployerAgent
from epoch_go.agents.model_builder_agent import ModelBuilderAgent
from epoch_go.agents.preprocessor_agent import PreprocessorAgent
from epoch_go.config import settings
from epoch_go.core.project_manager import project_manager, ProjectManager
from epoch_go.core.websocket_manager import ws_manager


class EpochGoCrewRunner:
    """Orchestrates the 3-agent CrewAI pipeline for an EpochGo AutoML project."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        pm: Optional[ProjectManager] = None,
    ):
        self.ollama_url = ollama_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.pm = pm or project_manager

        self.preprocessor = PreprocessorAgent(ollama_url=self.ollama_url, model=self.model)
        self.model_builder = ModelBuilderAgent(ollama_url=self.ollama_url, model=self.model)
        self.deployer = DeployerAgent(ollama_url=self.ollama_url, model=self.model)

    async def run_pipeline(self, project_id: str) -> Dict[str, Any]:
        """Runs Agent 1 -> Agent 2 -> Agent 3 sequentially with real-time updates."""
        meta = self.pm.get_metadata(project_id)
        if not meta:
            raise ValueError(f"Project '{project_id}' not found")

        project_dir = self.pm.get_project_dir(project_id)
        target_col = meta.get("target_column")
        task_type = meta.get("task_type", "classification")

        async def log_event(message: str, agent_name: str = "CrewAI"):
            self.pm.append_log(project_id, message, agent=agent_name)
            await ws_manager.broadcast_to_project(
                project_id,
                {
                    "type": "log",
                    "agent": agent_name,
                    "message": message,
                },
            )

        try:
            # Stage 1: Data Preprocessor Agent
            await log_event("🤖 Starting CrewAI Pipeline: Team of 3 Agents initialized.", agent_name="Orchestrator")
            self.pm.update_status(project_id, "preprocessing")
            await ws_manager.broadcast_to_project(project_id, {"type": "status", "status": "preprocessing", "step": 1})

            prep_res = await self.preprocessor.run(
                project_dir=project_dir,
                target_column=target_col,
                task_type=task_type,
                log_callback=lambda msg: log_event(msg, agent_name="Preprocessor Agent"),
            )

            # Stage 2: Model Building Agent
            self.pm.update_status(project_id, "model_building")
            await ws_manager.broadcast_to_project(project_id, {"type": "status", "status": "model_building", "step": 2})

            mb_res = await self.model_builder.run(
                project_dir=project_dir,
                target_column=target_col,
                task_type=task_type,
                log_callback=lambda msg: log_event(msg, agent_name="Model Builder Agent"),
            )

            champ_name = mb_res["champion_model_name"]
            meta["best_model"] = champ_name
            meta["benchmark_results"] = mb_res["benchmark_results"].get("baseline_results", [])
            meta["tuned_results"] = mb_res["benchmark_results"].get("tuned_results", [])
            meta["feature_importances"] = mb_res["benchmark_results"].get("feature_importances", [])
            self.pm.save_metadata(project_id, meta)

            # Stage 3: Model Serving Agent
            self.pm.update_status(project_id, "serving")
            await ws_manager.broadcast_to_project(project_id, {"type": "status", "status": "serving", "step": 3})

            dep_res = await self.deployer.run(
                project_dir=project_dir,
                target_column=target_col,
                task_type=task_type,
                best_model_name=champ_name,
                log_callback=lambda msg: log_event(msg, agent_name="Model Serving Agent"),
            )

            # Pipeline Completed!
            self.pm.update_status(project_id, "completed")
            await log_event("🎉 CrewAI Pipeline completed successfully! All artifacts and models ready.", agent_name="Orchestrator")
            await ws_manager.broadcast_to_project(
                project_id,
                {
                    "type": "status",
                    "status": "completed",
                    "step": 4,
                    "best_model": champ_name,
                },
            )

            return {
                "success": True,
                "project_id": project_id,
                "preprocessing": prep_res,
                "model_building": mb_res,
                "serving": dep_res,
            }

        except Exception as e:
            err_msg = f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
            self.pm.update_status(project_id, "failed", error=str(e))
            await log_event(f"❌ Error: {str(e)}", agent_name="Orchestrator")
            await ws_manager.broadcast_to_project(
                project_id,
                {"type": "status", "status": "failed", "error": str(e)},
            )
            return {"success": False, "error": str(e)}


crew_runner = EpochGoCrewRunner()
SlashAICrewRunner = EpochGoCrewRunner
