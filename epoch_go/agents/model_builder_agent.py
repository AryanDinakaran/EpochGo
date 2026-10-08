import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from epoch_go.config import settings
from epoch_go.tools.benchmark_runner import BenchmarkRunner


class ModelBuilderAgent:
    """Agent 2: Model Benchmark & Hyperparameter Tuning Specialist."""

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        self.ollama_url = (ollama_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    async def _query_llm(self, prompt: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                response = await res
                if response.status_code == 200:
                    return response.json().get("response")
        except Exception as e:
            print(f"[ModelBuilderAgent] Ollama query failed ({e}), using deterministic generator.")
        return None

    async def run(
        self,
        project_dir: Path,
        target_column: str,
        task_type: str,
        log_callback=None,
    ) -> Dict[str, Any]:
        cleaned_csv = project_dir / "cleaned_data.csv"
        script_path = project_dir / "model_building.py"
        report_path = project_dir / "model_building.md"

        if log_callback:
            await log_callback(f"Agent 2 (Model Builder): Starting benchmark of 7 baseline models for {task_type}...")

        # Run 7 baseline models + Top 3 RandomizedSearchCV tuning in a background thread to keep event loop free
        results = await asyncio.to_thread(
            BenchmarkRunner.run_benchmark_and_tuning,
            cleaned_csv_path=cleaned_csv,
            target_column=target_column,
            task_type=task_type,
            output_dir=project_dir,
        )

        champ_name = results["champion_model_name"]
        champ_metrics = results["champion_metrics"]

        if log_callback:
            await log_callback(
                f"Agent 2 (Model Builder): Baseline benchmark completed! Evaluated {len(results['baseline_results'])} models."
            )
            for b in results["baseline_results"][:3]:
                await log_callback(
                    f"  ⭐ {b['model_name']}: {b.get('primary_metric', 'Score')} = {b.get('primary_score')} (time: {b.get('fit_time')}s)"
                )
            await log_callback(
                f"Agent 2 (Model Builder): RandomizedSearchCV tuning completed! Winning model: '{champ_name}' "
                f"with {champ_metrics.get('primary_metric')} = {champ_metrics.get('primary_score')}."
            )
            await log_callback("Agent 2 (Model Builder): Generating reproducible model_building.py script and report...")

        # Generate reproducible training script
        script_code = self._generate_script(results, target_column, task_type)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        # Generate comprehensive markdown report
        report_md = await self._generate_report(results, target_column, task_type)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        if log_callback:
            await log_callback("Agent 2 (Model Builder): Completed! model.joblib, model_building.py, and model_building.md ready.")

        return {
            "success": True,
            "benchmark_results": results,
            "champion_model_name": champ_name,
            "script_path": str(script_path),
            "report_path": str(report_path),
        }

    def _generate_script(self, results: Dict[str, Any], target_col: str, task_type: str) -> str:
        champ_name = results["champion_model_name"].replace(" (Tuned)", "")
        champ_params = json.dumps(results["champion_params"], indent=4)

        model_instantiation = ""
        import_stmt = ""

        if "XGBoost" in champ_name:
            import_stmt = "from xgboost import XGBClassifier, XGBRegressor"
            cls_name = "XGBClassifier" if task_type == "classification" else "XGBRegressor"
            model_instantiation = f"{cls_name}(**params, random_state=42)"
        elif "LightGBM" in champ_name:
            import_stmt = "from lightgbm import LGBMClassifier, LGBMRegressor"
            cls_name = "LGBMClassifier" if task_type == "classification" else "LGBMRegressor"
            model_instantiation = f"{cls_name}(**params, random_state=42, verbose=-1)"
        elif "CatBoost" in champ_name:
            import_stmt = "from catboost import CatBoostClassifier, CatBoostRegressor"
            cls_name = "CatBoostClassifier" if task_type == "classification" else "CatBoostRegressor"
            model_instantiation = f"{cls_name}(**params, random_state=42, verbose=0)"
        elif "Random Forest" in champ_name:
            import_stmt = "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor"
            cls_name = "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"
            model_instantiation = f"{cls_name}(**params, random_state=42)"
        elif "Extra Trees" in champ_name:
            import_stmt = "from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor"
            cls_name = "ExtraTreesClassifier" if task_type == "classification" else "ExtraTreesRegressor"
            model_instantiation = f"{cls_name}(**params, random_state=42)"
        elif "Gradient Boosting" in champ_name:
            import_stmt = "from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor"
            cls_name = "GradientBoostingClassifier" if task_type == "classification" else "GradientBoostingRegressor"
            model_instantiation = f"{cls_name}(**params, random_state=42)"
        elif "Logistic Regression" in champ_name:
            import_stmt = "from sklearn.linear_model import LogisticRegression"
            model_instantiation = f"LogisticRegression(**params, random_state=42, max_iter=1000)"
        else:
            import_stmt = "from sklearn.linear_model import Ridge"
            model_instantiation = f"Ridge(**params, random_state=42)"

        code = f'''"""
Reproducible Model Training & Evaluation Pipeline
Generated by EpochGo Agent 2 (Model Building Architect)
Champion Model: {results["champion_model_name"]}
Target: {target_col} ({task_type})
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
{import_stmt}

def train_champion_model():
    data_path = Path("cleaned_data.csv")
    model_output_path = Path("model.joblib")

    if not data_path.exists():
        raise FileNotFoundError("cleaned_data.csv not found")

    df = pd.read_csv(data_path)
    target_col = "{target_col}"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {champ_params}

    print(f"Training champion model '{results['champion_model_name']}'...")
    model = {model_instantiation}
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train Score: {{train_score:.4f}} | Test Score: {{test_score:.4f}}")

    joblib.dump(model, model_output_path)
    print(f"✅ Model saved successfully to {{model_output_path}}")

if __name__ == "__main__":
    train_champion_model()
'''
        return code

    async def _generate_report(
        self,
        results: Dict[str, Any],
        target_col: str,
        task_type: str,
    ) -> str:
        prompt = f"""You are EpochGo's Lead Machine Learning Architect (Epochlypse Research). Write a professional, detailed Markdown report named `model_building.md`.
Results Summary:
Target Column: {target_col} ({task_type})
Baseline Results: {json.dumps(results['baseline_results'])}
Tuned Results: {json.dumps(results['tuned_results'])}
Champion Model: {results['champion_model_name']}
Champion Params: {json.dumps(results['champion_params'])}
Feature Importances: {json.dumps(results['feature_importances'])}

Requirements:
1. Executive Summary & Champion Model announcement
2. 7 Baseline Benchmark Leaderboard (Comparison Markdown Table of all 7 models with primary metrics, training time)
3. Top 3 Hyperparameter Tuning Analysis (Table comparing baseline score vs tuned score, best hyperparameters)
4. Top 10 Feature Importances table
5. Production Deployment Recommendation

Output ONLY the markdown text.
"""
        llm_output = await self._query_llm(prompt)
        if llm_output and len(llm_output.strip()) > 150:
            return llm_output.strip()

        # Deterministic high-quality markdown template fallback
        baseline_rows = ""
        for rank, b in enumerate(results["baseline_results"], 1):
            name = b.get("model_name", "Model")
            score = b.get("primary_score", 0)
            time_val = b.get("fit_time", 0)
            metric_col = f"{score:.4f}" if isinstance(score, (int, float)) and score > -900 else "N/A"
            badge = "🏆 Rank 1" if rank == 1 else f"Rank {rank}"
            baseline_rows += f"| {badge} | **{name}** | {metric_col} | {time_val}s |\n"

        tuned_rows = ""
        for t in results["tuned_results"]:
            name = t.get("model_name", "")
            base_score = t.get("baseline_primary_score", 0)
            tuned_score = t.get("primary_score", 0)
            impr = t.get("improvement", 0)
            params = "<br>".join([f"`{k}={v}`" for k, v in list(t.get("best_params", {}).items())[:4]])
            tuned_rows += f"| **{name}** | {base_score} | **{tuned_score}** | +{impr} | {params} |\n"

        feat_rows = ""
        for fi in results["feature_importances"]:
            feat_rows += f"| `{fi['feature']}` | {fi['importance'] * 100:.2f}% |\n"
        if not feat_rows:
            feat_rows = "| All Features | Uniform contribution |\n"

        champ = results["champion_model_name"]
        champ_score = results["champion_metrics"].get("primary_score")
        p_metric = results["champion_metrics"].get("primary_metric", "Score")

        return f"""# Machine Learning Benchmark & Hyperparameter Tuning Report

**Generated by**: EpochGo Model Building Agent (Epochlypse Research)  
**Task Type**: {task_type.capitalize()}  
**Target Variable**: `{target_col}`  
**Winning Model**: **{champ}** ({p_metric}: **{champ_score}**)  

---

## 1. Executive Summary

EpochGo evaluated **7 candidate baseline algorithms** across gradient boosting, ensemble decision trees, and linear architectures. The top 3 performers were selected for hyperparameter optimization via `RandomizedSearchCV`.

**Winner**: **{champ}** was designated as the champion model and serialized to `model.joblib`.

---

## 2. 7 Baseline Model Leaderboard

All baseline algorithms were trained on identical stratified splits with zero hyperparameter bias.

| Standing | Algorithm | {p_metric} | Training Time |
| :--- | :--- | :--- | :--- |
{baseline_rows}

---

## 3. Top 3 Hyperparameter Optimization

`RandomizedSearchCV` was deployed across multi-dimensional search spaces with cross-validation.

| Candidate Model | Baseline {p_metric} | Tuned {p_metric} | Improvement | Optimal Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
{tuned_rows}

---

## 4. Top Feature Importances

Calculated from the champion model's feature attribution weights.

| Feature Name | Relative Weight (%) |
| :--- | :--- |
{feat_rows}

---

## 5. Artifact Delivery
- **Model Binary**: `model.joblib`
- **Training Pipeline**: `model_building.py`
- **Next Stage**: Handover to Agent 3 (Model Serving) for FastAPI microservice deployment.
"""
