import asyncio
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from epoch_go.agents.crew import EpochGoCrewRunner
from epoch_go.core.project_manager import ProjectManager
from epoch_go.tools.benchmark_runner import BenchmarkRunner
from epoch_go.tools.data_profiler import DataProfiler
from epoch_go.tools.dynamic_server import DynamicServerManager


def test_data_profiler():
    sample_csv = Path("sample_data/customer_churn.csv")
    assert sample_csv.exists(), "Sample customer_churn.csv should exist"

    profile = DataProfiler.profile_csv(sample_csv, target_col="churn")
    assert profile["total_rows"] == 300
    assert "churn" in profile["columns"]
    assert len(profile["missing_values"]) > 0, "Missing values should be detected"
    print("✅ test_data_profiler passed!")


def test_end_to_end_pipeline():
    test_projects_dir = Path("tests/scratch_projects")
    if test_projects_dir.exists():
        shutil.rmtree(test_projects_dir)
    test_projects_dir.mkdir(parents=True, exist_ok=True)

    pm = ProjectManager(base_dir=test_projects_dir)
    with open("sample_data/customer_churn.csv", "rb") as f:
        csv_bytes = f.read()

    # 1. Project creation
    meta = pm.create_project(
        name="Test Churn Project",
        csv_file_bytes=csv_bytes,
        csv_filename="customer_churn.csv",
        target_column="churn",
        task_type="classification",
    )
    project_id = meta["id"]
    assert project_id is not None
    p_dir = pm.get_project_dir(project_id)

    # 2. Run CrewAI pipeline
    runner = EpochGoCrewRunner(pm=pm)
    result = asyncio.run(runner.run_pipeline(project_id))
    assert result["success"] is True, f"Pipeline failed: {result.get('error')}"

    # Verify Agent 1 artifacts
    assert (p_dir / "preprocessing.py").exists(), "preprocessing.py should exist"
    assert (p_dir / "preprocessing.md").exists(), "preprocessing.md should exist"
    assert (p_dir / "cleaned_data.csv").exists(), "cleaned_data.csv should exist"
    assert (p_dir / "preprocessor.joblib").exists(), "preprocessor.joblib should exist"

    # Verify Agent 2 artifacts
    assert (p_dir / "model_building.py").exists(), "model_building.py should exist"
    assert (p_dir / "model_building.md").exists(), "model_building.md should exist"
    assert (p_dir / "model.joblib").exists(), "model.joblib should exist"

    # Verify Agent 3 artifacts
    assert (p_dir / "serve.py").exists(), "serve.py should exist"

    # 3. Test dynamic prediction
    schema_info = DynamicServerManager.inspect_model_and_preprocessor(p_dir)
    assert schema_info["has_model"] is True
    assert len(schema_info["feature_schema"]) > 0

    sample_record = {f["name"]: f["sample"] for f in schema_info["feature_schema"]}
    pred_res = DynamicServerManager.predict(p_dir, [sample_record])
    assert "predictions" in pred_res
    assert len(pred_res["predictions"]) == 1
    assert "prediction" in pred_res["predictions"][0]
    print(f"✅ Prediction succeeded: {pred_res['predictions'][0]}")

    # Clean up test dir
    shutil.rmtree(test_projects_dir)
    print("✅ test_end_to_end_classification passed!")


def test_end_to_end_regression():
    test_projects_dir = Path("tests/scratch_reg_projects")
    if test_projects_dir.exists():
        shutil.rmtree(test_projects_dir)
    test_projects_dir.mkdir(parents=True, exist_ok=True)

    pm = ProjectManager(base_dir=test_projects_dir)
    with open("sample_data/house_prices.csv", "rb") as f:
        csv_bytes = f.read()

    # 1. Project creation
    meta = pm.create_project(
        name="House Price Predictor",
        csv_file_bytes=csv_bytes,
        csv_filename="house_prices.csv",
        target_column="price",
        task_type="regression",
    )
    project_id = meta["id"]
    p_dir = pm.get_project_dir(project_id)

    # 2. Run CrewAI pipeline
    runner = EpochGoCrewRunner(pm=pm)
    result = asyncio.run(runner.run_pipeline(project_id))
    assert result["success"] is True, f"Pipeline failed: {result.get('error')}"

    # Verify artifacts
    assert (p_dir / "preprocessing.py").exists()
    assert (p_dir / "preprocessing.md").exists()
    assert (p_dir / "model_building.py").exists()
    assert (p_dir / "model_building.md").exists()
    assert (p_dir / "model.joblib").exists()
    assert (p_dir / "serve.py").exists()

    # Dynamic prediction
    schema_info = DynamicServerManager.inspect_model_and_preprocessor(p_dir)
    sample_record = {f["name"]: f["sample"] for f in schema_info["feature_schema"]}
    pred_res = DynamicServerManager.predict(p_dir, [sample_record])
    assert "predictions" in pred_res
    print(f"✅ Regression prediction succeeded: {pred_res['predictions'][0]}")

    shutil.rmtree(test_projects_dir)
    print("✅ test_end_to_end_regression passed!")


if __name__ == "__main__":
    test_data_profiler()
    test_end_to_end_pipeline()
    test_end_to_end_regression()
    print("\n🎉 ALL TESTS (CLASSIFICATION & REGRESSION) PASSED SUCCESSFULLY!")
