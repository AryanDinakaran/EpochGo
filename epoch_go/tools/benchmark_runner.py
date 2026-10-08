import time
import joblib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)

# Benchmark models
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)

# Advanced Gradient Boosters
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class BenchmarkRunner:
    """Executes 7 baseline models, selects top 3, and performs hyperparameter tuning."""

    @staticmethod
    def get_candidate_models(task_type: str) -> Dict[str, Any]:
        """Returns 7 candidate models for the task type."""
        models = {}
        if task_type == "classification":
            if HAS_XGBOOST:
                models["XGBoost"] = XGBClassifier(
                    eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
                )
            if HAS_LIGHTGBM:
                models["LightGBM"] = LGBMClassifier(
                    random_state=42, n_jobs=-1, verbose=-1
                )
            if HAS_CATBOOST:
                models["CatBoost"] = CatBoostClassifier(
                    random_state=42, verbose=0, thread_count=-1
                )
            models["Random Forest"] = RandomForestClassifier(random_state=42, n_jobs=-1)
            models["Extra Trees"] = ExtraTreesClassifier(random_state=42, n_jobs=-1)
            models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
            models["Logistic Regression"] = LogisticRegression(
                max_iter=1000, random_state=42
            )
            # Fallback if any booster is unavailable to ensure 7 models
            if len(models) < 7:
                models["AdaBoost"] = AdaBoostClassifier(random_state=42)
        else:
            if HAS_XGBOOST:
                models["XGBoost"] = XGBRegressor(
                    random_state=42, n_jobs=-1, verbosity=0
                )
            if HAS_LIGHTGBM:
                models["LightGBM"] = LGBMRegressor(
                    random_state=42, n_jobs=-1, verbose=-1
                )
            if HAS_CATBOOST:
                models["CatBoost"] = CatBoostRegressor(
                    random_state=42, verbose=0, thread_count=-1
                )
            models["Random Forest"] = RandomForestRegressor(random_state=42, n_jobs=-1)
            models["Extra Trees"] = ExtraTreesRegressor(random_state=42, n_jobs=-1)
            models["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
            models["Ridge Regression"] = Ridge(random_state=42)
            # Fallback if any booster is unavailable
            if len(models) < 7:
                models["Linear Regression"] = LinearRegression()
            if len(models) < 7:
                models["AdaBoost"] = AdaBoostRegressor(random_state=42)

        # Slice to exact 7 models
        return dict(list(models.items())[:7])

    @staticmethod
    def get_param_distributions(model_name: str, task_type: str) -> Dict[str, Any]:
        """Provides hyperparameter tuning search spaces for candidate models."""
        dist = {}
        if "XGBoost" in model_name:
            dist = {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        elif "LightGBM" in model_name:
            dist = {
                "n_estimators": [50, 100, 150, 200],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
            }
        elif "CatBoost" in model_name:
            dist = {
                "iterations": [50, 100, 150, 200],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
            }
        elif "Random Forest" in model_name or "Extra Trees" in model_name:
            dist = {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif "Gradient Boosting" in model_name:
            dist = {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 7],
                "subsample": [0.7, 0.8, 1.0],
            }
        elif "Logistic Regression" in model_name:
            dist = {
                "C": [0.01, 0.1, 1.0, 10.0, 50.0],
                "solver": ["lbfgs", "liblinear"],
            }
        elif "Ridge" in model_name:
            dist = {
                "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
            }
        else:
            dist = {
                "n_estimators": [50, 100, 150],
            }
        return dist

    @classmethod
    def run_benchmark_and_tuning(
        cls,
        cleaned_csv_path: Path,
        target_column: str,
        task_type: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Runs baseline benchmark of 7 models, tunes top 3, and saves winning model.joblib."""
        df = pd.read_csv(cleaned_csv_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {cleaned_csv_path}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Stratified train/test split for classification if possible
        stratify = y if task_type == "classification" and y.value_counts().min() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        candidate_models = cls.get_candidate_models(task_type)
        baseline_results: List[Dict[str, Any]] = []

        # 1. Baseline Evaluation
        for name, model in candidate_models.items():
            t0 = time.time()
            try:
                model.fit(X_train, y_train)
                fit_time = time.time() - t0
                y_pred = model.predict(X_test)
                eval_metrics = cls._calculate_metrics(task_type, y_test, y_pred, model, X_test)
                eval_metrics["model_name"] = name
                eval_metrics["fit_time"] = round(fit_time, 3)
                eval_metrics["instance"] = model
                baseline_results.append(eval_metrics)
            except Exception as e:
                baseline_results.append({
                    "model_name": name,
                    "error": str(e),
                    "primary_score": -999.0,
                    "fit_time": 0,
                    "instance": None,
                })

        # Sort baseline results by primary metric
        # (F1 or Accuracy for classification, R2 for regression)
        baseline_results.sort(key=lambda r: r.get("primary_score", -999), reverse=True)

        # 2. Select Top 3 Models for Hyperparameter Tuning
        top_3 = [r for r in baseline_results if r.get("instance") is not None][:3]
        tuned_results: List[Dict[str, Any]] = []

        scoring = "f1_weighted" if task_type == "classification" else "r2"

        for entry in top_3:
            name = entry["model_name"]
            base_model = candidate_models[name]
            param_dist = cls.get_param_distributions(name, task_type)

            t0 = time.time()
            try:
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_dist,
                    n_iter=10,
                    scoring=scoring,
                    cv=3,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0,
                )
                search.fit(X_train, y_train)
                tune_time = time.time() - t0
                best_estimator = search.best_estimator_
                y_pred = best_estimator.predict(X_test)

                metrics = cls._calculate_metrics(task_type, y_test, y_pred, best_estimator, X_test)
                metrics["model_name"] = f"{name} (Tuned)"
                metrics["base_model_name"] = name
                metrics["tune_time"] = round(tune_time, 3)
                metrics["best_params"] = search.best_params_
                metrics["instance"] = best_estimator
                metrics["baseline_primary_score"] = entry.get("primary_score")
                metrics["improvement"] = round(metrics["primary_score"] - entry.get("primary_score", 0), 4)
                tuned_results.append(metrics)
            except Exception as e:
                tuned_results.append({
                    "model_name": f"{name} (Tuned)",
                    "base_model_name": name,
                    "error": str(e),
                    "primary_score": entry.get("primary_score", -999),
                    "best_params": {},
                    "instance": entry.get("instance"),
                })

        # Sort tuned results by primary score
        tuned_results.sort(key=lambda r: r.get("primary_score", -999), reverse=True)
        champion_entry = tuned_results[0] if tuned_results else top_3[0]
        champion_model = champion_entry["instance"]

        # 3. Save Winning Model
        model_save_path = output_dir / "model.joblib"
        joblib.dump(champion_model, model_save_path)

        # 4. Feature Importances (if available)
        feature_names = X.columns.tolist()
        feature_importances = cls._extract_feature_importances(champion_model, feature_names)

        # Clean instances from dicts before serializing to json/metadata
        clean_baselines = [{k: v for k, v in r.items() if k != "instance"} for r in baseline_results]
        clean_tuned = [{k: v for k, v in r.items() if k != "instance"} for r in tuned_results]

        return {
            "task_type": task_type,
            "target_column": target_column,
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "baseline_results": clean_baselines,
            "tuned_results": clean_tuned,
            "champion_model_name": champion_entry["model_name"],
            "champion_metrics": {k: v for k, v in champion_entry.items() if k not in ("instance", "best_params")},
            "champion_params": champion_entry.get("best_params", {}),
            "feature_importances": feature_importances,
            "model_path": str(model_save_path),
        }

    @staticmethod
    def _calculate_metrics(
        task_type: str, y_true, y_pred, model, X_test
    ) -> Dict[str, Any]:
        if task_type == "classification":
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

            roc_auc = None
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)
                    if probs.shape[1] == 2:
                        roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
                    else:
                        roc_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted"))
            except Exception:
                pass

            return {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                "primary_score": round(f1, 4),
                "primary_metric": "F1-Score",
            }
        else:
            r2 = float(r2_score(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))

            return {
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "primary_score": round(r2, 4),
                "primary_metric": "R² Score",
            }

    @staticmethod
    def _extract_feature_importances(model, feature_names: List[str]) -> List[Dict[str, Any]]:
        importances = []
        try:
            if hasattr(model, "feature_importances_"):
                raw_vals = model.feature_importances_
                total = sum(raw_vals) if sum(raw_vals) > 0 else 1.0
                pairs = sorted(zip(feature_names, raw_vals), key=lambda x: x[1], reverse=True)
                for f, val in pairs[:10]:
                    importances.append({"feature": f, "importance": round(float(val / total), 4)})
            elif hasattr(model, "coef_"):
                raw_vals = np.abs(model.coef_).flatten()
                pairs = sorted(zip(feature_names, raw_vals), key=lambda x: x[1], reverse=True)
                for f, val in pairs[:10]:
                    importances.append({"feature": f, "importance": round(float(val), 4)})
        except Exception:
            pass
        return importances
