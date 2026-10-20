import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataProfiler:
    """Profiles raw and preprocessed datasets to extract comprehensive statistics."""

    @staticmethod
    def profile_csv(csv_path: Path, target_col: Optional[str] = None) -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        return DataProfiler.profile_dataframe(df, target_col)

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        total_rows = len(df)
        total_cols = len(df.columns)

        # Basic summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        missing_summary = {}
        for col in df.columns:
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                missing_summary[col] = {
                    "count": cnt,
                    "percentage": round((cnt / total_rows) * 100, 2),
                }

        # Outliers in numeric columns using IQR
        outlier_summary = {}
        skewness_summary = {}
        for col in numeric_cols:
            if col == target_col:
                continue
            series = df[col].dropna()
            if len(series) < 5:
                continue
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            if len(outliers) > 0:
                outlier_summary[col] = {
                    "count": int(len(outliers)),
                    "percentage": round((len(outliers) / total_rows) * 100, 2),
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                }
            skew_val = float(series.skew())
            if not np.isnan(skew_val):
                skewness_summary[col] = round(skew_val, 3)

        # High cardinality in categorical & Granular Categorical Routing
        categorical_summary = {}
        low_card_cols = []
        high_card_cols = []
        extreme_card_cols = []

        for col in categorical_cols:
            if col == target_col:
                continue
            n_unique = int(df[col].nunique())
            categorical_summary[col] = {
                "unique_count": n_unique,
                "sample_values": df[col].dropna().unique()[:5].tolist(),
                "high_cardinality": n_unique > 20,
            }
            if n_unique <= 20:
                low_card_cols.append(col)
            elif n_unique <= 500:
                high_card_cols.append(col)
            else:
                extreme_card_cols.append(col)

        # Automated Identifier (ID / Primary Key / UUID) detection
        id_columns = []
        for col in df.columns:
            if col == target_col:
                continue
            n_unique = int(df[col].nunique())
            uniqueness_ratio = (n_unique / total_rows) if total_rows > 0 else 0
            col_lower = str(col).lower()
            is_id_name = any(k in col_lower for k in ["_id", "id", "uuid", "guid", "row_id", "index", "pk"])
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_float = pd.api.types.is_float_dtype(df[col])
            
            # Non-predictive identifiers:
            # 1. String/Object columns with uniqueness > 95% (e.g., UUIDs, hash strings, row keys)
            # 2. Columns explicitly named like an ID with uniqueness > 80% (unless it is a continuous float feature without ID naming)
            is_identifier = False
            if not is_numeric and uniqueness_ratio > 0.95 and total_rows > 30:
                is_identifier = True
            elif is_id_name and not is_float and uniqueness_ratio > 0.80 and total_rows > 30:
                is_identifier = True

            if is_identifier:
                id_columns.append(col)
                # Remove from categorical / numeric lists if present
                if col in low_card_cols:
                    low_card_cols.remove(col)
                if col in high_card_cols:
                    high_card_cols.remove(col)
                if col in extreme_card_cols:
                    extreme_card_cols.remove(col)
                if col in numeric_cols:
                    numeric_cols.remove(col)

        # Multicollinearity among numeric features
        high_correlations = []
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        c1, c2 = numeric_cols[i], numeric_cols[j]
                        if c1 != target_col and c2 != target_col:
                            val = corr_matrix.loc[c1, c2]
                            if not np.isnan(val) and val > 0.85:
                                high_correlations.append({
                                    "col1": c1,
                                    "col2": c2,
                                    "correlation": round(float(val), 3),
                                })
            except Exception:
                pass

        # Target column inspection
        target_info = None
        if target_col and target_col in df.columns:
            t_series = df[target_col].dropna()
            is_num = pd.api.types.is_numeric_dtype(t_series)
            target_info = {
                "name": target_col,
                "is_numeric": bool(is_num),
                "unique_values": int(t_series.nunique()),
                "missing": int(df[target_col].isna().sum()),
                "sample_distribution": t_series.value_counts().head(5).to_dict() if not is_num or t_series.nunique() <= 10 else "continuous",
            }

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": missing_summary,
            "outliers": outlier_summary,
            "skewness": skewness_summary,
            "categorical_summary": categorical_summary,
            "high_correlations": high_correlations,
            "recommend_pca": len(numeric_cols) > 20 or len(high_correlations) > 5,
            "id_columns": id_columns,
            "low_card_columns": low_card_cols,
            "high_card_columns": high_card_cols,
            "extreme_card_columns": extreme_card_cols,
            "target_info": target_info,
            "head": df.head(3).to_dict(orient="records"),
        }


def profile_to_markdown_summary(profile: Dict[str, Any]) -> str:
    """Helper to convert structured profile into human-readable summary for LLM context."""
    return json.dumps(profile, indent=2)
