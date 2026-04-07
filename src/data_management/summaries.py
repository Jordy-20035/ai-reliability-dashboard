"""Compact distribution summaries for baselines / dashboards (not full data)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def distribution_summary(df: pd.DataFrame, *, max_categories: int = 15) -> dict[str, Any]:
    """
    Small JSON-serializable snapshot: numeric quantiles + top categories per object column.
    """
    out: dict[str, Any] = {"n_rows": len(df), "columns": {}}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean) == 0:
                out["columns"][col] = {"kind": "numeric", "empty": True}
            else:
                out["columns"][col] = {
                    "kind": "numeric",
                    "mean": float(clean.mean()),
                    "std": float(clean.std()) if len(clean) > 1 else 0.0,
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "quantiles": {
                        "q05": float(np.quantile(clean, 0.05)),
                        "q50": float(np.quantile(clean, 0.50)),
                        "q95": float(np.quantile(clean, 0.95)),
                    },
                }
        else:
            vc = s.astype("string").value_counts().head(max_categories)
            out["columns"][col] = {
                "kind": "categorical",
                "n_unique": int(s.nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }
    return out
