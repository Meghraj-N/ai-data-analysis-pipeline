"""
src/eda.py
────────────────────────────────────────────────────────────────
Exploratory Data Analysis Module
Author : Meghraj Nikalje | B.Sc. Data Science, Mumbai University
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from src.logger import get_logger

log = get_logger(__name__)


class EDAEngine:
    """
    Automated EDA: descriptive stats, distributions,
    correlations, revenue analysis, and cohort breakdowns.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[dict] = None):
        self.df     = df
        self.config = config or {}
        self.stats: dict = {}
        log.info("EDAEngine initialised — %d rows × %d cols", *df.shape)

    # ── Core Stats ─────────────────────────────────────────────
    def compute_descriptive_stats(self) -> "EDAEngine":
        desc = self.df.describe(include="all").to_dict()
        self.stats["descriptive"] = {
            col: {k: (float(v) if pd.notna(v) and isinstance(v, (int, float, np.number)) else str(v))
                  for k, v in vals.items()}
            for col, vals in desc.items()
        }
        log.info("Descriptive stats computed for %d columns", len(self.df.columns))
        return self

    # ── Revenue Analysis ───────────────────────────────────────
    def compute_revenue_analysis(self) -> "EDAEngine":
        df = self.df

        if "total_revenue" not in df.columns:
            log.warning("'total_revenue' column not found — skipping revenue analysis.")
            return self

        rev = df["total_revenue"]

        self.stats["revenue"] = {
            "total":    round(float(rev.sum()), 2),
            "mean":     round(float(rev.mean()), 2),
            "median":   round(float(rev.median()), 2),
            "std":      round(float(rev.std()), 2),
            "min":      round(float(rev.min()), 2),
            "max":      round(float(rev.max()), 2),
            "p25":      round(float(rev.quantile(0.25)), 2),
            "p75":      round(float(rev.quantile(0.75)), 2),
            "p90":      round(float(rev.quantile(0.90)), 2),
        }

        # Category breakdown
        if "category" in df.columns:
            cat = df.groupby("category")["total_revenue"].agg(["sum", "mean", "count"]).round(2)
            cat["revenue_share_%"] = (cat["sum"] / cat["sum"].sum() * 100).round(1)
            self.stats["revenue_by_category"] = cat.rename(
                columns={"sum": "total", "mean": "avg_order", "count": "orders"}
            ).to_dict()

        # Region breakdown
        if "region" in df.columns:
            reg = df.groupby("region")["total_revenue"].agg(["sum", "mean", "count"]).round(2)
            reg["revenue_share_%"] = (reg["sum"] / reg["sum"].sum() * 100).round(1)
            self.stats["revenue_by_region"] = reg.rename(
                columns={"sum": "total", "mean": "avg_order", "count": "orders"}
            ).to_dict()

        # Monthly trend
        if "order_month" in df.columns:
            monthly = (
                df.groupby("order_month")["total_revenue"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "revenue", "count": "orders"})
                .round(2)
            )
            self.stats["monthly_trend"] = monthly.to_dict()

        # Sales rep performance
        if "sales_rep" in df.columns:
            rep = (
                df.groupby("sales_rep")["total_revenue"]
                .agg(["sum", "mean", "count"])
                .sort_values("sum", ascending=False)
                .round(2)
                .rename(columns={"sum": "total", "mean": "avg_order", "count": "orders"})
            )
            self.stats["sales_rep_performance"] = rep.to_dict()

        log.info("Revenue analysis complete — total: ₹%s", self.stats["revenue"]["total"])
        return self

    # ── Order Status Analysis ──────────────────────────────────
    def compute_order_analysis(self) -> "EDAEngine":
        if "status" not in self.df.columns:
            return self

        status_vc = self.df["status"].value_counts()
        self.stats["order_status"] = {
            "counts": status_vc.to_dict(),
            "pct":    (status_vc / len(self.df) * 100).round(1).to_dict(),
        }

        if "total_revenue" in self.df.columns:
            rev_by_status = self.df.groupby("status")["total_revenue"].sum().round(2).to_dict()
            self.stats["order_status"]["revenue_by_status"] = rev_by_status

        log.info("Order status analysis done: %s", self.stats["order_status"]["counts"])
        return self

    # ── Correlation Matrix ─────────────────────────────────────
    def compute_correlations(self) -> "EDAEngine":
        num_df = self.df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return self

        corr = num_df.corr().round(3)
        threshold = float(self.config.get("correlation_threshold", 0.7))

        # Find high-correlation pairs (excluding self-correlation)
        high_corr = []
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                val = corr.loc[c1, c2]
                if abs(val) >= threshold:
                    high_corr.append({"col_a": c1, "col_b": c2, "correlation": float(val)})

        self.stats["correlations"] = {
            "matrix":         {c: corr[c].to_dict() for c in cols},
            "high_corr_pairs": high_corr,
        }

        log.info("Correlation matrix computed. High-corr pairs (≥%.1f): %d", threshold, len(high_corr))
        return self

    # ── Run All ────────────────────────────────────────────────
    def run_all(self) -> dict:
        log.info("Running full EDA suite...")
        (
            self.compute_descriptive_stats()
                .compute_revenue_analysis()
                .compute_order_analysis()
                .compute_correlations()
        )
        log.info("EDA complete — %d analysis blocks generated.", len(self.stats))
        return self.stats

    def save(self, path: str = "report/summary_stats.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        log.info("EDA stats saved → %s", path)
