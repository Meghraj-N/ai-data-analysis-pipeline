"""
src/data_cleaner.py
────────────────────────────────────────────────────────────────
Professional Automated Data Cleaning Module
Author : Meghraj Nikalje | B.Sc. Data Science, Mumbai University
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from src.logger import get_logger

log = get_logger(__name__)


class DataCleaner:
    """
    End-to-end automated data cleaning pipeline.

    Steps (chained via .run_all()):
        1. remove_duplicates      — exact-row deduplication
        2. fix_data_types         — auto-parse dates & numerics
        3. handle_missing_values  — median/mode imputation
        4. normalise_text         — strip, collapse whitespace
        5. handle_outliers        — IQR cap or Z-score removal
        6. add_derived_columns    — feature engineering
    """

    def __init__(self, df: pd.DataFrame, config: Optional[dict] = None):
        self.df       = df.copy()
        self.original = df.copy()
        self._log: list[dict] = []
        self.config   = config or {}

        log.info("DataCleaner initialised — shape: %s", df.shape)

    # ── internal helper ───────────────────────────────────────
    def _record(self, step: str, detail: str) -> None:
        entry = {"step": step, "detail": detail, "ts": datetime.now().isoformat()}
        self._log.append(entry)
        log.info("[%s] %s", step, detail)

    # ── Step 1 ─────────────────────────────────────────────────
    def remove_duplicates(self) -> "DataCleaner":
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self._record("duplicates", f"Removed {removed} exact duplicate rows. Rows remaining: {len(self.df)}")
        return self

    # ── Step 2 ─────────────────────────────────────────────────
    def fix_data_types(self) -> "DataCleaner":
        changed = []
        for col in self.df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    parsed = pd.to_datetime(self.df[col], errors="coerce")
                    # Only accept if ≥80% of non-null values parsed successfully
                    success_rate = parsed.notna().sum() / max(self.df[col].notna().sum(), 1)
                    if success_rate >= 0.8:
                        self.df[col] = parsed
                        changed.append(f"'{col}' → datetime (success rate {success_rate:.0%})")
                except Exception:
                    pass
            elif self.df[col].dtype in ("object", "string"):
                converted = pd.to_numeric(self.df[col], errors="coerce")
                non_null  = converted.notna().sum()
                total     = self.df[col].notna().sum()
                if total > 0 and non_null / total >= 0.9:
                    self.df[col] = converted
                    changed.append(f"'{col}' → numeric (confidence {non_null/total:.0%})")

        detail = f"Type corrections: {', '.join(changed)}" if changed else "No type corrections needed."
        self._record("data_types", detail)
        return self

    # ── Step 3 ─────────────────────────────────────────────────
    def handle_missing_values(self) -> "DataCleaner":
        strategy = self.config.get("fill_numeric", "median")
        missing_before = int(self.df.isnull().sum().sum())
        details = []

        for col in self.df.columns:
            n_missing = int(self.df[col].isnull().sum())
            if n_missing == 0:
                continue

            pct = n_missing / len(self.df) * 100

            if self.df[col].dtype in [np.float64, np.int64, float, int]:
                fill_val = self.df[col].median() if strategy == "median" else self.df[col].mean()
                self.df[col] = self.df[col].fillna(round(fill_val, 4))
                details.append(f"'{col}': {n_missing} ({pct:.1f}%) filled with {strategy}={fill_val:.4f}")

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(method="ffill").fillna(method="bfill")
                details.append(f"'{col}': {n_missing} datetime nulls forward-filled")

            else:
                mode_val = (self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown")
                self.df[col] = self.df[col].fillna(mode_val)
                details.append(f"'{col}': {n_missing} ({pct:.1f}%) filled with mode='{mode_val}'")

        missing_after = int(self.df.isnull().sum().sum())
        self._record("missing_values",
                     f"Missing: {missing_before} → {missing_after}. " + " | ".join(details))
        return self

    # ── Step 4 ─────────────────────────────────────────────────
    def normalise_text(self) -> "DataCleaner":
        cols_cleaned = []
        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"[^\x20-\x7E]", "", regex=True)   # remove non-ASCII
            )
            cols_cleaned.append(col)

        self._record("normalise_text", f"Cleaned {len(cols_cleaned)} text columns: {cols_cleaned}")
        return self

    # ── Step 5 ─────────────────────────────────────────────────
    def handle_outliers(self) -> "DataCleaner":
        method     = self.config.get("outlier_method", "iqr")
        action     = self.config.get("outlier_action", "cap")
        iqr_mult   = float(self.config.get("iqr_multiplier", 1.5))
        z_thresh   = float(self.config.get("zscore_threshold", 3.0))

        numeric_cols  = self.df.select_dtypes(include=[np.number]).columns
        outlier_total = 0
        details       = []

        for col in numeric_cols:
            if method == "iqr":
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR    = Q3 - Q1
                lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
            else:
                mu, sigma = self.df[col].mean(), self.df[col].std()
                lower, upper = mu - z_thresh * sigma, mu + z_thresh * sigma

            mask  = (self.df[col] < lower) | (self.df[col] > upper)
            count = int(mask.sum())

            if count > 0:
                if action == "cap":
                    self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                else:
                    self.df = self.df[~mask]
                outlier_total += count
                details.append(f"'{col}': {count} outliers {action}ped → [{lower:.2f}, {upper:.2f}]")

        self._record("outliers",
                     f"Total outliers {action}ped: {outlier_total}. " + " | ".join(details) if details else "No outliers detected.")
        return self

    # ── Step 6 ─────────────────────────────────────────────────
    def add_derived_columns(self) -> "DataCleaner":
        added = []

        if {"quantity", "unit_price"}.issubset(self.df.columns):
            self.df["total_revenue"] = (self.df["quantity"] * self.df["unit_price"]).round(2)
            added.append("total_revenue = quantity × unit_price")

        if {"total_revenue", "quantity"}.issubset(self.df.columns):
            self.df["avg_unit_value"] = (self.df["total_revenue"] / self.df["quantity"].replace(0, np.nan)).round(2)
            added.append("avg_unit_value = total_revenue ÷ quantity")

        if "order_date" in self.df.columns:
            try:
                self.df["order_month"]    = self.df["order_date"].dt.to_period("M").astype(str)
                self.df["order_quarter"]  = self.df["order_date"].dt.to_period("Q").astype(str)
                self.df["order_dayofweek"]= self.df["order_date"].dt.day_name()
                added.append("order_month, order_quarter, order_dayofweek from order_date")
            except Exception as e:
                log.warning("Date feature extraction failed: %s", e)

        if "status" in self.df.columns:
            self.df["is_completed"] = (self.df["status"].str.lower() == "completed").astype(int)
            added.append("is_completed (binary flag)")

        self._record("feature_engineering", f"Added {len(added)} columns: {' | '.join(added)}")
        return self

    # ── Public API ─────────────────────────────────────────────
    def run_all(self) -> pd.DataFrame:
        log.info("Starting full cleaning pipeline...")
        (
            self.remove_duplicates()
                .fix_data_types()
                .handle_missing_values()
                .normalise_text()
                .handle_outliers()
                .add_derived_columns()
        )
        log.info("Cleaning complete. Final shape: %s", self.df.shape)
        return self.df

    def get_report(self) -> dict:
        return {
            "original_shape":    self.original.shape,
            "cleaned_shape":     self.df.shape,
            "rows_removed":      self.original.shape[0] - self.df.shape[0],
            "columns_added":     self.df.shape[1] - self.original.shape[1],
            "missing_before":    int(self.original.isnull().sum().sum()),
            "missing_after":     int(self.df.isnull().sum().sum()),
            "missing_reduction": f"{(1 - self.df.isnull().sum().sum() / max(self.original.isnull().sum().sum(), 1)) * 100:.1f}%",
            "steps":             self._log,
            "generated_at":      datetime.now().isoformat(),
        }
