"""
tests/test_data_cleaner.py
Unit tests for the DataCleaner module.
Run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.data_cleaner import DataCleaner


# ── Fixtures ───────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "order_id":    [1, 2, 2, 3, 4],       # row 2 is a duplicate
        "product":     ["Laptop", "Chair", "Chair", "Phone", "Monitor"],
        "category":    ["Electronics", "Furniture", "Furniture", None, "Electronics"],
        "quantity":    [1, 3, 3, None, 2],
        "unit_price":  [55000, 8500, 8500, 32000, 18000],
        "order_date":  ["2024-01-05", "2024-01-07", "2024-01-07", "2024-01-08", "2024-01-10"],
        "status":      ["Completed", "Completed", "Completed", "Pending", "Completed"],
    })


# ── Tests ──────────────────────────────────────────────────────────
class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.remove_duplicates()
        assert len(cleaner.df) == 4, "Should remove 1 duplicate row"

    def test_no_duplicate_no_change(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        cleaner = DataCleaner(df)
        cleaner.remove_duplicates()
        assert len(cleaner.df) == 3


class TestFixDataTypes:
    def test_parses_date_column(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.remove_duplicates().fix_data_types()
        assert pd.api.types.is_datetime64_any_dtype(cleaner.df["order_date"])

    def test_original_unchanged(self, sample_df):
        cleaner = DataCleaner(sample_df)
        original_dtypes = sample_df.dtypes.to_dict()
        cleaner.run_all()
        # Original DataFrame must not be mutated
        assert sample_df.dtypes.to_dict() == original_dtypes


class TestHandleMissingValues:
    def test_fills_numeric_nulls(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.remove_duplicates().fix_data_types().handle_missing_values()
        assert cleaner.df["quantity"].isnull().sum() == 0

    def test_fills_categorical_nulls(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.remove_duplicates().fix_data_types().handle_missing_values()
        assert cleaner.df["category"].isnull().sum() == 0

    def test_zero_missing_after_clean(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        assert cleaner.df.isnull().sum().sum() == 0


class TestHandleOutliers:
    def test_caps_extreme_values(self):
        df = pd.DataFrame({"price": [100, 110, 105, 99, 10000]})  # 10000 is outlier
        cleaner = DataCleaner(df)
        cleaner.handle_outliers()
        assert cleaner.df["price"].max() < 10000, "Outlier should be capped"

    def test_no_nulls_introduced(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        assert cleaner.df.isnull().sum().sum() == 0


class TestAddDerivedColumns:
    def test_total_revenue_added(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        assert "total_revenue" in cleaner.df.columns

    def test_order_month_added(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        assert "order_month" in cleaner.df.columns

    def test_is_completed_binary(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        assert "is_completed" in cleaner.df.columns
        assert set(cleaner.df["is_completed"].unique()).issubset({0, 1})


class TestGetReport:
    def test_report_shape(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        report = cleaner.get_report()
        assert "original_shape" in report
        assert "cleaned_shape"  in report
        assert "steps"          in report
        assert isinstance(report["steps"], list)

    def test_report_missing_reduction(self, sample_df):
        cleaner = DataCleaner(sample_df)
        cleaner.run_all()
        report = cleaner.get_report()
        assert report["missing_after"] <= report["missing_before"]
