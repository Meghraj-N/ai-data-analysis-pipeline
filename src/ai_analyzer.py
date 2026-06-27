"""
src/ai_analyzer.py
────────────────────────────────────────────────────────────────
AI Insight Engine — Claude API Integration
Author : Meghraj Nikalje | B.Sc. Data Science, Mumbai University
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from src.logger import get_logger

log = get_logger(__name__)


class AIAnalyzer:
    """
    Sends compact dataset summaries to the Claude API and
    receives structured business insight reports.

    Design principles:
    - Never sends raw rows — only aggregated statistics (privacy & efficiency)
    - Structured prompts with explicit output format instructions
    - Graceful fallback if API is unavailable
    """

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: Optional[str] = None, config: Optional[dict] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.config  = config or {}
        self.model   = self.config.get("model", "claude-sonnet-4-6")
        self.max_tokens = int(self.config.get("max_tokens", 2000))
        self.timeout = int(self.config.get("timeout_seconds", 45))

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set.\n"
                "  Set it with:  export ANTHROPIC_API_KEY='your-key-here'\n"
                "  Get a key at: https://console.anthropic.com"
            )
        log.info("AIAnalyzer initialised — model: %s", self.model)

    # ── Private: call API ──────────────────────────────────────
    def _call(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "system":     system_prompt,
            "messages":   [{"role": "user", "content": user_prompt}],
        }
        log.debug("Calling Claude API — model: %s, max_tokens: %d", self.model, self.max_tokens)

        try:
            resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()["content"][0]["text"]
            log.info("Claude API response received — %d chars", len(result))
            return result
        except requests.exceptions.Timeout:
            log.error("Claude API timed out after %ds", self.timeout)
            raise
        except requests.exceptions.HTTPError as e:
            log.error("Claude API HTTP error: %s", e)
            raise
        except Exception as e:
            log.error("Claude API unexpected error: %s", e)
            raise

    # ── Private: build compact payload ─────────────────────────
    def _build_stats_payload(self, df: pd.DataFrame, stats: dict) -> dict:
        max_rows = int(self.config.get("max_rows_in_prompt", 200))

        payload: dict = {
            "dataset_info": {
                "rows": min(df.shape[0], max_rows),
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }
        }

        # Revenue metrics
        if "revenue" in stats:
            payload["revenue_kpis"] = stats["revenue"]

        # Category breakdown (top 10)
        if "revenue_by_category" in stats:
            top_cats = dict(
                sorted(stats["revenue_by_category"].get("total", {}).items(),
                       key=lambda x: x[1], reverse=True)[:10]
            )
            payload["revenue_by_category"] = top_cats

        if "revenue_by_region" in stats:
            payload["revenue_by_region"] = {
                k: round(float(v), 2)
                for k, v in stats["revenue_by_region"].get("total", {}).items()
            }

        if "monthly_trend" in stats:
            payload["monthly_revenue"] = stats["monthly_trend"].get("revenue", {})

        if "order_status" in stats:
            payload["order_status_counts"] = stats["order_status"].get("counts", {})
            payload["order_status_pct"]    = stats["order_status"].get("pct", {})

        if "sales_rep_performance" in stats:
            top_reps = dict(
                sorted(stats["sales_rep_performance"].get("total", {}).items(),
                       key=lambda x: x[1], reverse=True)[:5]
            )
            payload["top_sales_reps"] = top_reps

        if "correlations" in stats:
            payload["high_corr_pairs"] = stats["correlations"].get("high_corr_pairs", [])

        return payload

    # ── Public: generate business insights ────────────────────
    def generate_insights(self, df: pd.DataFrame, stats: dict) -> str:
        log.info("Generating AI business insights...")
        payload = self._build_stats_payload(df, stats)

        system = (
            "You are a senior data analyst writing a professional business intelligence report. "
            "Be concise, specific, and insight-driven. Use exact numbers from the data. "
            "Use markdown formatting (bold, bullets, headers). Avoid generic filler statements."
        )

        user = f"""
Analyse the following sales dataset summary and write a professional business insights report.

**Dataset Statistics (JSON):**
```json
{json.dumps(payload, indent=2, default=str)}
```

**Required Report Structure (use these exact headings):**

## 📋 Executive Summary
2–3 sentence overview with the single most important finding.

## 🏆 Top Performers
Which categories, regions, or products are driving the most revenue? Include % share.

## 📈 Trends & Patterns
Monthly or temporal patterns. Is revenue growing or declining? Any seasonality?

## ⚠️ Risks & Concerns
Order cancellations, pending revenue, low-performing segments, any anomalies.

## 💡 Recommendations
Exactly 3 numbered, specific, actionable recommendations for the business team.

## 📊 Key Metrics Snapshot
Quick-reference table of 5 essential KPIs.

Keep the full report under 600 words. Be specific — use the actual numbers from the data.
""".strip()

        return self._call(system, user)

    # ── Public: data quality narrative ────────────────────────
    def summarise_cleaning(self, cleaning_report: dict) -> str:
        log.info("Generating cleaning narrative...")

        system = (
            "You are a data quality analyst writing a brief, professional data quality summary "
            "for a non-technical business audience. Be clear and avoid jargon."
        )

        user = f"""
Below is an automated data cleaning log. Write a 3–4 sentence paragraph summarising what data quality issues were found and corrected.

**Cleaning Report:**
```json
{json.dumps(cleaning_report, indent=2, default=str)}
```

Format: Plain paragraph, professional tone. Mention: original shape, issues found, how they were fixed, final shape.
""".strip()

        return self._call(system, user)

    # ── Public: generate executive PDF summary ─────────────────
    def generate_executive_summary(self, df: pd.DataFrame, stats: dict,
                                   cleaning_report: dict) -> str:
        log.info("Generating full executive summary...")
        insights  = self.generate_insights(df, stats)
        dq_note   = self.summarise_cleaning(cleaning_report)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""# AI AUTOMATION IN DATA ANALYSIS — EXECUTIVE REPORT

**Generated:** {timestamp}
**Author:** Meghraj Nikalje | B.Sc. Data Science, Mumbai University
**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns

---

## Data Quality Summary

{dq_note}

---

{insights}

---

*Report auto-generated by AI Data Analysis Pipeline v2.0*
"""
