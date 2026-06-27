"""
main.py
────────────────────────────────────────────────────────────────
AI Automation in Data Analysis — Master Pipeline
Author : Meghraj Nikalje | B.Sc. Data Science, Mumbai University
────────────────────────────────────────────────────────────────

USAGE:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="your-key-here"
    python main.py
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys
import json
import time
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Local modules ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.logger       import get_logger
from src.data_cleaner import DataCleaner
from src.eda          import EDAEngine
from src.visualiser   import Visualiser
from src.ai_analyzer  import AIAnalyzer

# ── Bootstrap logger before config is loaded ──────────────────────
log = get_logger("main")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def load_config(path: str = "config/config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("config.yaml not found — using defaults.")
        return {}


def banner(title: str) -> None:
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def step(n: int, title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  STEP {n} — {title}")
    print(f"{'─' * 62}")


def save_text(content: str, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Saved → %s", path)


# ══════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════
def step1_load(cfg: dict) -> pd.DataFrame:
    step(1, "Load Raw Data")
    filepath = cfg.get("data", {}).get("input_file", "sample_data.csv")

    if not Path(filepath).exists():
        log.error("Input file not found: %s", filepath)
        sys.exit(1)

    df = pd.read_csv(filepath, encoding=cfg.get("data", {}).get("encoding", "utf-8"))
    log.info("Loaded '%s' — %d rows × %d columns", filepath, *df.shape)
    print(f"  ✔ File: {filepath}")
    print(f"  ✔ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  ✔ Columns: {list(df.columns)}")
    print(f"\n  Sample (3 rows):\n{df.head(3).to_string(index=False)}")
    return df


def step2_clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    step(2, "Automated Data Cleaning")
    t0      = time.time()
    cleaner = DataCleaner(df, config=cfg.get("cleaning", {}))
    clean   = cleaner.run_all()
    report  = cleaner.get_report()
    elapsed = time.time() - t0

    print(f"  ✔ Shape: {report['original_shape']} → {report['cleaned_shape']}")
    print(f"  ✔ Missing values: {report['missing_before']} → {report['missing_after']}")
    print(f"  ✔ Columns added: {report['columns_added']}")
    print(f"  ✔ Completed in {elapsed:.2f}s")

    out = cfg.get("data", {}).get("output_dir", "report")
    clean.to_csv(f"{out}/cleaned_data.csv", index=False)
    save_text(json.dumps(report, indent=2, default=str), f"{out}/cleaning_report.json")
    return clean, report


def step3_eda(df: pd.DataFrame, cfg: dict) -> dict:
    step(3, "Exploratory Data Analysis")
    t0  = time.time()
    eda = EDAEngine(df, config=cfg.get("eda", {}))
    stats = eda.run_all()

    out = cfg.get("data", {}).get("output_dir", "report")
    eda.save(f"{out}/summary_stats.json")

    rev = stats.get("revenue", {})
    if rev:
        print(f"  ✔ Total Revenue:    ₹{rev.get('total', 0):>12,.2f}")
        print(f"  ✔ Average Order:    ₹{rev.get('mean', 0):>12,.2f}")
        print(f"  ✔ Median Order:     ₹{rev.get('median', 0):>12,.2f}")

    status = stats.get("order_status", {}).get("counts", {})
    if status:
        print(f"\n  Order Status Breakdown:")
        for s, c in status.items():
            pct = stats["order_status"]["pct"].get(s, 0)
            print(f"    {s:<15} {c:>4} orders ({pct}%)")

    print(f"\n  ✔ EDA completed in {time.time() - t0:.2f}s")
    return stats


def step4_visualise(df: pd.DataFrame, stats: dict, cfg: dict) -> str:
    step(4, "Generate Visualisations")
    t0  = time.time()
    out = cfg.get("data", {}).get("output_dir", "report")
    vis = Visualiser(df, stats, config=cfg.get("visualisation", {}))

    try:
        path = vis.build_dashboard(output_dir=out)
        print(f"  ✔ Dashboard saved → {path}")
        print(f"  ✔ Completed in {time.time() - t0:.2f}s")
        return path
    except Exception as e:
        log.warning("Visualisation failed (scipy may not be installed): %s", e)
        print(f"  ⚠ Visualisation skipped — install scipy: pip install scipy")
        return ""


def step5_ai_insights(df: pd.DataFrame, stats: dict, cleaning_report: dict, cfg: dict) -> None:
    step(5, "AI-Generated Business Insights (Claude API)")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ⚠  ANTHROPIC_API_KEY not set — skipping AI step.")
        print("     Set it: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("     Get key: https://console.anthropic.com")
        return

    t0 = time.time()
    try:
        ai      = AIAnalyzer(api_key=api_key, config=cfg.get("ai", {}))
        report  = ai.generate_executive_summary(df, stats, cleaning_report)
        out     = cfg.get("data", {}).get("output_dir", "report")
        save_text(report, f"{out}/ai_report.md")
        print(f"  ✔ AI report saved → {out}/ai_report.md")
        print(f"  ✔ Completed in {time.time() - t0:.2f}s")
        print(f"\n{'─' * 62}")
        print("  INSIGHTS PREVIEW (first 800 chars):")
        print(f"{'─' * 62}")
        print(report[:800] + ("..." if len(report) > 800 else ""))
    except Exception as e:
        log.error("AI step failed: %s", e)
        print(f"  ✗ AI step failed: {e}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    start = time.time()
    banner("AI AUTOMATION IN DATA ANALYSIS PIPELINE v2.0")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Author  : Meghraj Nikalje | B.Sc. Data Science, Mumbai University")

    cfg = load_config()

    df              = step1_load(cfg)
    clean_df, c_rep = step2_clean(df, cfg)
    stats           = step3_eda(clean_df, cfg)
    _               = step4_visualise(clean_df, stats, cfg)
    step5_ai_insights(clean_df, stats, c_rep, cfg)

    out = cfg.get("data", {}).get("output_dir", "report")
    banner("PIPELINE COMPLETE ✅")
    print(f"  Total time : {time.time() - start:.2f}s")
    print(f"  Output dir : ./{out}/")
    print(f"\n  Generated files:")
    for f in sorted(Path(out).iterdir()):
        size = f.stat().st_size
        print(f"    📄 {f.name:<35} ({size:,} bytes)")
    print()


if __name__ == "__main__":
    main()
