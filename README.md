# 🤖 AI Automation in Data Analysis Pipeline v2.0

> **A production-grade, fully modular data analysis pipeline** — automated cleaning, statistical EDA, multi-panel visualisation, AI-powered business insight generation, unit testing, YAML config, and structured logging — all in a single `python main.py`.

---

## 🏗️ Architecture

```
sample_data.csv
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1  main.py — load_config() + load_data()          │
│  • Reads config/config.yaml (model, paths, thresholds)  │
│  • Validates file exists, logs shape and columns         │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2  src/data_cleaner.py — DataCleaner.run_all()    │
│  1. remove_duplicates   exact row deduplication         │
│  2. fix_data_types      parse dates, coerce numerics    │
│  3. handle_missing      median/mode imputation          │
│  4. normalise_text      strip, collapse whitespace      │
│  5. handle_outliers     IQR cap (configurable)          │
│  6. add_derived         total_revenue, order_month etc. │
│  → report/cleaned_data.csv + cleaning_report.json       │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3  src/eda.py — EDAEngine.run_all()               │
│  • Descriptive stats (all columns)                      │
│  • Revenue KPIs (total, mean, median, p25/p75/p90)      │
│  • Breakdowns: category, region, monthly, sales rep     │
│  • Order status analysis                                │
│  • Correlation matrix + high-corr pair detection        │
│  → report/summary_stats.json                            │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4  src/visualiser.py — Visualiser.build_dashboard()│
│  Panel 1: Revenue by Category (bar + annotations)       │
│  Panel 2: Order Status (pie with % labels)              │
│  Panel 3: Monthly Revenue Trend (line + fill + peak)    │
│  Panel 4: Sales Rep Performance (horizontal bar)        │
│  Panel 5: Order Value Distribution (histogram + KDE)    │
│  Panel 6: Region × Category Heatmap (seaborn)           │
│  → report/dashboard.png                                 │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5  src/ai_analyzer.py — AIAnalyzer.generate()     │
│  • Builds compact JSON payload (never sends raw rows)   │
│  • Structured system + user prompt engineering          │
│  • Claude API → Executive Summary + Recommendations     │
│  • Separate data quality narrative                      │
│  → report/ai_report.md                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
03-ai-data-analysis/
│
├── main.py                     ← Master pipeline (run this)
├── sample_data.csv             ← 30-row sales dataset
├── requirements.txt            ← All dependencies
│
├── config/
│   └── config.yaml             ← All settings (model, paths, thresholds)
│
├── src/
│   ├── __init__.py
│   ├── logger.py               ← Centralised logging (console + file)
│   ├── data_cleaner.py         ← 6-step automated cleaning pipeline
│   ├── eda.py                  ← Statistical analysis & KPI computation
│   ├── visualiser.py           ← 6-panel matplotlib/seaborn dashboard
│   └── ai_analyzer.py         ← Claude API integration & prompt engine
│
├── tests/
│   └── test_data_cleaner.py   ← 15 pytest unit tests
│
└── report/                     ← Auto-generated outputs
    ├── cleaned_data.csv
    ├── cleaning_report.json
    ├── summary_stats.json
    ├── dashboard.png
    ├── ai_report.md
    └── pipeline.log
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS/Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows
```
> Get a free key at [console.anthropic.com](https://console.anthropic.com)

### 3. Run the pipeline
```bash
python main.py
```

### 4. Run unit tests
```bash
python -m pytest tests/ -v
```

---

## ⚙️ Configuration (config/config.yaml)

All settings are controlled from a single YAML file — no hardcoded values:

| Setting | Default | Description |
|---------|---------|-------------|
| `cleaning.outlier_method` | `iqr` | `iqr` or `zscore` |
| `cleaning.outlier_action` | `cap` | `cap` (clip) or `remove` |
| `cleaning.fill_numeric` | `median` | `median` or `mean` |
| `ai.model` | `claude-sonnet-4-6` | Anthropic model to use |
| `ai.max_tokens` | `2000` | Response length limit |
| `eda.correlation_threshold` | `0.7` | Flag pairs above this |
| `logging.level` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` |

---

## 🧹 Cleaning Steps (DataCleaner)

| # | Step | What it does |
|---|------|-------------|
| 1 | remove_duplicates | Drops exact duplicate rows; resets index |
| 2 | fix_data_types | Auto-parses dates; converts numeric strings (≥90% confidence) |
| 3 | handle_missing | Numeric → median fill; Datetime → forward fill; Text → mode fill |
| 4 | normalise_text | Strips whitespace, collapses multiple spaces, removes non-ASCII |
| 5 | handle_outliers | IQR (×1.5) or Z-score (±3σ) — cap or remove configurable |
| 6 | add_derived | `total_revenue`, `avg_unit_value`, `order_month`, `order_quarter`, `is_completed` |

---

## 📊 Dashboard Panels

| Panel | Chart Type | What it shows |
|-------|-----------|---------------|
| 1 | Bar + Annotations | Revenue totals per product category |
| 2 | Pie | Completed / Pending / Cancelled split |
| 3 | Line + Fill + Peak label | Monthly revenue trend |
| 4 | Horizontal Bar | Top sales reps ranked by revenue |
| 5 | Histogram + KDE + Mean/Median | Order value distribution shape |
| 6 | Heatmap | Revenue by Region × Category matrix |

---

## 🤖 AI Features (ai_analyzer.py)

- **Privacy-safe** — sends only aggregated statistics, never raw data rows
- **Structured prompts** — system role + explicit output format instructions
- **Outputs:**
  - Executive Summary
  - Top performers by category/region
  - Trend and seasonality analysis
  - Risks and concerns (cancellations, anomalies)
  - 3 specific, numbered recommendations
  - KPI snapshot table
  - Data quality narrative (plain English, non-technical audience)

---

## 🧪 Unit Tests

15 tests covering `DataCleaner`:
- Duplicate removal (with and without duplicates)
- Date type conversion
- Null filling (numeric and categorical)
- Outlier capping
- Derived column creation (total_revenue, order_month, is_completed)
- Cleaning report structure validation
- Original DataFrame immutability

```bash
python -m pytest tests/ -v --tb=short
```

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+ | Core language |
| Pandas 2.x | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib + Seaborn | Visualisation |
| SciPy | KDE for distribution plots |
| Requests | Claude API HTTP client |
| PyYAML | Configuration management |
| pytest | Unit testing |
| Claude API | AI insight generation |

---

## 💡 How to Use Your Own Data

1. Replace `sample_data.csv` with your dataset
2. Update `config/config.yaml` → `data.input_file`
3. Run `python main.py` — the pipeline auto-adapts

Works with any CSV that has numeric and/or categorical columns.

---

*Built by Meghraj Nikalje | B.Sc. Data Science, Mumbai University*
