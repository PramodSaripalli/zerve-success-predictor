# Zerve User Success Predictor
### Zerve x HackerEarth Hackathon Submission

---

## The Question

**Which user behaviors are most predictive of long-term success on Zerve?**

This project answers that question end-to-end: from raw event logs to a deployed prediction API that classifies any user's success trajectory and explains *why* — in real time.

---

## What I Built

A full medallion data pipeline + ML model + REST API, built entirely on the Zerve platform.

```
Raw Events (409,287 rows)
        ↓
  [Bronze Layer]  — Ingest, deduplicate, audit
        ↓
  [Silver Layer]  — Clean, type, extract behavioral signals
        ↓
  [Gold Layer]    — One row per user, 59 features, success tier labels
        ↓
  [XGBoost Model] — 95% accuracy, 0.994 AUC, SHAP explanations
        ↓
  [FastAPI Endpoint] — Live predictions with behavioral drivers
```

---

## Defining "Success"

Success isn't binary. I defined a **4-tier ladder** grounded in Zerve's actual platform mechanics:

| Tier | Label | Definition |
|------|-------|-----------|
| 0 | **Visitor** | Signed in, minimal interaction. High churn risk. |
| 1 | **Explorer** | Opened canvases, started an AI chat. Curious but not committed. |
| 2 | **Engaged** | Regular credit usage or sustained agent tool usage. Often Python SDK users. |
| 3 | **Power User** | High credit consumption + multi-session + deep agent tool breadth. Core retained user. |

**Distribution across 4,774 users:**
- Visitor: 72% (3,436)
- Engaged: 12.4% (590)
- Explorer: 9.4% (450)
- Power User: 6.2% (298)

---

## Key Findings

### 1. Starting an AI chat is the single most intentional early signal
`agent_chats_started` is the dominant predictor for the Explorer tier — with a SHAP value of 2.2, far above any other feature. Users who initiate even one AI chat are fundamentally different from those who don't.

**Product implication:** Zerve should optimize the first-session experience to drive users to their first AI chat as quickly as possible. This is the activation moment.

### 2. The Engaged tier is the Python SDK cohort
`used_python_sdk` dominates Engaged tier predictions. These users don't appear active in browser logs — they're running agent jobs programmatically. They look like Visitors in the UI but are actually high-value developers.

**Product implication:** SDK users need a separate retention strategy. They won't respond to UI-based nudges.

### 3. Power Users are defined by breadth, not just volume
`total_events` and `unique_event_types` are the top Power User predictors. Power Users don't just do one thing a lot — they use more of the platform in more ways across more sessions.

**Product implication:** Expanding feature discovery (not just depth of one feature) drives users toward Power User status.

### 4. Onboarding form completion is a qualifying signal
`submitted_onboarding_form` appears in the top 10 SHAP features. Users who fill it out show meaningfully higher engagement downstream. This means the form isn't just friction — it's self-selection by motivated users.

**Product implication:** Don't remove the onboarding form. Optimize it to attract higher-intent signups.

### 5. Session depth beats session count
`tool_calls_per_chat` and `session_span_days` rank above raw frequency metrics. How deeply a user engages per session predicts success better than how often they return.

---

## Limitations & Honest Caveats

### Success tiers are analyst-defined, not ground truth
The 4 tiers were defined using rule-based thresholds on credit usage, agent tool calls, and session count. These thresholds were calibrated to produce a meaningful distribution, not validated against Zerve's internal retention or revenue data. A user labeled "Power User" here is a heavy platform user — but whether they are a paid subscriber or a long-term retained user is unknown without access to billing data.

**Validation proxy used:** Power Users show a median `session_span_days` of 0.925 vs 0.0 for Visitors, and 5 unique sessions vs 1 — consistent with genuine long-term engagement.

### The thresholds could be replaced with clustering
An unsupervised approach (k-means or DBSCAN) would let the data define natural groupings without analyst-imposed boundaries. This is a logical next step to validate or challenge the current tier definitions.

### Data covers 3.5 months only
The dataset spans September–December 2025. Seasonal patterns, product launches, or onboarding changes during this period could affect the generalizability of findings.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 95.0% |
| 5-Fold CV Accuracy | 94.7% ± 0.6% |
| Macro OvR AUC | 0.994 |
| Power User F1 | 0.84 |
| Engaged F1 | 0.83 |
| **Majority-class baseline** | **72.0%** |
| **Improvement over baseline** | **+23 percentage points** |

A naive classifier that always predicts "Visitor" (the majority class at 72%) would achieve 72% accuracy without learning anything. Our model at 95% represents a meaningful 23-point improvement, with strong F1 scores on the hardest minority classes (Power User and Engaged) — the ones Zerve actually cares about identifying.

The model is trained on behavioral features only — no tier-defining features (credit counts, agent tool call totals, session counts) are used as inputs. The model learns from signals like onboarding behavior, canvas interactions, chat initiation, and session depth, not from the same variables used to define success.

---

## The API

A live REST endpoint that classifies any user and explains the prediction.

**Base URL:** `http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata, features, performance |
| POST | `/predict` | Predict success tier for one user |
| POST | `/predict/batch` | Predict for up to 500 users |
| GET | `/predict/example` | Sample Power User prediction |

### Sample Request
```bash
POST /predict
{
  "total_events": 150,
  "agent_chats_started": 8,
  "unique_canvases": 4,
  "session_span_days": 4.5,
  "used_web_app": 1,
  "tool_calls_per_chat": 6.5
}
```

### Sample Response
```json
{
  "predicted_tier": 3,
  "predicted_label": "Power User",
  "confidence": 0.979,
  "tier_probabilities": {
    "Visitor": 0.0016,
    "Explorer": 0.0061,
    "Engaged": 0.0133,
    "Power User": 0.979
  },
  "top_behavioral_drivers": [
    {
      "feature": "total_events",
      "value": 150.0,
      "shap_impact": 0.9912,
      "direction": "increases"
    },
    {
      "feature": "tool_calls_per_chat",
      "value": 6.5,
      "shap_impact": 0.4157,
      "direction": "increases"
    }
  ],
  "predicted_at": "2026-03-20T00:30:49Z"
}
```

The response includes not just the predicted tier but **why** — the top 3 behavioral drivers with their SHAP impact values and direction.

---

## Pipeline Architecture

### Bronze Layer (`01_bronze.py`)
- Reads raw CSV (409,287 events, 107 columns)
- Deduplicates on `uuid` (event ID)
- Adds audit columns: `_bronze_loaded_at`, `_source_file`, `_row_hash`
- Outputs: `bronze_events.parquet`

### Silver Layer (`02_silver.py`)
- Reduces 107 columns to 35 meaningful ones
- Renames and types all columns
- Extracts `canvas_id` from URL paths
- Adds event category flags: `is_engagement_event`, `is_ai_event`, `is_execution_event`
- Identifies SDK source: web browser vs Python backend
- Outputs: `silver_events.parquet`

### Gold Layer (`03_gold_v2.py`)
- Aggregates to one row per user
- Engineers 59 behavioral features across 8 categories:
  - Volume (total events, unique types, active days)
  - Credit usage (primary engagement signal)
  - Agent tool call depth (8 tool types tracked individually)
  - Agent session behavior (chats, messages, prompts)
  - Manual execution (run_block, run_all_blocks, run_upto_block)
  - Canvas/workflow complexity (creates, opens, edges, files)
  - Onboarding behavior (submitted, skipped, completed tour)
  - Temporal (session span, first/last seen)
- Assigns 4-tier success labels
- Outputs: `gold_users_v2.parquet`

### Model Layer (`04_model.py`)
- XGBoost multiclass classifier (4 classes)
- Class weighting to handle 72% Visitor imbalance
- Early stopping on log-loss
- SHAP TreeExplainer for per-prediction explanations
- Outputs: `zerve_xgb_model.json`, `shap_by_tier.png`, `shap_overall.png`

### API Layer (`05_api.py`)
- FastAPI with auto-generated OpenAPI docs
- Per-prediction SHAP explanations
- Batch endpoint (up to 500 users)
- `/model/info` endpoint documents training data and performance

---

## Data

- **Source:** Zerve platform event logs (PostHog)
- **Period:** September 1 – December 8, 2025
- **Events:** 409,287
- **Users:** 4,774
- **Event types:** 141

---

## How to Run

```bash
# Install dependencies
pip install pandas pyarrow xgboost shap scikit-learn fastapi uvicorn

# Run pipeline
python 01_bronze.py
python 02_silver.py
python 03_gold_v2.py
python 04_model.py

# Start API
python 05_api.py

# API docs
open http://localhost:8000/docs
```

---

## Files

```
01_bronze.py          — Bronze ingestion layer
02_silver.py          — Silver cleaning layer
03_gold_v2.py         — Gold feature engineering layer
04_model.py           — XGBoost model + SHAP analysis
05_api.py             — FastAPI prediction endpoint
zerve_xgb_model.json  — Trained model weights
confusion_matrix.png  — Model evaluation
shap_by_tier.png      — SHAP feature importance per tier
shap_overall.png      — Top 20 behavioral predictors overall
```
