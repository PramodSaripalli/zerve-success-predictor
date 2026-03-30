# Zerve User Success Predictor
### Zerve x HackerEarth Hackathon 2026

**Live API:** https://zerve-success-predictor-production.up.railway.app  
**API Docs:** https://zerve-success-predictor-production.up.railway.app/docs

---

## The Question

**Which user behaviors are most predictive of long-term success on Zerve?**

Success on a data platform isn't just signing up, it's when someone actually builds something, runs it, and comes back. To measure that, I defined a four-tier engagement ladder grounded in Zerve's actual platform mechanics. Then I built a full pipeline from raw event logs to a deployed prediction API that classifies any user and recommends exactly what to do to move them toward success.

---

## Pipeline

```
Raw Events (409,287 rows, 107 columns)
        ↓
  [Bronze]   Ingest, deduplicate on uuid
        ↓
  [Silver]   107 → 28 clean columns, behavioral flags added
        ↓
  [Gold]     One row per user, 53 features, 4-tier success labels
        ↓
  [Model]    XGBoost, 96% accuracy, SHAP explanations per prediction
        ↓
  [Cohort]   Time-based analysis — week-1 behavior vs long-term outcome
        ↓
  [Cluster]  K-means validation of tier definitions
        ↓
  [API]      Live FastAPI — predictions + risk level + intervention engine
```

---

## Defining Success

Success isn't binary. I built a four-tier ladder based on what Zerve's platform actually measures:

| Tier | Label | Definition |
|------|-------|-----------|
| 0 | **Visitor** | Signed in, no meaningful action |
| 1 | **Explorer** | Opened a canvas or started an AI chat |
| 2 | **Engaged** | Regular credit usage or agent tool activity — mostly Python SDK users |
| 3 | **Power User** | 10+ credit events + 10+ agent tool calls + 3+ sessions |

**Distribution across 4,774 users:**

| Tier | Count | % |
|------|-------|---|
| Visitor | 3,437 | 72% |
| Engaged | 591 | 12.4% |
| Explorer | 450 | 9.4% |
| Power User | 296 | 6.2% |

---

## Key Findings

### 1. Starting an AI chat is a 6.8x multiplier
Users who start an AI chat in week one are **6.8x more likely** to become Power Users — 21% conversion vs 3.1% for users who never chat. This is the single most actionable early signal in the entire dataset.

### 2. The Engaged tier is the Python SDK cohort
`used_python_sdk` is the top SHAP predictor for Engaged users. These users don't appear active in browser logs — they run jobs programmatically via the SDK. They look like Visitors in the UI but are genuinely high-value developers.

**Product implication:** SDK users need a separate retention strategy. UI nudges won't reach them.

### 3. Power Users are defined by breadth, not volume
`total_events` and `unique_event_types` dominate Power User predictions. Power Users use 20 unique event types in week one versus 1 for Visitors. They don't go deep on one feature — they explore the whole platform.

### 4. Session depth beats session count
`tool_calls_per_chat` and `session_span_days` outrank raw visit frequency. One deep session predicts success better than five shallow ones.

### 5. Onboarding form completion is a qualifying signal
Users who fill out the onboarding form show meaningfully higher downstream engagement. The form self-selects motivated users — it's a filter, not friction.

### 6. October cohort had 40.3% success rate
September cohort: 22.3%. October: 40.3%. Nearly double. Something changed in the product that month that drove better activation.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.0% |
| 5-Fold CV Accuracy | 94.8% ± 0.5% |
| Macro OvR AUC | 0.994 |
| Power User F1 | 0.82 |
| Engaged F1 | 0.83 |
| Majority-class baseline | 72.0% |
| Improvement over baseline | **+22.8 percentage points** |

18 tier-defining features were excluded from model input to prevent leakage. The model learns from behavioral signals — chat initiation, onboarding behavior, session depth, SDK usage — not from the same variables used to define the tiers.

---

## Clustering Validation

K-means clustering was run independently to validate whether the rule-based tiers reflect real behavioral groups.

- **Adjusted Rand Index: 0.331** — moderate agreement
- **Power User cluster: 100% pure** — k-means found the same users independently
- **Engaged cluster: 75% pure** — broadly validated
- **Optimal k by silhouette: 2** — the data most naturally splits active vs inactive

The four-tier framework is a product segmentation decision layered on top of a fundamentally binary behavioral split. Power Users are real. Explorer and Engaged are useful but not strongly distinct natural clusters.

---

## Cohort Analysis

| Signup Month | Users | Power User % | Success % |
|-------------|-------|-------------|-----------|
| Sep 2025 | 999 | 4.2% | 22.3% |
| Oct 2025 | 472 | 23.3% | 40.3% |
| Nov 2025 | 2,000 | 5.8% | 17.6% |
| Dec 2025 | 1,300 | 2.2% | 9.4% |

---

## The API

**Base URL:** `https://zerve-success-predictor-production.up.railway.app`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Killer stat + version |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata and performance |
| POST | `/predict` | Predict tier for one user |
| POST | `/predict/batch` | Predict for up to 500 users |
| GET | `/predict/example` | Sample Power User prediction |

Every prediction returns tier, risk level, confidence, top 3 SHAP drivers, and a recommended intervention grounded in the analysis.

**Sample response:**
```json
{
  "predicted_tier": 3,
  "predicted_label": "Power User",
  "risk_level": "None",
  "confidence": 0.9565,
  "top_behavioral_drivers": [
    {"feature": "total_events", "shap_impact": 0.88, "direction": "increases"},
    {"feature": "tool_calls_per_chat", "shap_impact": 0.40, "direction": "increases"}
  ],
  "recommended_intervention": {
    "action": "Retain",
    "message": "No action needed. Monitor for credit exhaustion.",
    "priority": "Low"
  }
}
```

---
## Recommendations for Zerve

Based on the analysis, here are four product interventions grounded directly in the findings:

**1. Trigger the first AI chat as early as possible**
Users who start an AI chat in week one are 6.8x more likely to become Power Users. The onboarding flow should make starting a chat the first action, not an optional step.

**2. Build a separate retention track for SDK users**
The Engaged tier is almost entirely Python SDK developers. They look inactive in the UI but are high-value. UI-based nudges won't reach them — they need API-level communication like usage emails or SDK-specific docs.

**3. Optimize for feature breadth, not depth**
Power Users use 20 unique event types in week one versus 1 for Visitors. Introducing users to multiple features early — not just one workflow — is what drives Power User growth.

**4. Don't remove the onboarding form**
Users who submit the onboarding form show higher downstream engagement. It self-selects motivated users. It's a qualifying filter, not friction.

---
