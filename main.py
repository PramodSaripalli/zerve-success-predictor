import os
import numpy as np
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier
from datetime import datetime

MODEL_PATH  = "zerve_xgb_model.json"
TIER_LABELS = {0:"Visitor", 1:"Explorer", 2:"Engaged", 3:"Power User"}
TIER_DESC   = {
    "Visitor":    "Signed in but minimal interaction. High churn risk.",
    "Explorer":   "Started at least one AI chat. Curious but not committed.",
    "Engaged":    "Regular usage via Python SDK or agent tools. Likely a developer.",
    "Power User": "High event volume across diverse behaviors. Core retained user.",
}
RISK_LEVEL = {"Visitor":"High","Explorer":"Medium","Engaged":"Low","Power User":"None"}
FEATURES = [
    "total_events","unique_event_types","unique_days_active","unique_canvases",
    "agent_chats_started","agent_messages_sent","agent_started_from_prompt",
    "agent_suprise_me","suggestions_accepted","errors_investigated",
    "run_block_manual","run_all_blocks","run_upto_block","total_manual_executions",
    "canvases_created","canvases_opened","blocks_created_manual","blocks_deleted",
    "blocks_renamed","blocks_copied","edges_created","files_uploaded","files_downloaded",
    "signed_up","new_user_created","submitted_onboarding_form","skipped_onboarding",
    "completed_onboarding_tour","fullscreen_opens","link_clicks",
    "session_span_days","used_python_sdk","used_web_app",
    "tool_calls_per_chat","execution_per_chat",
]

print("[API] Loading model...")
model     = XGBClassifier()
model.load_model(MODEL_PATH)
explainer = shap.TreeExplainer(model)
print("[API] Ready.")

app = FastAPI(
    title="Zerve User Success Predictor",
    description="Predicts user success tier and recommends product interventions. Zerve x HackerEarth 2026.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def recommend_action(features: dict, label: str) -> dict:
    if label == "Power User":
        return {"action":"Retain","message":"No action needed. Monitor for credit exhaustion.","priority":"Low"}
    if features.get("agent_chats_started", 0) == 0:
        return {"action":"Activate","message":"Prompt user to start their first AI chat. Users who chat in week 1 are 6.6x more likely to become Power Users.","priority":"Critical"}
    if features.get("unique_event_types", 0) < 5:
        return {"action":"Expand","message":"Guide user to explore more platform features. Power Users use 20 unique event types vs 1 for Visitors.","priority":"High"}
    if features.get("used_python_sdk", 0) == 0 and features.get("used_web_app", 0) == 1:
        return {"action":"Deepen","message":"Introduce user to the Python SDK. SDK users represent a distinct high-value cohort with strong retention.","priority":"High"}
    if features.get("session_span_days", 0) < 1:
        return {"action":"Re-engage","message":"Send re-engagement notification. Power Users show median session span of 0.93 days vs 0.0 for Visitors.","priority":"Medium"}
    if features.get("tool_calls_per_chat", 0) < 3:
        return {"action":"Deepen","message":"Encourage deeper agent usage. Show user how to chain agent tool calls for end-to-end workflows.","priority":"Medium"}
    return {"action":"Nurture","message":"User is progressing. Monitor for next natural upgrade trigger.","priority":"Low"}

class UserFeatures(BaseModel):
    total_events:              float = Field(0, ge=0)
    unique_event_types:        float = Field(0, ge=0)
    unique_days_active:        float = Field(0, ge=0)
    unique_canvases:           float = Field(0, ge=0)
    agent_chats_started:       float = Field(0, ge=0)
    agent_messages_sent:       float = Field(0, ge=0)
    agent_started_from_prompt: float = Field(0, ge=0)
    agent_suprise_me:          float = Field(0, ge=0)
    suggestions_accepted:      float = Field(0, ge=0)
    errors_investigated:       float = Field(0, ge=0)
    run_block_manual:          float = Field(0, ge=0)
    run_all_blocks:             float = Field(0, ge=0)
    run_upto_block:            float = Field(0, ge=0)
    total_manual_executions:   float = Field(0, ge=0)
    canvases_created:          float = Field(0, ge=0)
    canvases_opened:           float = Field(0, ge=0)
    blocks_created_manual:     float = Field(0, ge=0)
    blocks_deleted:            float = Field(0, ge=0)
    blocks_renamed:            float = Field(0, ge=0)
    blocks_copied:             float = Field(0, ge=0)
    edges_created:             float = Field(0, ge=0)
    files_uploaded:            float = Field(0, ge=0)
    files_downloaded:          float = Field(0, ge=0)
    signed_up:                 float = Field(0, ge=0)
    new_user_created:          float = Field(0, ge=0)
    submitted_onboarding_form: float = Field(0, ge=0)
    skipped_onboarding:        float = Field(0, ge=0)
    completed_onboarding_tour: float = Field(0, ge=0)
    fullscreen_opens:          float = Field(0, ge=0)
    link_clicks:               float = Field(0, ge=0)
    session_span_days:         float = Field(0, ge=0)
    used_python_sdk:           float = Field(0, ge=0)
    used_web_app:              float = Field(0, ge=0)
    tool_calls_per_chat:       float = Field(0, ge=0)
    execution_per_chat:        float = Field(0, ge=0)

class BatchRequest(BaseModel):
    users: list[dict]

def predict_one(features: dict) -> dict:
    X     = pd.DataFrame([features])[FEATURES].fillna(0)
    probs = model.predict_proba(X)[0]
    tier  = int(np.argmax(probs))
    label = TIER_LABELS[tier]
    sv    = explainer.shap_values(X)[0, :, tier]
    top   = np.argsort(np.abs(sv))[-3:][::-1]
    return {
        "predicted_tier":   tier,
        "predicted_label":  label,
        "description":      TIER_DESC[label],
        "risk_level":       RISK_LEVEL[label],
        "confidence":       round(float(probs[tier]), 4),
        "tier_probabilities": {TIER_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "top_behavioral_drivers": [
            {"feature": FEATURES[i], "value": round(float(X.iloc[0][FEATURES[i]]), 4),
             "shap_impact": round(float(sv[i]), 4),
             "direction": "increases" if sv[i] > 0 else "decreases"}
            for i in top
        ],
        "recommended_intervention": recommend_action(features, label),
        "predicted_at": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/")
def root():
    return {
        "name":        "Zerve User Success Predictor",
        "version":     "2.0.0",
        "killer_stat": "Users who start an AI chat in week 1 are 6.6x more likely to become Power Users",
        "docs":        "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}

@app.get("/model/info")
def info():
    return {
        "model": "XGBoostClassifier", "features": len(FEATURES),
        "feature_names": FEATURES, "tiers": TIER_LABELS,
        "accuracy": 0.95, "auc": 0.994, "cv_accuracy": "0.947 +/- 0.006",
        "baseline": 0.72, "improvement": "+23pp over majority-class baseline",
        "killer_stat": "6.6x — week-1 chat users vs non-chat users Power User conversion",
        "trained_on": "409,287 events | 4,774 users | Sep-Dec 2025",
    }

@app.post("/predict")
def predict(user: UserFeatures):
    try:
        return predict_one(user.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def batch(req: BatchRequest):
    if len(req.users) > 500:
        raise HTTPException(status_code=400, detail="Max 500 users per batch.")
    try:
        return {"count": len(req.users),
                "results": [{**predict_one(u), "index": i} for i, u in enumerate(req.users)]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/example")
def example():
    return predict_one(UserFeatures(
        total_events=150, unique_event_types=12, unique_days_active=5,
        unique_canvases=4, agent_chats_started=8, agent_messages_sent=20,
        submitted_onboarding_form=1, session_span_days=4.5,
        used_web_app=1, tool_calls_per_chat=6.5,
    ).model_dump())

@app.get("/demo/user-journey")
def user_journey():
    """Live story: shows how predictions change as a user progresses through Zerve."""
    steps = [
        {"label": "Day 1 — Just signed up",        "features": {"total_events": 2, "signed_up": 1}},
        {"label": "Day 1 — Started first AI chat",  "features": {"total_events": 8, "agent_chats_started": 1, "unique_event_types": 3, "used_web_app": 1}},
        {"label": "Day 3 — Running blocks",         "features": {"total_events": 25, "agent_chats_started": 2, "unique_event_types": 6, "run_block_manual": 3, "tool_calls_per_chat": 4.0, "used_web_app": 1}},
        {"label": "Day 7 — Deep agent usage",       "features": {"total_events": 80, "agent_chats_started": 6, "unique_event_types": 12, "tool_calls_per_chat": 8.0, "session_span_days": 3.0, "used_web_app": 1, "unique_canvases": 3}},
    ]
    journey = []
    for step in steps:
        r = predict_one(step["features"])
        journey.append({
            "step":                     step["label"],
            "predicted_label":          r["predicted_label"],
            "risk_level":               r["risk_level"],
            "confidence":               r["confidence"],
            "recommended_intervention": r["recommended_intervention"],
        })
    return {
        "story":   "How predictions evolve as a user engages more deeply with Zerve",
        "journey": journey,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
