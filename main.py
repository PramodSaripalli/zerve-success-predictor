"""
ZERVE SUCCESS PREDICTOR — FastAPI Endpoint (Deployment Ready)
"""

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

MODEL_PATH = "zerve_xgb_model.json"
TIER_LABELS = {0: "Visitor", 1: "Explorer", 2: "Engaged", 3: "Power User"}
TIER_DESCRIPTIONS = {
    "Visitor":    "Signed in but minimal platform interaction. High churn risk.",
    "Explorer":   "Started at least one AI chat. Curious but not yet committed.",
    "Engaged":    "Regular usage via Python SDK or agent tools. Likely a developer.",
    "Power User": "High event volume across diverse behaviors. Core retained user.",
}
FEATURE_COLS = [
    "total_events", "unique_event_types", "unique_days_active",
    "unique_canvases", "agent_chats_started", "agent_messages_sent",
    "agent_started_from_prompt", "agent_suprise_me", "suggestions_accepted",
    "errors_investigated", "run_block_manual", "run_all_blocks",
    "run_upto_block", "total_manual_executions", "canvases_created",
    "canvases_opened", "blocks_created_manual", "blocks_deleted",
    "blocks_renamed", "blocks_copied", "edges_created", "files_uploaded",
    "files_downloaded", "signed_up", "new_user_created",
    "submitted_onboarding_form", "skipped_onboarding",
    "completed_onboarding_tour", "fullscreen_opens", "link_clicks",
    "session_span_days", "used_python_sdk", "used_web_app",
    "tool_calls_per_chat", "execution_per_chat",
]

print("[API] Loading model...")
model = XGBClassifier()
model.load_model(MODEL_PATH)
explainer = shap.TreeExplainer(model)
print("[API] Model ready.")

app = FastAPI(
    title="Zerve User Success Predictor",
    description="Predicts which success tier a Zerve user will reach based on their behavioral signals. Built for the Zerve x HackerEarth Hackathon 2026.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    run_all_blocks:            float = Field(0, ge=0)
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

def predict_single(features: dict) -> dict:
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[0]
    tier = int(np.argmax(probs))
    label = TIER_LABELS[tier]
    confidence = round(float(probs[tier]), 4)
    tier_probs = {TIER_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)}
    shap_vals = explainer.shap_values(X)
    sv_for_tier = shap_vals[0, :, tier]
    top_idx = np.argsort(np.abs(sv_for_tier))[-3:][::-1]
    drivers = [
        {
            "feature": FEATURE_COLS[i],
            "value": round(float(X.iloc[0][FEATURE_COLS[i]]), 4),
            "shap_impact": round(float(sv_for_tier[i]), 4),
            "direction": "increases" if sv_for_tier[i] > 0 else "decreases",
        }
        for i in top_idx
    ]
    return {
        "predicted_tier": tier,
        "predicted_label": label,
        "description": TIER_DESCRIPTIONS[label],
        "confidence": confidence,
        "tier_probabilities": tier_probs,
        "top_behavioral_drivers": drivers,
        "predicted_at": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/")
def root():
    return {"name": "Zerve User Success Predictor", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "features": len(FEATURE_COLS)}

@app.get("/model/info")
def model_info():
    return {
        "model_type": "XGBoostClassifier",
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "tiers": TIER_LABELS,
        "tier_descriptions": TIER_DESCRIPTIONS,
        "accuracy": 0.95,
        "auc": 0.994,
        "cv_accuracy": "0.947 +/- 0.006",
        "baseline_accuracy": 0.72,
        "improvement_over_baseline": "23 percentage points",
        "trained_on": "409,287 events | 4,774 users | Sep-Dec 2025",
    }

@app.post("/predict")
def predict(user: UserFeatures):
    try:
        return predict_single(user.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if len(request.users) > 500:
        raise HTTPException(status_code=400, detail="Max 500 users per batch.")
    try:
        results = []
        for i, user_dict in enumerate(request.users):
            result = predict_single(user_dict)
            result["index"] = i
            results.append(result)
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/example")
def predict_example():
    example = UserFeatures(
        total_events=150, unique_event_types=12, unique_days_active=5,
        unique_canvases=4, agent_chats_started=8, agent_messages_sent=20,
        agent_started_from_prompt=3, suggestions_accepted=5, run_block_manual=6,
        total_manual_executions=10, canvases_created=3, canvases_opened=12,
        submitted_onboarding_form=1, session_span_days=4.5, used_web_app=1,
        tool_calls_per_chat=6.5, execution_per_chat=1.2,
    )
    return predict_single(example.model_dump())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("05_api:app", host="0.0.0.0", port=port, reload=False)
