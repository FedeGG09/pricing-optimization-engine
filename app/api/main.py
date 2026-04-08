from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from app.core.config import settings
from app.core.security import create_access_token

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from app.api.agents_routes import router as agents_router
from app.api.shared_runtime import (
    AUDIT_TABLE,
    SQLITE_PATH,
    build_enriched_payload,
    ensure_audit_schema,
    json_safe,
    load_metrics,
    load_reference_list,
    log_decision,
    pricing_response,
    read_audit_decision,
    safe_json_load_text,
    table_exists,
)
from app.pricing_engine.dynamic_pricing_engine import get_engine

app = FastAPI(title="Dynamic Pricing API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router)
engine = get_engine()


class PricingRequest(BaseModel):
    account_id: str
    product_id: str

    month: Optional[int] = None
    dayofweek: Optional[int] = None
    list_price_usd: Optional[float] = None
    base_price_usd: Optional[float] = None
    negotiated_price_usd: Optional[float] = None
    quantity: Optional[float] = None
    stock_units: Optional[float] = None
    province: Optional[str] = None
    zone: Optional[str] = None

    segment_rule_based: Optional[str] = None
    segment_kmeans_label: Optional[str] = None

    risk_score_0_100: Optional[float] = None
    rfm_score: Optional[float] = None
    price_floor_usd: Optional[float] = None
    price_ceiling_usd: Optional[float] = None
    avg_discount_pct: Optional[float] = None
    avg_margin_pct: Optional[float] = None
    elasticity_estimate: Optional[float] = None
    urgency_score: Optional[float] = None
    churn_risk: Optional[float] = None
    compliance_rate: Optional[float] = None
    seasonality_index: Optional[float] = None
    price_sensitivity: Optional[float] = None
    annual_budget_multiplier: Optional[float] = None
    service_attachment: Optional[float] = None
    margin_target_pct: Optional[float] = None
    industry_fit_score: Optional[float] = None
    funnel_stage: Optional[str] = None
    next_best_action: Optional[str] = None
    is_peak_season_proxy: Optional[int] = None
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None


class SimulationRequest(PricingRequest):
    candidate_multipliers: Optional[List[float]] = None


class LookupResponse(BaseModel):
    count: int
    rows: List[Dict[str, Any]]


def read_table(table_name: str) -> pd.DataFrame:
    if not SQLITE_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, table_name):
            return pd.DataFrame()
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()


def latest_audit_rows(limit: int = 20) -> list[dict[str, Any]]:
    if not SQLITE_PATH.exists():
        return []

    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, AUDIT_TABLE):
            return []
        df = pd.read_sql_query(
            f"SELECT * FROM {AUDIT_TABLE} ORDER BY id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )

    rows = df.to_dict(orient="records")
    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                **row,
                "request_json": safe_json_load_text(row.get("request_json")),
                "enriched_payload_json": safe_json_load_text(row.get("enriched_payload_json")),
                "response_json": safe_json_load_text(row.get("response_json")),
            }
        )
    return parsed_rows


def _table_count_payload(limit: int = 20) -> Dict[str, Any]:
    rows = latest_audit_rows(limit)
    return {"count": len(rows), "rows": rows}


@app.on_event("startup")
def startup_event() -> None:
    ensure_audit_schema()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/pricing/model-metrics")
def model_metrics() -> Dict[str, Any]:
    return load_metrics()


@app.post("/pricing/recommend")
def recommend_price(req: PricingRequest) -> Dict[str, Any]:
    try:
        context = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=(
                {
                    k: v
                    for k, v in req.model_dump().items() if k not in {"account_id", "product_id"} and v is not None
                }
                if hasattr(req, "model_dump")
                else {
                    k: v
                    for k, v in req.dict().items() if k not in {"account_id", "product_id"} and v is not None
                }
            ),
        )
        recommendation = engine.recommend(context["payload"])
        decision_id = log_decision(
            endpoint="/pricing/recommend",
            account_id=req.account_id,
            product_id=req.product_id,
            context_source=context["context_source"],
            request_json=req.model_dump() if hasattr(req, "model_dump") else req.dict(),
            enriched_payload_json=context["payload"],
            response_json=json_safe(recommendation),
            status_code=200,
            model_version=getattr(recommendation, "model_version", None),
        )
        return pricing_response(
            recommendation=recommendation,
            context=context,
            decision_id=decision_id,
            fallback_used=(context["context_source"] != "historical_exact_match"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log_decision(
            endpoint="/pricing/recommend",
            account_id=req.account_id,
            product_id=req.product_id,
            context_source="error",
            request_json=req.model_dump() if hasattr(req, "model_dump") else req.dict(),
            enriched_payload_json={"error": "failed_to_build_payload"},
            response_json={"error": str(exc)},
            status_code=500,
            error=str(exc),
            model_version=None,
        )
        raise HTTPException(status_code=500, detail=f"Error generando recomendación: {exc}") from exc


@app.post("/pricing/explain")
def explain_price(req: PricingRequest) -> Dict[str, Any]:
    try:
        context = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=(
                {
                    k: v
                    for k, v in req.model_dump().items() if k not in {"account_id", "product_id"} and v is not None
                }
                if hasattr(req, "model_dump")
                else {
                    k: v
                    for k, v in req.dict().items() if k not in {"account_id", "product_id"} and v is not None
                }
            ),
        )
        explanation = engine.explain(context["payload"])
        decision_id = log_decision(
            endpoint="/pricing/explain",
            account_id=req.account_id,
            product_id=req.product_id,
            context_source=context["context_source"],
            request_json=req.model_dump() if hasattr(req, "model_dump") else req.dict(),
            enriched_payload_json=context["payload"],
            response_json=json_safe(explanation),
            status_code=200,
            model_version=getattr(explanation, "model_version", None) if not isinstance(explanation, dict) else explanation.get("model_version"),
        )
        return {
            "status": "ok",
            "decision_id": decision_id,
            "context_source": context["context_source"],
            "fallback_used": context["context_source"] != "historical_exact_match",
            "context": context,
            "explanation": json_safe(explanation),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error explicando recomendación: {exc}") from exc


@app.post("/pricing/simulate")
def simulate_price(req: SimulationRequest) -> Dict[str, Any]:
    try:
        context = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=(
                {
                    k: v
                    for k, v in req.model_dump().items() if k not in {"account_id", "product_id", "candidate_multipliers"} and v is not None
                }
                if hasattr(req, "model_dump")
                else {
                    k: v
                    for k, v in req.dict().items() if k not in {"account_id", "product_id", "candidate_multipliers"} and v is not None
                }
            ),
        )
        scenarios = engine.simulate(context["payload"], candidate_multipliers=req.candidate_multipliers)
        scenarios_dict = [json_safe(s) for s in scenarios]
        decision_id = log_decision(
            endpoint="/pricing/simulate",
            account_id=req.account_id,
            product_id=req.product_id,
            context_source=context["context_source"],
            request_json=req.model_dump() if hasattr(req, "model_dump") else req.dict(),
            enriched_payload_json=context["payload"],
            response_json={"scenarios": scenarios_dict},
            status_code=200,
            model_version=getattr(engine, "model_version", None),
        )
        return {
            "status": "ok",
            "decision_id": decision_id,
            "context_source": context["context_source"],
            "fallback_used": context["context_source"] != "historical_exact_match",
            "context": context,
            "scenarios": scenarios_dict,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error simulando escenarios: {exc}") from exc


@app.get("/pricing/audit/latest")
def audit_latest(limit: int = 20) -> Dict[str, Any]:
    return {"count": len(latest_audit_rows(limit)), "rows": latest_audit_rows(limit)}


@app.get("/pricing/audit/{decision_id}")
def audit_decision(decision_id: int) -> Dict[str, Any]:
    row = read_audit_decision(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail="No se encontró la decisión")

    return {
        **row,
        "request_json": safe_json_load_text(row.get("request_json")),
        "enriched_payload_json": safe_json_load_text(row.get("enriched_payload_json")),
        "response_json": safe_json_load_text(row.get("response_json")),
    }


@app.get("/reference/accounts", response_model=LookupResponse)
def list_accounts(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    rows = load_reference_list("account_master", "account", limit=limit)
    if not rows:
        rows = load_reference_list("account_features", "account_id", limit=limit)
    return {"count": len(rows), "rows": rows}


@app.get("/reference/products", response_model=LookupResponse)
def list_products(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    rows = load_reference_list("product_master", "product", limit=limit)
    if not rows:
        rows = load_reference_list("product_features", "product_id", limit=limit)
    return {"count": len(rows), "rows": rows}


@app.get("/reference/provinces", response_model=LookupResponse)
def list_provinces(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for table_name in ["account_master", "account_features", "pricing_training_set"]:
        df = read_table(table_name)
        if df.empty:
            continue
        for col in ["province", "province_name", "region", "zone"]:
            if col in df.columns:
                series = df[col].dropna().astype(str).str.strip()
                rows.extend([{"id": v, "name": v} for v in sorted(series.unique().tolist())])
                break
        if rows:
            break
    rows = rows[:limit]
    return {"count": len(rows), "rows": rows}


@app.get("/reference/zones", response_model=LookupResponse)
def list_zones(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for table_name in ["account_master", "account_features", "pricing_training_set", "product_master", "product_features"]:
        df = read_table(table_name)
        if df.empty:
            continue

        for col in ["zone", "zone_name", "region", "region_name", "territory", "province", "province_name"]:
            if col in df.columns:
                series = df[col].dropna().astype(str).str.strip()
                unique_values = sorted([v for v in series.unique().tolist() if v and v.lower() != "nan"])
                candidates.extend([{"id": v, "name": v} for v in unique_values])

    seen = set()
    rows = []
    for item in candidates:
        key = item["name"]
        if key not in seen:
            seen.add(key)
            rows.append(item)

    rows = rows[:limit]
    if not rows:
        rows = [{"id": "default", "name": "Zona no disponible"}]

    return {"count": len(rows), "rows": rows}


@app.get("/reference/catalog")
def catalog() -> Dict[str, Any]:
    return {
        "accounts": load_reference_list("account_master", "account", limit=5000),
        "products": load_reference_list("product_master", "product", limit=5000),
        "provinces": list_provinces(limit=5000)["rows"],
        "zones": list_zones(limit=5000)["rows"],
        "metrics": model_metrics(),
    }
#
