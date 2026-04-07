from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.pricing_engine.service import PricingService

router = APIRouter()
pricing_service = PricingService()


class PricingRecommendRequest(BaseModel):
    product_id: str
    account_id: str | None = None
    transaction_date: str | None = None
    current_price: float | None = None
    list_price: float | None = None
    base_price: float | None = None
    cost_price: float | None = None
    quantity: float = 1.0
    province: str | None = None
    sector: str | None = None
    industry: str | None = None
    customer_segment: str = "recurrent"
    stock_available: float | None = None
    seasonality_index: float | None = None
    macro_index: float | None = None
    demand_history_index: float | None = None
    event_factor: float | None = None
    competitor_price_index: float | None = None
    logistics_cost_index: float | None = None
    purchasing_power_index: float | None = None
    discount_history_avg: float | None = None
    price_sensitivity: float | None = None
    elasticity_estimate: float | None = None
    payment_behavior_score: float | None = None
    loyalty_score: float | None = None
    frequency_12m: float | None = None
    days_since_last_purchase: int | None = None
    overdue_days: int = 0
    on_time_payment_ratio: float | None = None
    payment_type: str | None = None
    min_margin_pct: float = 0.20
    discount_cap_pct: float = 0.15
    margin_target_pct: float = 0.30
    explain: bool = True
    extra: dict = Field(default_factory=dict)


class PricingSimulateRequest(PricingRecommendRequest):
    price_deltas_pct: list[float] = Field(default_factory=lambda: [-0.10, -0.05, 0.0, 0.05, 0.10])
    base_units: float | None = None


class PricingExplainRequest(PricingRecommendRequest):
    pass


@router.post("/pricing/recommend")
def pricing_recommend(body: PricingRecommendRequest, user=Depends(get_current_user)):
    result = pricing_service.recommend(body.model_dump())
    return {"user": user, "result": result}


@router.post("/pricing/simulate")
def pricing_simulate(body: PricingSimulateRequest, user=Depends(get_current_user)):
    result = pricing_service.simulate(body.model_dump())
    return {"user": user, "result": result}


@router.post("/pricing/explain")
def pricing_explain(body: PricingExplainRequest, user=Depends(get_current_user)):
    result = pricing_service.explain(body.model_dump())
    return {"user": user, "result": result}


@router.post("/pricing/train")
def pricing_train(user=Depends(get_current_user)):
    result = pricing_service.train()
    return {"user": user, "result": result}


@router.get("/health")
def health():
    return {"status": "ok", "service": "pricing-engine"}


@router.get("/metrics")
def metrics():
    from app.core.observability import metrics_response
    return metrics_response()
