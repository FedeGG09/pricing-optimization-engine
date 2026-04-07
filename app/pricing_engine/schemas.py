from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PricingContext(BaseModel):
    product_id: str
    account_id: str | None = None
    transaction_date: datetime | None = None

    current_price: float | None = None
    list_price: float | None = None
    base_price: float | None = None
    cost_price: float | None = None
    quantity: float = 1.0

    province: str | None = None
    sector: str | None = None
    industry: str | None = None

    customer_segment: Literal["new", "recurrent", "vip", "at_risk"] = "recurrent"
    loyalty_score: float | None = None
    frequency_12m: float | None = None
    days_since_last_purchase: int | None = None

    payment_behavior_score: float | None = None
    overdue_days: int = 0
    on_time_payment_ratio: float | None = None
    payment_type: str | None = None

    stock_available: float | None = None
    stock_rotations_90d: float | None = None
    stock_out_days_90d: float | None = None

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
    category: str | None = None
    family: str | None = None
    subfamily: str | None = None
    business_model: str | None = None
    asset_type: str | None = None
    service_intensity: str | None = None
    criticality: int | None = None

    min_margin_pct: float = 0.20
    discount_cap_pct: float = 0.15
    margin_target_pct: float = 0.30
    explain: bool = True


class SimulationPoint(BaseModel):
    price_delta_pct: float
    price_multiplier: float
    expected_units: float
    expected_revenue: float
    expected_margin: float
    expected_margin_pct: float
    scenario_label: str


class PricingRecommendationResponse(BaseModel):
    product_id: str
    account_id: str | None = None
    timestamp: datetime
    current_price: float
    model_price: float
    final_price: float
    discount_pct: float
    margin_pct: float
    min_allowed_price: float
    max_allowed_price: float
    suggested_action: str
    confidence: float
    risk_flags: list[str] = Field(default_factory=list)
    factor_scores: dict[str, float] = Field(default_factory=dict)
    factor_impact: dict[str, float] = Field(default_factory=dict)
    explanation: str
    llm_explanation: str | None = None
    anomaly_score: float | None = None
    anomaly_detected: bool = False
    model_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PricingSimulationResponse(BaseModel):
    product_id: str
    current_price: float
    base_units: float
    baseline_revenue: float
    baseline_margin: float
    scenarios: list[SimulationPoint]
    explanation: str
    llm_explanation: str | None = None


class PricingExplainResponse(BaseModel):
    product_id: str
    explanation: str
    factor_scores: dict[str, float]
    factor_impact: dict[str, float]
    rule_trace: list[str]
    llm_explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
