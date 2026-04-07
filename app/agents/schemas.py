from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MarketInsight(BaseModel):
    province: Optional[str] = None
    zone: Optional[str] = None
    market_pressure_0_100: float = Field(default=50.0, ge=0, le=100)
    competitive_position: str = "balanced"
    summary: str = ""
    recommended_positioning: str = "maintain"


class BehavioralProfile(BaseModel):
    account_id: str
    segment: str = "new_or_low_activity"
    sensitivity_bucket: str = "medium"
    recency_days: Optional[float] = None
    frequency_90d: Optional[float] = None
    rfm_score: Optional[float] = None
    churn_risk: Optional[float] = None
    compliance_rate: Optional[float] = None
    price_sensitivity: Optional[float] = None
    summary: str = ""


class DiscountGuardrail(BaseModel):
    allowed: bool = True
    floor_price_usd: Optional[float] = None
    ceiling_price_usd: Optional[float] = None
    max_discount_pct: float = Field(default=0.12, ge=0, le=1)
    min_margin_pct: float = Field(default=0.25, ge=0, le=1)
    blocked_reason: Optional[str] = None
    rule_hits: list[str] = []


class AnomalyFinding(BaseModel):
    code: str
    severity: str = "low"
    title: str
    detail: str
    recommended_action: str


class StrategyRecommendation(BaseModel):
    title: str
    executive_summary: str
    action_plan: list[str] = []
    expected_effect: str = ""
    risks: list[str] = []


class CommercialNarrative(BaseModel):
    executive_summary: str
    client_context: str
    product_context: str
    zone_context: str
    recommended_action: str
    key_arguments: list[str] = []
    risks: list[str] = []
    tone: str = "balanced"
    confidence: float = Field(default=0.7, ge=0, le=1)


class AgentBundle(BaseModel):
    market: MarketInsight
    behavior: BehavioralProfile
    guardrail: DiscountGuardrail
    anomalies: list[AnomalyFinding] = []
    strategy: Optional[StrategyRecommendation] = None
    narrative: Optional[CommercialNarrative] = None
    extra: dict[str, Any] = {}