from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RuleResult:
    model_price: float
    final_price: float
    discount_pct: float
    min_allowed_price: float
    max_allowed_price: float
    factor_scores: dict[str, float]
    factor_impact: dict[str, float]
    risk_flags: list[str]
    suggested_action: str
    explanation: str
    rule_trace: list[str]


class PricingRuleEngine:
    def __init__(self, min_floor_buffer: float = 0.0):
        self.min_floor_buffer = min_floor_buffer

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    def apply(
        self,
        row: pd.Series,
        model_price: float,
        current_price: float,
        model_margin_pct: float | None = None,
    ) -> RuleResult:
        cost_price = float(row.get("cost_price", current_price * 0.65))
        min_margin_pct = float(row.get("min_margin_pct", 0.20))
        margin_target_pct = float(row.get("margin_target_pct", 0.30))
        discount_cap_pct = float(row.get("discount_cap_pct", 0.15))
        min_allowed_price = max(
            float(row.get("min_price_usd", 0.0)),
            cost_price * (1.0 + min_margin_pct),
            current_price * (1.0 - discount_cap_pct),
        )
        max_allowed_price = max(
            float(row.get("max_price_usd", current_price * 1.15)),
            current_price * 1.02,
            model_price * 1.05,
        )

        seasonality = float(row.get("seasonality_index", 0.0))
        loyalty = float(row.get("loyalty_score", 0.5))
        payment = float(row.get("payment_behavior_score", 0.5))
        logistics = float(row.get("logistics_cost_index", 0.7))
        competition = float(row.get("competitor_price_index", 0.7))
        purchasing_power = float(row.get("purchasing_power_index", 0.7))
        inventory_pressure = float(row.get("inventory_pressure", 0.5))
        elasticity = float(row.get("elasticity_estimate", 1.0))
        urgency = float(row.get("urgency_score", 0.5))
        demand_pressure = float(row.get("demand_pressure", 0.0))

        seasonal_multiplier = 1.0 + (0.04 * seasonality) + float(row.get("event_factor", 0.0))
        loyalty_multiplier = 1.0 - max(0.0, loyalty - 0.5) * 0.05
        payment_multiplier = 1.0 - max(0.0, payment - 0.5) * 0.03
        location_multiplier = 1.0 + (0.03 * (1 - logistics)) + (0.02 * (1 - competition)) - (0.02 * (purchasing_power - 0.7))
        stock_multiplier = 1.0 + (0.05 if inventory_pressure <= 0.3 else 0.02 if inventory_pressure <= 0.7 else -0.05)
        urgency_multiplier = 1.0 + 0.03 * max(0.0, urgency - 0.5)
        elasticity_multiplier = 1.0 - 0.04 * max(0.0, elasticity - 1.0)
        demand_multiplier = 1.0 + 0.03 * demand_pressure

        combined_multiplier = (
            seasonal_multiplier
            * loyalty_multiplier
            * payment_multiplier
            * location_multiplier
            * stock_multiplier
            * urgency_multiplier
            * elasticity_multiplier
            * demand_multiplier
        )

        adjusted_by_rules = model_price * combined_multiplier
        final_price = self._clamp(adjusted_by_rules, min_allowed_price, max_allowed_price)

        if model_margin_pct is None and current_price > 0:
            model_margin_pct = max(0.0, (current_price - cost_price) / current_price)

        discount_pct = max(0.0, 1.0 - final_price / current_price) if current_price > 0 else 0.0
        current_margin_pct = max(0.0, (current_price - cost_price) / current_price) if current_price > 0 else 0.0
        final_margin_pct = max(0.0, (final_price - cost_price) / final_price) if final_price > 0 else 0.0

        risk_flags: list[str] = []
        if final_price <= min_allowed_price * 1.001:
            risk_flags.append("at_floor")
        if final_price >= max_allowed_price * 0.999:
            risk_flags.append("at_ceiling")
        if float(row.get("payment_risk", 0.4)) > 0.6:
            risk_flags.append("payment_risk_high")
        if float(row.get("stock_available", 999.0)) <= 3:
            risk_flags.append("low_stock")
        if float(row.get("churn_risk", 0.3)) > 0.6:
            risk_flags.append("churn_risk_high")
        if float(row.get("margin_buffer_pct", 0.25)) < 0.1:
            risk_flags.append("thin_margin")
        if final_margin_pct < margin_target_pct:
            risk_flags.append("below_target_margin")

        if "low_stock" in risk_flags and "payment_risk_high" in risk_flags:
            suggested_action = "hold_price_with_prepayment"
        elif final_price < current_price:
            suggested_action = "discount"
        elif final_price > current_price:
            suggested_action = "increase"
        else:
            suggested_action = "hold"

        rule_trace = [
            f"seasonality_multiplier={seasonal_multiplier:.4f}",
            f"loyalty_multiplier={loyalty_multiplier:.4f}",
            f"payment_multiplier={payment_multiplier:.4f}",
            f"location_multiplier={location_multiplier:.4f}",
            f"stock_multiplier={stock_multiplier:.4f}",
            f"urgency_multiplier={urgency_multiplier:.4f}",
            f"elasticity_multiplier={elasticity_multiplier:.4f}",
            f"demand_multiplier={demand_multiplier:.4f}",
            f"combined_multiplier={combined_multiplier:.4f}",
            f"clamped_range=[{min_allowed_price:.2f}, {max_allowed_price:.2f}]",
        ]

        explanation = (
            f"El modelo propuso {model_price:.2f} y las reglas lo ajustaron a {final_price:.2f}. "
            f"Se protegió un piso mínimo de {min_allowed_price:.2f} y un techo de {max_allowed_price:.2f}. "
            f"El margen estimado pasó de {current_margin_pct:.2%} a {final_margin_pct:.2%}."
        )

        factor_scores = {}
        factor_impact = {}
        return RuleResult(
            model_price=round(float(model_price), 2),
            final_price=round(float(final_price), 2),
            discount_pct=round(float(discount_pct), 4),
            min_allowed_price=round(float(min_allowed_price), 2),
            max_allowed_price=round(float(max_allowed_price), 2),
            factor_scores=factor_scores,
            factor_impact=factor_impact,
            risk_flags=risk_flags,
            suggested_action=suggested_action,
            explanation=explanation,
            rule_trace=rule_trace,
        )
