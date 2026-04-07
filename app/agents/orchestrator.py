from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.agents.behavioral_profiler import BehavioralProfilerAgent
from app.agents.discount_logic import DiscountLogicAgent
from app.agents.market_scout import MarketScoutAgent
from app.agents.narrator import NarratorAgent
from app.agents.schemas import AgentBundle, AnomalyFinding, StrategyRecommendation
from app.services.pricing_context import PricingContextService


@dataclass
class PricingAgentOrchestrator:
    context_service: PricingContextService = None  # type: ignore[assignment]
    market_scout: MarketScoutAgent = None  # type: ignore[assignment]
    behavioral_profiler: BehavioralProfilerAgent = None  # type: ignore[assignment]
    discount_logic: DiscountLogicAgent = None  # type: ignore[assignment]
    narrator: NarratorAgent = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.context_service is None:
            self.context_service = PricingContextService()
        if self.market_scout is None:
            self.market_scout = MarketScoutAgent()
        if self.behavioral_profiler is None:
            self.behavioral_profiler = BehavioralProfilerAgent()
        if self.discount_logic is None:
            self.discount_logic = DiscountLogicAgent()
        if self.narrator is None:
            self.narrator = NarratorAgent()

    def build_bundle(
        self,
        *,
        account_id: str,
        product_id: str,
        overrides: Optional[dict[str, Any]] = None,
        pricing_snapshot: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        context = self.context_service.build_context(
            account_id=account_id,
            product_id=product_id,
            overrides=overrides,
        )

        market = self.market_scout.analyze(context)
        behavior = self.behavioral_profiler.analyze(context)

        recommendation = pricing_snapshot or {}
        guardrail = self.discount_logic.evaluate(context, recommendation=recommendation)
        anomalies = self.detect_anomalies(
            context=context,
            recommendation=recommendation,
            guardrail=guardrail.model_dump() if hasattr(guardrail, "model_dump") else guardrail.__dict__,
        )

        narrative = self.narrator.explain(
            context=context,
            recommendation=recommendation,
            market=market.model_dump() if hasattr(market, "model_dump") else market.__dict__,
            behavior=behavior.model_dump() if hasattr(behavior, "model_dump") else behavior.__dict__,
            guardrail=guardrail.model_dump() if hasattr(guardrail, "model_dump") else guardrail.__dict__,
            anomalies=[a.model_dump() if hasattr(a, "model_dump") else a.__dict__ for a in anomalies],
        )

        bundle = AgentBundle(
            market=market,
            behavior=behavior,
            guardrail=guardrail,
            anomalies=anomalies,
            strategy=self.recommend_strategy(context=context, recommendation=recommendation, guardrail=guardrail),
            narrative=narrative,
            extra={
                "context_source": context.get("context_source"),
                "used_sources": context.get("used_sources"),
            },
        )

        return bundle.model_dump() if hasattr(bundle, "model_dump") else bundle.dict()

    def detect_anomalies(
        self,
        *,
        context: dict[str, Any],
        recommendation: dict[str, Any],
        guardrail: dict[str, Any],
    ) -> list[AnomalyFinding]:
        payload = context.get("payload", {})
        anomalies: list[AnomalyFinding] = []

        floor_price = self._to_float(recommendation.get("floor_price_usd") or guardrail.get("floor_price_usd") or payload.get("price_floor_usd"))
        ceiling_price = self._to_float(recommendation.get("ceiling_price_usd") or guardrail.get("ceiling_price_usd") or payload.get("price_ceiling_usd"))
        recommended_price = self._to_float(recommendation.get("recommended_price_usd") or payload.get("negotiated_price_usd"))
        list_price = self._to_float(payload.get("list_price_usd"))
        base_price = self._to_float(payload.get("base_price_usd"))
        margin_target = self._to_float(payload.get("margin_target_pct"), 0.25) or 0.25
        risk_score = self._to_float(payload.get("risk_score_0_100"), 50.0) or 50.0
        stock_units = self._to_float(payload.get("stock_units"), 10.0) or 10.0

        if floor_price is not None and recommended_price is not None and recommended_price < floor_price:
            anomalies.append(
                AnomalyFinding(
                    code="below_floor",
                    severity="high",
                    title="Precio por debajo del piso",
                    detail=f"El precio recomendado ({recommended_price:.2f}) quedó por debajo del piso ({floor_price:.2f}).",
                    recommended_action="Subir el precio al piso mínimo antes de publicar.",
                )
            )

        if ceiling_price is not None and recommended_price is not None and recommended_price > ceiling_price:
            anomalies.append(
                AnomalyFinding(
                    code="above_ceiling",
                    severity="medium",
                    title="Precio por encima del techo",
                    detail=f"El precio recomendado ({recommended_price:.2f}) excede el techo ({ceiling_price:.2f}).",
                    recommended_action="Reducir el precio al techo máximo permitido.",
                )
            )

        if list_price and recommended_price and list_price > 0:
            discount = 1.0 - (recommended_price / list_price)
            if discount > 0.18:
                anomalies.append(
                    AnomalyFinding(
                        code="deep_discount",
                        severity="medium",
                        title="Descuento demasiado alto",
                        detail=f"El descuento implícito es {discount:.2%}, por encima de una zona conservadora.",
                        recommended_action="Revisar si el descuento está justificado por riesgo o inventario.",
                    )
                )

        if base_price and recommended_price:
            margin = (recommended_price - base_price) / max(recommended_price, 1e-6)
            if margin < margin_target:
                anomalies.append(
                    AnomalyFinding(
                        code="margin_below_target",
                        severity="high",
                        title="Margen debajo del objetivo",
                        detail=f"El margen estimado ({margin:.2%}) está por debajo del objetivo ({margin_target:.2%}).",
                        recommended_action="Ajustar el precio o renegociar condiciones.",
                    )
                )

        if risk_score >= 80:
            anomalies.append(
                AnomalyFinding(
                    code="high_risk_client",
                    severity="medium",
                    title="Cliente de riesgo alto",
                    detail="El score de riesgo del cliente es elevado y conviene proteger margen y condiciones.",
                    recommended_action="Limitar descuentos y pedir aprobación comercial.",
                )
            )

        if stock_units <= 5:
            anomalies.append(
                AnomalyFinding(
                    code="low_stock",
                    severity="medium",
                    title="Stock bajo",
                    detail="El inventario disponible es bajo; el descuento agresivo no ayuda a proteger disponibilidad.",
                    recommended_action="Revisar disponibilidad antes de conceder descuento.",
                )
            )

        return anomalies

    def recommend_strategy(
        self,
        *,
        context: dict[str, Any],
        recommendation: dict[str, Any],
        guardrail: Any = None,
    ) -> StrategyRecommendation:
        payload = context.get("payload", {})
        segment = str(payload.get("segment_rule_based") or payload.get("segment_kmeans_label") or "new_or_low_activity")
        market = self.market_scout.analyze(context)

        if segment == "vip":
            action_plan = [
                "Mantener protección de margen.",
                "Evitar descuentos agresivos.",
                "Ofrecer valor agregado en servicio / disponibilidad.",
            ]
            title = "Estrategia premium"
            expected_effect = "Preserva margen y refuerza posicionamiento."
        elif segment == "at_risk":
            action_plan = [
                "Usar descuento controlado sólo si mejora la conversión.",
                "Proteger margen con piso estricto.",
                "Acompañar con seguimiento comercial.",
            ]
            title = "Estrategia defensiva"
            expected_effect = "Reduce fuga sin destruir rentabilidad."
        else:
            action_plan = [
                "Mantener balance entre cierre y rentabilidad.",
                "Usar argumentos de valor y disponibilidad.",
                "Conceder descuento sólo si mejora la probabilidad de cierre.",
            ]
            title = "Estrategia balanceada"
            expected_effect = "Sostiene conversión y orden de márgenes."

        risks = [
            "Sensibilidad a cambios de precio.",
            "Restricciones del stock o del piso comercial.",
        ]
        if market.market_pressure_0_100 >= 70:
            risks.append("Zona con presión competitiva alta.")

        return StrategyRecommendation(
            title=title,
            executive_summary=f"Segmento {segment} con foco {market.recommended_positioning}.",
            action_plan=action_plan,
            expected_effect=expected_effect,
            risks=risks,
        )

    def what_if_simulation(
        self,
        *,
        context: dict[str, Any],
        recommendation: dict[str, Any],
        candidate_multipliers: list[float],
    ) -> list[dict[str, Any]]:
        payload = context.get("payload", {})
        base_price = self._to_float(recommendation.get("recommended_price_usd") or payload.get("negotiated_price_usd") or payload.get("list_price_usd"), 0.0) or 0.0
        floor_price = self._to_float(recommendation.get("floor_price_usd") or payload.get("price_floor_usd"), base_price * 0.85) or (base_price * 0.85)
        ceiling_price = self._to_float(recommendation.get("ceiling_price_usd") or payload.get("price_ceiling_usd"), base_price * 1.15) or (base_price * 1.15)
        base_cost = self._to_float(payload.get("base_price_usd"), base_price * 0.72) or (base_price * 0.72)
        elasticity = self._to_float(payload.get("elasticity_estimate"), -1.0) or -1.0

        scenarios: list[dict[str, Any]] = []
        for mult in candidate_multipliers:
            candidate = base_price * float(mult)
            candidate = max(floor_price, min(ceiling_price, candidate))
            margin = (candidate - base_cost) / max(candidate, 1e-6)
            demand_factor = max(0.1, 1.0 + elasticity * (candidate / max(base_price, 1e-6) - 1.0))
            revenue_score = candidate * demand_factor
            scenarios.append(
                {
                    "candidate_multiplier": float(mult),
                    "candidate_price_usd": round(candidate, 4),
                    "expected_margin_pct": round(margin, 4),
                    "expected_revenue_score": round(revenue_score, 4),
                    "feasible": margin >= 0,
                    "notes": "OK" if margin >= 0 else "Below margin threshold",
                }
            )
        return scenarios

    @staticmethod
    def _to_float(value: Any, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default