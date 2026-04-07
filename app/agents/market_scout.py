from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from app.agents.schemas import MarketInsight


@dataclass
class MarketScoutAgent:
    """
    Market Scout:
    - No requiere LLM para funcionar.
    - Usa señales internas como provincia, zona, stock y estacionalidad.
    - Después se puede sumar una capa HF para redactar el brief.
    """

    def analyze(self, context: dict[str, Any]) -> MarketInsight:
        payload = context.get("payload", {})
        province = payload.get("province") or context.get("account_row", {}).get("province")
        zone = payload.get("zone") or context.get("account_row", {}).get("zone")

        seasonality = float(payload.get("seasonality_index") or 0.0)
        stock_units = float(payload.get("stock_units") or 0.0)
        risk_score = float(payload.get("risk_score_0_100") or 50.0)
        margin_target = float(payload.get("margin_target_pct") or 0.25)
        price_floor = payload.get("price_floor_usd")
        price_ceiling = payload.get("price_ceiling_usd")
        list_price = payload.get("list_price_usd")
        negotiated = payload.get("negotiated_price_usd")

        competitive_pressure = 50.0
        if seasonality > 0:
            competitive_pressure -= min(seasonality * 15.0, 15.0)
        else:
            competitive_pressure += min(abs(seasonality) * 10.0, 10.0)

        if stock_units <= 5:
            competitive_pressure += 12.0
        elif stock_units >= 100:
            competitive_pressure -= 6.0

        if risk_score >= 70:
            competitive_pressure -= 8.0

        if margin_target >= 0.30:
            competitive_pressure += 4.0

        if list_price and negotiated and float(list_price) > 0:
            price_position = 1.0 - (float(negotiated) / float(list_price))
            if price_position > 0.12:
                competitive_pressure += 6.0

        competitive_pressure = max(0.0, min(100.0, competitive_pressure))

        if competitive_pressure >= 70:
            positioning = "premium"
            recommended_positioning = "protect_margin"
        elif competitive_pressure >= 45:
            positioning = "balanced"
            recommended_positioning = "maintain"
        else:
            positioning = "aggressive"
            recommended_positioning = "seek_conversion"

        summary = (
            f"Provincia={province or 'n/d'}, zona={zone or 'n/d'}. "
            f"Presión competitiva estimada={competitive_pressure:.0f}/100. "
            f"Estrategia sugerida={recommended_positioning}."
        )

        return MarketInsight(
            province=str(province) if province is not None else None,
            zone=str(zone) if zone is not None else None,
            market_pressure_0_100=competitive_pressure,
            competitive_position=positioning,
            summary=summary,
            recommended_positioning=recommended_positioning,
        )