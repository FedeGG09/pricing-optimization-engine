from __future__ import annotations

import json
from typing import Any

from app.agents.hf_client import HuggingFaceLLMClient


PRICING_EXPLAIN_SYSTEM = """
Eres un agente senior de pricing dinámico y revenue management.
Debes explicar decisiones con foco comercial, financiero y operativo.
Debes responder en JSON con:
- summary
- key_drivers
- commercial_recommendation
- risks
- next_actions
"""

PRICING_STRATEGY_SYSTEM = """
Eres un estratega comercial industrial.
Debes proponer acciones de negocio concretas para sostener margen, volumen y conversión.
Debes responder en JSON con:
- summary
- strategies
- channels
- expected_impact
- cautions
"""

ANOMALY_SYSTEM = """
Eres un agente de control de anomalías de pricing.
Debes explicar si una recomendación parece errónea, riesgosa o inconsistente.
Debes responder en JSON con:
- summary
- anomaly_reasons
- severity
- recommended_fix
"""

SCENARIO_SYSTEM = """
Eres un agente de simulación de escenarios de pricing.
Debes explicar el efecto esperado de cambios de precio sobre demanda, margen y revenue.
Debes responder en JSON con:
- summary
- scenarios
- interpretation
"""


class PricingAgent:
    def __init__(self, client: HuggingFaceLLMClient | None = None):
        self.client = client or HuggingFaceLLMClient()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {"raw": text}

    def explain(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.client.generate(PRICING_EXPLAIN_SYSTEM, json.dumps(context, ensure_ascii=False, indent=2))
        return self._parse_json(result.text)

    def strategy(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.client.generate(PRICING_STRATEGY_SYSTEM, json.dumps(context, ensure_ascii=False, indent=2))
        return self._parse_json(result.text)

    def anomaly(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.client.generate(ANOMALY_SYSTEM, json.dumps(context, ensure_ascii=False, indent=2))
        return self._parse_json(result.text)

    def scenario(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.client.generate(SCENARIO_SYSTEM, json.dumps(context, ensure_ascii=False, indent=2))
        return self._parse_json(result.text)
