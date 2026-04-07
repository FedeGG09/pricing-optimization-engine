from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.schemas import CommercialNarrative
from app.services.hf_client import HFLLMClient, get_hf_client


@dataclass
class NarratorAgent:
    """
    Narrator:
    - Convierte la salida técnica en lenguaje ejecutivo/comercial.
    - Usa Structured Outputs de Hugging Face para devolver JSON parsable.
    """

    client: HFLLMClient = None  # type: ignore[assignment]
    model: str | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = get_hf_client()
        if self.model is None:
            self.model = None  # usa default del cliente

    def explain(
        self,
        *,
        context: dict[str, Any],
        recommendation: dict[str, Any],
        market: dict[str, Any],
        behavior: dict[str, Any],
        guardrail: dict[str, Any],
        anomalies: list[dict[str, Any]] | None = None,
    ) -> CommercialNarrative:
        payload = context.get("payload", {})
        province = payload.get("province") or "n/d"
        zone = payload.get("zone") or "n/d"
        product_id = payload.get("product_id") or "n/d"
        account_id = payload.get("account_id") or "n/d"

        system_prompt = (
            "Sos un analista comercial senior de revenue management. "
            "Debés explicar recomendaciones de pricing de forma ejecutiva, clara y breve. "
            "Respondé con un JSON válido que respete el esquema pedido."
        )

        user_prompt = f"""
Contexto del caso:
- account_id: {account_id}
- product_id: {product_id}
- province: {province}
- zone: {zone}

Salida del motor de pricing:
{recommendation}

Señales de mercado:
{market}

Señales de comportamiento:
{behavior}

Reglas de seguridad:
{guardrail}

Anomalías detectadas:
{anomalies or []}

Instrucciones:
- executive_summary: 2 o 3 frases para dirección/ventas.
- client_context: explicar el comportamiento del cliente.
- product_context: explicar el producto/categoría.
- zone_context: explicar la zona/provincia.
- recommended_action: decir qué acción comercial seguir.
- key_arguments: 3 a 5 bullets cortos.
- risks: 2 a 4 riesgos o alertas.
- tone: una palabra entre balanced, optimistic, cautious, urgent.
- confidence: valor entre 0 y 1.
"""

        try:
            return self.client.structured(
                schema_model=CommercialNarrative,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                reasoning_effort="medium",
            )
        except Exception:
            # Fallback determinístico si HF no está disponible
            summary = (
                f"El precio recomendado para {product_id} en {province}/{zone} "
                f"prioriza margen y control de riesgo."
            )
            return CommercialNarrative(
                executive_summary=summary,
                client_context=f"Cliente {account_id} con comportamiento resumido en {behavior.get('segment', 'n/d')}.",
                product_context=f"Producto {product_id} evaluado con foco en rentabilidad y elasticidad.",
                zone_context=f"Zona {zone} / Provincia {province}.",
                recommended_action="Mantener recomendación y validar cierre con el equipo comercial.",
                key_arguments=[
                    "La recomendación respeta pisos y techos de la política.",
                    "El segmento y el riesgo condicionan el nivel de descuento.",
                    "La estacionalidad y el stock también influyen en el precio final.",
                ],
                risks=[
                    "Dependencia de datos históricos incompletos.",
                    "Posible sensibilidad del cliente ante cambios bruscos.",
                ],
                tone="balanced",
                confidence=0.55,
            )