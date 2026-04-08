from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.orchestrator import PricingAgentOrchestrator
from app.api.shared_runtime import (
    build_enriched_payload,
    read_audit_decision,
    safe_json_load_text,
)
from app.pricing_engine.dynamic_pricing_engine import get_engine
from app.services.hf_client import get_hf_client

logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException, Depends
from app.api.deps import get_current_user

router = APIRouter(
    prefix="/agents",
    tags=["agents"],
    dependencies=[Depends(get_current_user)],
)

orchestrator = PricingAgentOrchestrator()
pricing_engine = get_engine()


class AgentRequest(BaseModel):
    account_id: str
    product_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    pricing_snapshot: Optional[dict[str, Any]] = None


class WhatIfRequest(AgentRequest):
    candidate_multipliers: list[float] = Field(
        default_factory=lambda: [0.85, 0.90, 0.95, 1.00, 1.03, 1.05, 1.08, 1.10]
    )


class NarratorChatRequest(BaseModel):
    decision_id: Optional[int] = None
    account_id: Optional[str] = None
    product_id: Optional[str] = None
    question: str
    overrides: dict[str, Any] = Field(default_factory=dict)


def _ensure_snapshot(req: AgentRequest) -> dict[str, Any]:
    if req.pricing_snapshot:
        return req.pricing_snapshot

    ctx = build_enriched_payload(
        account_id=req.account_id,
        product_id=req.product_id,
        overrides=req.overrides,
    )
    recommendation = pricing_engine.recommend(ctx["payload"])
    if hasattr(recommendation, "model_dump"):
        return recommendation.model_dump()
    if hasattr(recommendation, "__dict__"):
        return recommendation.__dict__
    return dict(recommendation)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


def _compact_narrator_context(
    *,
    request_json: dict[str, Any],
    enriched_payload: dict[str, Any],
    response_json: dict[str, Any],
    decision_id: Optional[int],
) -> dict[str, Any]:
    """
    Reduce el contexto a lo que realmente necesita el LLM para narrar:
    cliente, producto, zona, señal comercial, recomendación y riesgos.
    """

    core = {
        "decision_id": decision_id,
        "account_id": request_json.get("account_id"),
        "product_id": request_json.get("product_id"),
        "province": enriched_payload.get("province"),
        "zone": enriched_payload.get("zone"),
        "segment_rule_based": enriched_payload.get("segment_rule_based"),
        "segment_kmeans_label": enriched_payload.get("segment_kmeans_label"),
        "risk_score_0_100": enriched_payload.get("risk_score_0_100"),
        "rfm_score": enriched_payload.get("rfm_score"),
        "compliance_rate": enriched_payload.get("compliance_rate"),
        "price_sensitivity": enriched_payload.get("price_sensitivity"),
        "avg_discount_pct": enriched_payload.get("avg_discount_pct"),
        "avg_margin_pct": enriched_payload.get("avg_margin_pct"),
        "price_floor_usd": enriched_payload.get("price_floor_usd"),
        "price_ceiling_usd": enriched_payload.get("price_ceiling_usd"),
        "negotiated_price_usd": enriched_payload.get("negotiated_price_usd"),
        "list_price_usd": enriched_payload.get("list_price_usd"),
        "base_price_usd": enriched_payload.get("base_price_usd"),
        "stock_units": enriched_payload.get("stock_units"),
        "seasonality_index": enriched_payload.get("seasonality_index"),
        "recency_days": enriched_payload.get("recency_days"),
        "frequency_90d": enriched_payload.get("frequency_90d"),
        "funnel_stage": enriched_payload.get("funnel_stage"),
        "next_best_action": enriched_payload.get("next_best_action"),
    }

    recommendation = {
        "status": response_json.get("status"),
        "decision_id": response_json.get("decision_id"),
        "context_source": response_json.get("context_source"),
        "fallback_used": response_json.get("fallback_used"),
        "price": response_json.get("price", {}),
        "discount": response_json.get("discount", {}),
        "margin": response_json.get("margin", {}),
        "reason": response_json.get("reason"),
        "factor_scores": response_json.get("factor_scores", {}),
        "rule_adjustments": response_json.get("rule_adjustments", {}),
        "model_version": response_json.get("model_version"),
    }

    compact = {
        "core_context": core,
        "pricing_recommendation": recommendation,
    }

    # limpieza final de nulos para achicar prompt
    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if v is not None and v != {} and v != []}
        if isinstance(obj, list):
            return [_clean(v) for v in obj if v is not None]
        return obj

    return _clean(compact)


@router.post("/explain-pricing")
def explain_pricing(req: AgentRequest) -> dict[str, Any]:
    try:
        ctx = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=req.overrides,
        )
        snapshot = _ensure_snapshot(req)
        bundle = orchestrator.build_bundle(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=req.overrides,
            pricing_snapshot=snapshot,
        )
        return {
            "status": "ok",
            "context_source": ctx["context_source"],
            "bundle": bundle,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generando explicación IA: {exc}") from exc


@router.post("/detect-anomalies")
def detect_anomalies(req: AgentRequest) -> dict[str, Any]:
    try:
        ctx = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=req.overrides,
        )
        snapshot = _ensure_snapshot(req)
        guardrail = orchestrator.discount_logic.evaluate(ctx, recommendation=snapshot)
        anomalies = orchestrator.detect_anomalies(
            context=ctx,
            recommendation=snapshot,
            guardrail=guardrail.model_dump() if hasattr(guardrail, "model_dump") else guardrail.__dict__,
        )
        return {
            "status": "ok",
            "context_source": ctx["context_source"],
            "anomalies": [a.model_dump() if hasattr(a, "model_dump") else a.__dict__ for a in anomalies],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error detectando anomalías: {exc}") from exc


@router.post("/recommend-strategy")
def recommend_strategy(req: AgentRequest) -> dict[str, Any]:
    try:
        ctx = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=req.overrides,
        )
        snapshot = _ensure_snapshot(req)
        guardrail = orchestrator.discount_logic.evaluate(ctx, recommendation=snapshot)
        strategy = orchestrator.recommend_strategy(
            context=ctx,
            recommendation=snapshot,
            guardrail=guardrail,
        )
        return {
            "status": "ok",
            "context_source": ctx["context_source"],
            "strategy": strategy.model_dump() if hasattr(strategy, "model_dump") else strategy.__dict__,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generando estrategia: {exc}") from exc


@router.post("/what-if-simulation")
def what_if_simulation(req: WhatIfRequest) -> dict[str, Any]:
    try:
        ctx = build_enriched_payload(
            account_id=req.account_id,
            product_id=req.product_id,
            overrides=req.overrides,
        )
        snapshot = _ensure_snapshot(req)
        scenarios = orchestrator.what_if_simulation(
            context=ctx,
            recommendation=snapshot,
            candidate_multipliers=req.candidate_multipliers,
        )
        return {
            "status": "ok",
            "context_source": ctx["context_source"],
            "scenarios": scenarios,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error simulando escenarios: {exc}") from exc


@router.post("/narrator-chat")
def narrator_chat(req: NarratorChatRequest) -> dict[str, Any]:
    hf = get_hf_client()

    try:
        audit_row = None
        if req.decision_id is not None:
            audit_row = read_audit_decision(req.decision_id)

        if audit_row:
            response_json = safe_json_load_text(audit_row.get("response_json")) or {}
            enriched_payload = safe_json_load_text(audit_row.get("enriched_payload_json")) or {}
            request_json = safe_json_load_text(audit_row.get("request_json")) or {}
        else:
            if not req.account_id or not req.product_id:
                raise HTTPException(
                    status_code=400,
                    detail="Para chatear sin decision_id necesitás account_id y product_id.",
                )

            # clave: cuando no hay auditoría, generamos la recomendación real
            ctx = build_enriched_payload(
                account_id=req.account_id,
                product_id=req.product_id,
                overrides=req.overrides,
            )
            recommendation = pricing_engine.recommend(ctx["payload"])
            response_json = _to_plain_dict(recommendation)
            enriched_payload = ctx["payload"]
            request_json = {
                "account_id": req.account_id,
                "product_id": req.product_id,
                **req.overrides,
            }

        compact_context = _compact_narrator_context(
            request_json=request_json,
            enriched_payload=enriched_payload,
            response_json=response_json,
            decision_id=req.decision_id,
        )

        system_prompt = (
            "Sos un analista comercial senior. "
            "Respondé en español, claro, ejecutivo y breve. "
            "No inventes datos. "
            "Usá únicamente el contexto provisto. "
            "Si el margen está debajo del objetivo, explicalo y proponé acción comercial."
        )

        user_prompt = f"""
Contexto compacto:
{json.dumps(compact_context, ensure_ascii=False, indent=2)}

Pregunta del usuario:
{req.question}

Respondé con esta estructura:
- respuesta directa
- impacto comercial
- riesgo principal
- siguiente acción sugerida

Máximo 120 palabras.
"""

        answer = hf.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.15,
            max_tokens=220,
        )

        if not answer.strip():
            raise RuntimeError("HF devolvió respuesta vacía")

        return {
            "status": "ok",
            "decision_id": req.decision_id,
            "answer": answer,
            "source": "audit" if audit_row else "live_context",
        }

    except Exception as exc:
        logger.exception("Narrator chat failed")

        print("\n=== HF NARRATOR DEBUG ===")
        print("EXC:", repr(exc))
        print("HF_LAST_ERROR:", getattr(hf, "last_error", None))
        print("MODEL:", getattr(hf, "model", None))
        print("BASE_URL:", getattr(hf, "base_url", None))
        print("TOKEN_SET:", bool(getattr(hf, "token", "")))
        print("ACCOUNT:", req.account_id)
        print("PRODUCT:", req.product_id)
        print("DECISION_ID:", req.decision_id)
        print("QUESTION:", req.question)
        print("=========================\n")

        return {
            "status": "ok",
            "decision_id": req.decision_id,
            "answer": (
                "No pude usar el LLM en este momento, pero el contexto fue recuperado correctamente. "
                "La recomendación debe revisarse con foco en margen, riesgo y alineación comercial."
            ),
            "source": "fallback",
            "debug_error": repr(exc),
            "hf_last_error": getattr(hf, "last_error", None),
        }
