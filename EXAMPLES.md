# Ejemplo de request

```json
{
  "product_id": "equipos_movimiento-de-tierra_camiones-articulados_001",
  "account_id": "ACC00001",
  "province": "Buenos Aires",
  "sector": "Construcción",
  "customer_segment": "vip",
  "current_price": 192870.94,
  "list_price": 205000,
  "base_price": 189508.87,
  "stock_available": 4,
  "seasonality_index": 0.62,
  "price_sensitivity": 0.28,
  "payment_behavior_score": 0.82,
  "loyalty_score": 0.91,
  "overdue_days": 0,
  "discount_history_avg": 0.05,
  "explain": true
}
```

# Ejemplo de response

```json
{
  "final_price": 196500.0,
  "discount_pct": 0.041,
  "suggested_action": "increase",
  "risk_flags": ["at_floor"],
  "factor_scores": {"seasonality": 81.0, "loyalty": 91.0}
}
```
