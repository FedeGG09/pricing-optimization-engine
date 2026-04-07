# Industrial Pricing Engine Upgrade

Arquitectura objetivo para pricing dinámico, explicabilidad y agentes IA.

## Qué agrega este módulo

- Motor híbrido: modelo ML + reglas de negocio
- Features de pricing, fidelización, pago, provincia, stock y elasticidad
- Explicación por factor y narrativa LLM vía Hugging Face
- Detección de anomalías
- Simulación de escenarios
- API FastAPI lista para local y producción

## Endpoints

- `POST /api/v1/pricing/recommend`
- `POST /api/v1/pricing/simulate`
- `POST /api/v1/pricing/explain`
- `POST /api/v1/pricing/train`

## Flujo

`datos -> feature engineering -> modelo -> reglas -> precio final -> explicación`

## Entrenamiento

```bash
python scripts/train_pricing_model.py --data-dir ../industrial_revenue_growth_engine/out --artifacts-dir ./artifacts
```

## Levantar local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker

```bash
docker compose up --build
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned


.\.venv\Scripts\Activate.ps1