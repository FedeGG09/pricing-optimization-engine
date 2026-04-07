# Arquitectura objetivo

## Hallazgos del ZIP actual

- Existe una base correcta de plataforma: FastAPI, JWT, SQL Server, Chroma, observabilidad y una capa de agentes.
- El pricing actual se apoya en un ETL agregado por producto con una heurística simple; eso sirve como demo, pero no como motor productivo.
- La orquestación de agentes está acoplada a instancias globales y no separa claramente recomendación, simulación, explicación y auditoría.
- La carga del LLM local es frágil para despliegues CPU-only y no tiene fallback robusto.
- Falta un contrato explícito de features de pricing con variables comerciales, financieras y operativas.
- Faltan endpoints dedicados al caso de uso objetivo: `/pricing/recommend`, `/pricing/simulate`, `/pricing/explain`.

## Arquitectura propuesta

`datos -> feature engineering -> modelo ML -> motor de reglas -> precio final -> explicación LLM -> auditoría`

### Capas

1. `pricing_engine/feature_builder.py`
   - Une transactions, products, accounts y leads.
   - Resuelve defaults cuando faltan datos.
   - Construye variables estacionales, de fidelización, pago, ubicación, stock y elasticidad.

2. `pricing_engine/model_manager.py`
   - Entrena y persiste un modelo de `price_ratio`.
   - Guarda metadata, métricas e importancias.
   - Incluye detector de anomalías.

3. `pricing_engine/rules.py`
   - Aplica floors/ceilings.
   - Ajusta por stock, margen, pago, provincia y urgencia.
   - Devuelve trace y flags de riesgo.

4. `agents/hf_client.py`
   - Envoltorio para Hugging Face Inference API o endpoint compatible.
   - Fallback local si el servicio no está disponible.

5. `agents/pricing_agent.py`
   - Explicación.
   - Estrategia comercial.
   - Anomalías.
   - Simulación narrativa.

6. `pricing_engine/service.py`
   - Caso de uso productivo.
   - Ensambla ML, reglas y agentes.
   - Expone recommend / simulate / explain / train.

7. `api/routes.py`
   - Contrato REST estable para frontend y otros consumidores.

## Mejoras críticas adicionales

- Separar entrenamiento offline del runtime online.
- Versionar artifacts del modelo.
- Registrar feedback de usuario y outcomes reales para reentrenamiento.
- Añadir evaluación por segmento y por provincia.
- Incorporar bandit/RL después de estabilizar el baseline supervisado.
- Persistir decisiones de pricing en audit log y fact table.
