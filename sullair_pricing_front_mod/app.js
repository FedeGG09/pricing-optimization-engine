const API_BASE = "https://pricing-optimization-engine.onrender.com";
const LLM_TOKEN_KEY = "pricing_engine_llm_token";

const el = (id) => document.getElementById(id);

const state = {
  catalog: null,
  metrics: null,
  recommendation: null,
  agentBundle: null,
};

let llmToken = sessionStorage.getItem(LLM_TOKEN_KEY) || "";

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("es-AR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("es-AR", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function toText(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string" || typeof value === "number") return String(value);

  if (Array.isArray(value)) {
    return value.map(toText).filter(Boolean).join(", ");
  }

  if (typeof value === "object") {
    return (
      value.name ??
      value.title ??
      value.label ??
      value.zone ??
      value.region ??
      value.province ??
      value.account_name ??
      value.product_name ??
      value.customer_name ??
      value.client_name ??
      value.description ??
      value.id ??
      JSON.stringify(value)
    );
  }

  return String(value);
}

function normalizeCatalogItems(items = []) {
  return items.map((item) => {
    const id =
      item.id ??
      item.account_id ??
      item.product_id ??
      item.code ??
      item.zone_id ??
      item.region_id ??
      item.province_id ??
      item.customer_id;

    const name =
      toText(
        item.name ??
          item.account_name ??
          item.product_name ??
          item.customer_name ??
          item.client_name ??
          item.province ??
          item.zone ??
          item.region ??
          item.title ??
          item.label ??
          item.description
      ) || String(id ?? "Sin nombre");

    return {
      id: id ?? name,
      name,
      raw: item,
    };
  });
}

function setText(id, value) {
  const node = el(id);
  if (node) node.textContent = value;
}

function showError(message) {
  const box = el("errorBox");
  if (!box) return;
  box.textContent = message;
  box.classList.remove("hidden");
}

function clearError() {
  const box = el("errorBox");
  if (!box) return;
  box.textContent = "";
  box.classList.add("hidden");
}

async function fetchJSON(url, options = {}, retries = 2, timeoutMs = 12000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    const text = await res.text();

    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }

    if (!res.ok) {
      const msg = data?.detail || data?.message || data?.raw || `HTTP ${res.status}`;
      throw new Error(msg);
    }

    return data;
  } catch (err) {
    if (retries > 0) {
      await new Promise((r) => setTimeout(r, 700));
      return fetchJSON(url, options, retries - 1, timeoutMs);
    }
    throw new Error(`No se pudo conectar con ${url}. ${err.message}`);
  } finally {
    clearTimeout(timer);
  }
}

function getAiHeaders() {
  return llmToken ? { Authorization: `Bearer ${llmToken}` } : {};
}

function setAiStatus(connected) {
  const badge = el("aiStatusBadge");
  const connectBtn = el("connectAiBtn");
  const disconnectBtn = el("disconnectAiBtn");
  const passwordInput = el("llmPassword");

  if (badge) {
    badge.textContent = connected ? "IA conectada" : "IA desconectada";
    badge.classList.toggle("chip--active", connected);
  }

  if (connectBtn) {
    connectBtn.textContent = connected ? "Reconectar IA" : "Conectar IA";
  }

  if (disconnectBtn) {
    disconnectBtn.disabled = !connected;
  }

  if (passwordInput && connected) {
    passwordInput.value = "";
  }
}

function isAiConnected() {
  return Boolean(llmToken);
}

function persistAiToken(token) {
  llmToken = token || "";
  if (llmToken) {
    sessionStorage.setItem(LLM_TOKEN_KEY, llmToken);
  } else {
    sessionStorage.removeItem(LLM_TOKEN_KEY);
  }
  setAiStatus(Boolean(llmToken));
  updateActionButtonsState();
}

async function loginAi(password) {
  return fetchJSON(`${API_BASE}/auth/llm-login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
}

async function connectAi() {
  clearError();

  const passwordInput = el("llmPassword");
  const password = passwordInput?.value?.trim();

  if (!password) {
    showError("Ingresá la contraseña para conectar la IA.");
    return;
  }

  const connectBtn = el("connectAiBtn");
  const previousLabel = connectBtn?.textContent;

  try {
    if (connectBtn) {
      connectBtn.disabled = true;
      connectBtn.textContent = "Conectando...";
    }

    const data = await loginAi(password);
    persistAiToken(data.access_token || "");
    if (!llmToken) {
      throw new Error("No se recibió token de acceso.");
    }
    if (passwordInput) passwordInput.value = "";
  } catch (error) {
    persistAiToken("");
    showError(error.message || "No se pudo conectar la IA");
  } finally {
    if (connectBtn) {
      connectBtn.disabled = false;
      connectBtn.textContent = previousLabel || "Conectar IA";
    }
    updateActionButtonsState();
  }
}

function disconnectAi() {
  persistAiToken("");
  const passwordInput = el("llmPassword");
  if (passwordInput) passwordInput.value = "";
  clearError();
}

function requireAiToken() {
  if (!llmToken) {
    throw new Error("Conectá la IA primero.");
  }
}

function aiFetchOptions(extra = {}) {
  return {
    ...extra,
    headers: {
      ...(extra.headers || {}),
      ...getAiHeaders(),
    },
  };
}

function populateSelect(id, items, placeholder, allowAll = false) {
  const select = el(id);
  if (!select) return;

  select.innerHTML = "";

  const emptyOption = document.createElement("option");
  emptyOption.value = "";
  emptyOption.textContent = placeholder;
  select.appendChild(emptyOption);

  if (allowAll) {
    const allOption = document.createElement("option");
    allOption.value = "__all__";
    allOption.textContent = "Todos";
    select.appendChild(allOption);
  }

  items.forEach((item) => {
    const option = document.createElement("option");
    const optionValue = item.id !== undefined && item.id !== null ? String(item.id) : item.name;
    option.value = optionValue;
    option.textContent = `${toText(item.name)}${item.id !== undefined && item.id !== null ? ` (${item.id})` : ""}`;
    select.appendChild(option);
  });
}

function updateActionButtonsState() {
  const accountNode = el("accountSelect");
  const productNode = el("productSelect");

  const hasAccount = !!accountNode?.value?.trim();
  const hasProduct = !!productNode?.value?.trim();
  const aiReady = isAiConnected();

  const pricingEnabled = hasAccount && hasProduct;

  const recommendBtn = el("recommendBtn");
  if (recommendBtn) recommendBtn.disabled = !pricingEnabled;

  const btnExplainAi = el("btnExplainAi");
  if (btnExplainAi) btnExplainAi.disabled = !(pricingEnabled && aiReady);

  const btnAnomalies = el("btnAnomalies");
  if (btnAnomalies) btnAnomalies.disabled = !(pricingEnabled && aiReady);

  const btnStrategy = el("btnStrategy");
  if (btnStrategy) btnStrategy.disabled = !(pricingEnabled && aiReady);

  const btnWhatIf = el("btnWhatIf");
  if (btnWhatIf) btnWhatIf.disabled = !(pricingEnabled && aiReady);

  const narratorBtn = el("narratorChatBtn");
  if (narratorBtn) narratorBtn.disabled = !aiReady;
}

async function loadCatalog() {
  let bootstrap = null;

  try {
    bootstrap = await fetchJSON(`${API_BASE}/reference/catalog`);
  } catch {
    bootstrap = null;
  }

  let accounts = [];
  let products = [];
  let provinces = [];
  let zones = [];

  if (bootstrap) {
    accounts = normalizeCatalogItems(bootstrap.accounts || []);
    products = normalizeCatalogItems(bootstrap.products || []);
    provinces = normalizeCatalogItems(bootstrap.provinces || []);
    zones = normalizeCatalogItems(bootstrap.zones || []);
  } else {
    const accountsRes = await fetchJSON(`${API_BASE}/reference/accounts`);
    const productsRes = await fetchJSON(`${API_BASE}/reference/products`);
    const provincesRes = await fetchJSON(`${API_BASE}/reference/provinces`);
    const zonesRes = await fetchJSON(`${API_BASE}/reference/zones`);

    accounts = normalizeCatalogItems(accountsRes.rows || []);
    products = normalizeCatalogItems(productsRes.rows || []);
    provinces = normalizeCatalogItems(provincesRes.rows || []);
    zones = normalizeCatalogItems(zonesRes.rows || []);
  }

  if (!accounts.length || !products.length) {
    throw new Error("No se pudieron cargar clientes o productos desde la API.");
  }

  state.catalog = { accounts, products, provinces, zones };

  populateSelect("accountSelect", accounts, "Seleccionar cliente");
  populateSelect("productSelect", products, "Seleccionar producto");
  populateSelect("provinceSelect", provinces, "Seleccionar provincia", true);
  populateSelect("zoneSelect", zones, "Seleccionar zona / región", true);
}

async function loadMetrics() {
  let data = null;

  try {
    data = await fetchJSON(`${API_BASE}/pricing/model-metrics`);
  } catch {
    const bootstrap = await fetchJSON(`${API_BASE}/reference/catalog`);
    data = bootstrap.metrics || bootstrap.model_metrics || { status: "pending" };
  }

  state.metrics = data;

  const metrics = data.metrics || data;

  setText("metricMae", formatNumber(metrics.mae, 4));
  setText("metricRmse", formatNumber(metrics.rmse, 4));
  setText("metricR2", formatNumber(metrics.r2, 4));
  setText("metricMape", `${formatNumber(metrics.mape_pct, 2)}%`);

  setText("summaryMae", formatNumber(metrics.mae, 4));
  setText("summaryRmse", formatNumber(metrics.rmse, 4));
  setText("trainingRows", formatNumber(data.training_rows ?? metrics.training_rows, 0));
  setText("validationRows", formatNumber(data.validation_rows ?? metrics.validation_rows, 0));
  setText("featureCount", formatNumber(data.feature_count ?? metrics.feature_count, 0));
  setText("modelState", data.status === "pending" ? "Pendiente" : "OK");

  if (data.trained_at_utc) {
    setText("trainedAtBadge", `Entrenado: ${new Date(data.trained_at_utc).toLocaleDateString("es-AR")}`);
  } else if (metrics.trained_at_utc) {
    setText("trainedAtBadge", `Entrenado: ${new Date(metrics.trained_at_utc).toLocaleDateString("es-AR")}`);
  } else {
    setText("trainedAtBadge", "Entrenado: —");
  }

  setText("modelStatusPill", data.status === "pending" ? "Métricas pendientes" : "Métricas disponibles");
}

async function loadAuditLatest() {
  try {
    const data = await fetchJSON(`${API_BASE}/pricing/audit/latest?limit=10`);
    renderAuditPanel(data.rows || []);
  } catch {
    renderAuditPanel([]);
  }
}

function renderAuditPanel(rows = []) {
  const container = document.getElementById("auditLatestRows");
  if (!container) return;

  container.innerHTML = "";

  if (!rows.length) {
    container.innerHTML = `
      <div class="inline-item">
        <strong>Sin auditoría</strong>
        <span>No hay decisiones registradas todavía.</span>
      </div>
    `;
    return;
  }

  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "inline-item";
    item.innerHTML = `
      <strong>#${row.id} · ${row.account_id} / ${row.product_id}</strong>
      <span>${row.created_at_utc} · ${row.context_source} · ${row.status_code}</span>
    `;
    container.appendChild(item);
  });
}

function buildPricingPayload() {
  const accountValue = el("accountSelect")?.value?.trim() || "";
  const productValue = el("productSelect")?.value?.trim() || "";

  if (!accountValue || !productValue) {
    throw new Error("Seleccioná un cliente y un producto válidos antes de calcular.");
  }

  const monthValue = Number(el("monthInput")?.value);
  const qtyValue = Number(el("qtyInput")?.value);

  return {
    account_id: accountValue,
    product_id: productValue,
    province: el("provinceSelect")?.value && el("provinceSelect").value !== "__all__" ? el("provinceSelect").value : undefined,
    zone: el("zoneSelect")?.value && el("zoneSelect").value !== "__all__" ? el("zoneSelect").value : undefined,
    month: Number.isFinite(monthValue) ? monthValue : undefined,
    quantity: Number.isFinite(qtyValue) ? qtyValue : 1,
    list_price_usd: el("listPriceInput")?.value ? Number(el("listPriceInput").value) : undefined,
    base_price_usd: el("basePriceInput")?.value ? Number(el("basePriceInput").value) : undefined,
  };
}

function buildAgentRequest() {
  const payload = buildPricingPayload();

  return {
    account_id: payload.account_id,
    product_id: payload.product_id,
    overrides: {
      province: payload.province,
      zone: payload.zone,
      month: payload.month,
      quantity: payload.quantity,
      list_price_usd: payload.list_price_usd,
      base_price_usd: payload.base_price_usd,
    },
  };
}

function renderFactorScores(scores = {}) {
  const container = el("factorScores");
  if (!container) return;

  container.innerHTML = "";

  const entries = Object.entries(scores);
  if (!entries.length) {
    container.innerHTML = `<div class="factor-item"><div class="factor-item__top"><span>Sin scores</span><span>—</span></div></div>`;
    return;
  }

  entries.forEach(([key, value]) => {
    const clamped = Math.max(0, Math.min(100, Number(value) || 0));
    const item = document.createElement("div");
    item.className = "factor-item";
    item.innerHTML = `
      <div class="factor-item__top">
        <span>${key.replace(/_/g, " ")}</span>
        <span>${formatNumber(value, 0)}</span>
      </div>
      <div class="factor-track"><div style="width:${clamped}%"></div></div>
    `;
    container.appendChild(item);
  });
}

function renderChart(reco) {
  const chart = el("chartBars");
  if (!chart) return;

  chart.innerHTML = "";

  const points = [
    { label: "Floor", value: reco?.price?.floor ?? 0 },
    { label: "Model", value: reco?.price?.model_price ?? 0 },
    { label: "Final", value: reco?.price?.recommended ?? 0 },
    { label: "Ceiling", value: reco?.price?.ceiling ?? 0 },
  ];

  const maxValue = Math.max(...points.map((p) => Number(p.value) || 0), 1);

  points.forEach((point) => {
    const bar = document.createElement("div");
    bar.className = "chart-bar";

    const height = ((Number(point.value) || 0) / maxValue) * 100;
    bar.innerHTML = `
      <div class="chart-bar__money">${formatMoney(point.value)}</div>
      <div class="chart-bar__value" style="height:${Math.max(height, 8)}%"></div>
      <div class="chart-bar__label">${point.label}</div>
    `;
    chart.appendChild(bar);
  });
}

function renderRecommendation(data) {
  state.recommendation = data;

  const recommendationEmpty = el("recommendationEmpty");
  const recommendationPanel = el("recommendationPanel");

  if (recommendationEmpty) recommendationEmpty.classList.add("hidden");
  if (recommendationPanel) recommendationPanel.classList.remove("hidden");

  setText("priceRecommended", formatMoney(data.price?.recommended));
  setText(
    "discountRecommended",
    data.discount?.recommended_pct !== undefined && data.discount?.recommended_pct !== null
      ? `${formatNumber(data.discount.recommended_pct * 100, 2)}%`
      : "—"
  );
  setText(
    "marginExpected",
    data.margin?.expected_pct !== undefined && data.margin?.expected_pct !== null
      ? `${formatNumber(data.margin.expected_pct * 100, 2)}%`
      : "—"
  );
  setText("fallbackUsed", data.fallback_used ? "Sí" : "No");
  setText("recommendationReason", data.reason || "Sin explicación disponible");
  setText("contextSource", data.context_source || "—");
  setText("modelVersion", data.model_version || "—");

  setText("fallbackBadge", `Fallback: ${data.fallback_used ? "Sí" : "No"}`);
  renderFactorScores(data.factor_scores || {});
  renderChart(data);
}

function renderNarrative(bundle) {
  const narrative = bundle?.narrative || {};
  const market = bundle?.market || {};
  const behavior = bundle?.behavior || {};
  const guardrail = bundle?.guardrail || {};

  setText("agentNarrativeTone", narrative.tone || "—");
  setText("agentExecutiveSummary", narrative.executive_summary || "Todavía no se generó explicación IA.");
  setText("agentClientContext", narrative.client_context || behavior.summary || "—");
  setText("agentProductContext", narrative.product_context || "—");
  setText("agentZoneContext", narrative.zone_context || market.summary || "—");
  setText("agentRecommendedAction", narrative.recommended_action || "—");

  const keyArgs = el("agentKeyArguments");
  if (keyArgs) {
    keyArgs.innerHTML = "";
    (narrative.key_arguments || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      keyArgs.appendChild(li);
    });
    if (!keyArgs.children.length) {
      const li = document.createElement("li");
      li.textContent = "Sin argumentos generados aún.";
      keyArgs.appendChild(li);
    }
  }

  const risks = el("agentRisks");
  if (risks) {
    risks.innerHTML = "";
    (narrative.risks || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      risks.appendChild(li);
    });
    if (!risks.children.length) {
      const li = document.createElement("li");
      li.textContent = "Sin riesgos reportados.";
      risks.appendChild(li);
    }
  }

  setText("agentMarketSummary", market.summary || "—");
  setText("agentBehaviorSummary", behavior.summary || "—");
  setText(
    "agentGuardrailSummary",
    guardrail.allowed === false
      ? `Bloqueado: ${guardrail.blocked_reason || "guardrail no disponible"}`
      : `Permitido · descuento máx. ${(Number(guardrail.max_discount_pct || 0) * 100).toFixed(1)}%`
  );
  setText("agentContextState", bundle?.extra?.context_source || "—");
}

function renderAnomalies(anomalies = []) {
  const container = el("agentAnomalies");
  if (!container) return;

  container.innerHTML = "";

  if (!anomalies.length) {
    container.innerHTML = `
      <div class="inline-item">
        <strong>Sin anomalías</strong>
        <span>El agente no encontró alertas relevantes para este caso.</span>
      </div>
    `;
    return;
  }

  anomalies.forEach((item) => {
    const card = document.createElement("div");
    card.className = "inline-item";
    card.innerHTML = `
      <strong>[${toText(item.severity || "low").toUpperCase()}] ${toText(item.title || item.code)}</strong>
      <span>${toText(item.detail || "")}</span>
      <span class="inline-item__action">${toText(item.recommended_action || "")}</span>
    `;
    container.appendChild(card);
  });
}

function renderStrategy(strategy = {}) {
  setText("agentStrategyTitle", strategy.title || "—");
  setText("agentStrategySummary", strategy.executive_summary || "—");

  const plan = el("agentActionPlan");
  if (plan) {
    plan.innerHTML = "";
    (strategy.action_plan || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      plan.appendChild(li);
    });
    if (!plan.children.length) {
      const li = document.createElement("li");
      li.textContent = "Sin plan de acción.";
      plan.appendChild(li);
    }
  }

  const risks = el("agentStrategyRisks");
  if (risks) {
    risks.innerHTML = "";
    (strategy.risks || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      risks.appendChild(li);
    });
    if (!risks.children.length) {
      const li = document.createElement("li");
      li.textContent = "Sin riesgos reportados.";
      risks.appendChild(li);
    }
  }
}

function renderWhatIfScenarios(scenarios = []) {
  const tbody = el("scenarioTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";

  if (!scenarios.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-row">Todavía no se ejecutó simulación.</td></tr>`;
    return;
  }

  scenarios.forEach((s) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatNumber(s.candidate_multiplier, 2)}x</td>
      <td>${formatMoney(s.candidate_price_usd)}</td>
      <td>${formatNumber((Number(s.expected_margin_pct) || 0) * 100, 2)}%</td>
      <td>${formatNumber(s.expected_revenue_score, 2)}</td>
      <td>${s.feasible ? "OK" : "Revisar"}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function requestRecommendation() {
  clearError();

  try {
    const payload = buildPricingPayload();

    const btn = el("recommendBtn");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Calculando...";
    }

    const data = await fetchJSON(`${API_BASE}/pricing/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    renderRecommendation(data);
  } catch (error) {
    showError(error.message || "Error generando recomendación");
  } finally {
    const btn = el("recommendBtn");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Calcular recomendación";
    }
    updateActionButtonsState();
  }
}

async function requestExplainAI() {
  clearError();

  try {
    const payload = buildAgentRequest();

    const btn = el("btnExplainAi");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Generando...";
    }

    requireAiToken();
    const data = await fetchJSON(
      `${API_BASE}/agents/explain-pricing`,
      aiFetchOptions({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    );

    state.agentBundle = data.bundle || null;
    renderNarrative(data.bundle || {});
    renderAnomalies(data.bundle?.anomalies || []);
    if (data.bundle?.strategy) renderStrategy(data.bundle.strategy);
  } catch (error) {
    showError(error.message || "Error generando explicación IA");
  } finally {
    const btn = el("btnExplainAi");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Generar explicación IA";
    }
    updateActionButtonsState();
  }
}

async function requestAnomalies() {
  clearError();

  try {
    const payload = buildAgentRequest();

    const btn = el("btnAnomalies");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Analizando...";
    }

    requireAiToken();
    const data = await fetchJSON(
      `${API_BASE}/agents/detect-anomalies`,
      aiFetchOptions({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    );

    renderAnomalies(data.anomalies || []);
  } catch (error) {
    showError(error.message || "Error detectando anomalías");
  } finally {
    const btn = el("btnAnomalies");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Detectar anomalías";
    }
    updateActionButtonsState();
  }
}

async function requestStrategy() {
  clearError();

  try {
    const payload = buildAgentRequest();

    const btn = el("btnStrategy");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Pensando...";
    }

    requireAiToken();
    const data = await fetchJSON(
      `${API_BASE}/agents/recommend-strategy`,
      aiFetchOptions({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    );

    renderStrategy(data.strategy || {});
  } catch (error) {
    showError(error.message || "Error generando estrategia");
  } finally {
    const btn = el("btnStrategy");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Recomendar estrategia";
    }
    updateActionButtonsState();
  }
}

async function requestWhatIf() {
  clearError();

  try {
    const payload = buildAgentRequest();

    const btn = el("btnWhatIf");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Simulando...";
    }

    requireAiToken();
    const data = await fetchJSON(
      `${API_BASE}/agents/what-if-simulation`,
      aiFetchOptions({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...payload,
          candidate_multipliers: [0.85, 0.9, 0.95, 1.0, 1.03, 1.05, 1.08, 1.1],
        }),
      })
    );

    renderWhatIfScenarios(data.scenarios || []);
  } catch (error) {
    showError(error.message || "Error simulando escenarios");
  } finally {
    const btn = el("btnWhatIf");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Simular escenarios";
    }
    updateActionButtonsState();
  }
}

async function sendNarratorChat() {
  clearError();

  const questionEl = document.getElementById("narratorQuestion");
  const answerEl = document.getElementById("narratorAnswer");
  const decisionIdEl = document.getElementById("narratorDecisionId");

  const question = questionEl?.value?.trim();
  if (!question) {
    showError("Escribí una pregunta para el Narrator.");
    return;
  }

  const decisionId = decisionIdEl?.value ? Number(decisionIdEl.value) : undefined;

  try {
    const payload = {
      question,
      decision_id: Number.isFinite(decisionId) ? decisionId : undefined,
      account_id: el("accountSelect")?.value || undefined,
      product_id: el("productSelect")?.value || undefined,
      overrides: {
        province: el("provinceSelect")?.value && el("provinceSelect").value !== "__all__" ? el("provinceSelect").value : undefined,
        zone: el("zoneSelect")?.value && el("zoneSelect").value !== "__all__" ? el("zoneSelect").value : undefined,
      },
    };

    requireAiToken();
    const data = await fetchJSON(
      `${API_BASE}/agents/narrator-chat`,
      aiFetchOptions({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    );

    if (answerEl) answerEl.textContent = data.answer || "Sin respuesta";
  } catch (error) {
    showError(error.message || "Error consultando al Narrator");
  }
}

function wireTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const contents = {
    roadmap: el("tab-roadmap"),
    llm: el("tab-llm"),
    audit: el("tab-audit"),
  };

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("tab-btn--active"));
      btn.classList.add("tab-btn--active");

      Object.values(contents).forEach((c) => c.classList.remove("tab-content--active"));
      contents[btn.dataset.tab].classList.add("tab-content--active");
    });
  });
}

function setDefaults() {
  const now = new Date();
  const monthInput = el("monthInput");
  const qtyInput = el("qtyInput");
  if (monthInput) monthInput.value = String(now.getMonth() + 1);
  if (qtyInput) qtyInput.value = "1";
}

async function refreshAll() {
  clearError();
  setText("modelStatusPill", "Cargando métricas...");
  setText("fallbackBadge", "Fallback: —");

  try {
    await loadCatalog();
    await loadMetrics();
    await loadAuditLatest();
    updateActionButtonsState();
  } catch (e) {
    showError(e.message || "No se pudo cargar el dashboard");
  }
}

function bindEvents() {
  el("recommendBtn")?.addEventListener("click", requestRecommendation);
  el("btnExplainAi")?.addEventListener("click", requestExplainAI);
  el("btnAnomalies")?.addEventListener("click", requestAnomalies);
  el("btnStrategy")?.addEventListener("click", requestStrategy);
  el("btnWhatIf")?.addEventListener("click", requestWhatIf);
  el("refreshBtn")?.addEventListener("click", refreshAll);
  document.getElementById("narratorChatBtn")?.addEventListener("click", sendNarratorChat);
  el("connectAiBtn")?.addEventListener("click", connectAi);
  el("disconnectAiBtn")?.addEventListener("click", disconnectAi);

  el("llmPassword")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      connectAi();
    }
  });

  ["accountSelect", "productSelect"].forEach((id) => {
    el(id)?.addEventListener("change", updateActionButtonsState);
  });
}

async function init() {
  setDefaults();
  wireTabs();
  bindEvents();
  setAiStatus(isAiConnected());
  updateActionButtonsState();
  await refreshAll();
}

init();
