/* ═══════════════════════════════════════════════════════════════════════════
   Manufacturing Quality Control Agent — Dashboard v2 JavaScript
   All original API functions preserved. New: multi-page navigation,
   KPI cards, alert list, Analytics charts, Data & Models page.
   ═══════════════════════════════════════════════════════════════════════════ */

const API = "";   // Same origin — FastAPI serves this file

// ── Chart instances ───────────────────────────────────────────────────────────
let trendChart    = null;
let statusChart   = null;
let featureChart  = null;
let chartTemp     = null;
let chartPres     = null;
let chartVib      = null;
let chartSpeed    = null;
let chartRate     = null;

// ── State ─────────────────────────────────────────────────────────────────────
let _datasetRecords = [];  // cached full dataset for analytics charts
let _analyticsReady = false;

// ═══════════════════════════════════════════════════════════════════════════════
// Navigation
// ═══════════════════════════════════════════════════════════════════════════════

function navigateTo(page) {
  // Hide all pages
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

  // Show target page
  const pageEl = document.getElementById("page-" + page);
  if (pageEl) pageEl.classList.add("active");

  // Activate nav item
  const navEl = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (navEl) navEl.classList.add("active");

  // Lazy-render analytics charts only when the page is first visited
  if (page === "analytics" && !_analyticsReady && _datasetRecords.length) {
    renderAnalyticsCharts(_datasetRecords);
    _analyticsReady = true;
  }
}

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility
// ═══════════════════════════════════════════════════════════════════════════════

function showLoading(show) {
  document.getElementById("loadingOverlay").classList.toggle("hidden", !show);
}

async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// API action functions  (all original signatures preserved)
// ═══════════════════════════════════════════════════════════════════════════════

async function trainModels() {
  showLoading(true);
  const trainStatus = document.getElementById("trainStatus");
  trainStatus.textContent = "Training models, please wait…";
  try {
    const data = await apiFetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const acc = data.defect_prediction?.test_accuracy;
    const anomCount = data.anomaly_detection?.anomalies_detected;
    trainStatus.textContent =
      `✓ Training complete — Random Forest accuracy: ${acc !== undefined ? (acc * 100).toFixed(1) + "%" : "N/A"}` +
      (anomCount !== undefined ? ` | Anomalies detected in training set: ${anomCount}` : "");
  } catch (e) {
    trainStatus.textContent = "✗ Error: " + e.message;
  } finally {
    showLoading(false);
  }
}

async function analyzeLatest() {
  showLoading(true);
  try {
    const data = await apiFetch("/api/analyze-latest");
    renderReport(data);
    // Switch to analysis page so results are visible
    navigateTo("analysis");
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    showLoading(false);
  }
}

async function analyzeRow() {
  const idx = parseInt(document.getElementById("rowIndex").value, 10);
  if (isNaN(idx)) { alert("Enter a valid row number."); return; }
  showLoading(true);
  try {
    const data = await apiFetch(`/api/analyze-row/${idx}`);
    renderReport(data);
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    showLoading(false);
  }
}

async function analyzeManual() {
  const body = {
    temperature:     parseFloat(document.getElementById("p_temp").value),
    pressure:        parseFloat(document.getElementById("p_pres").value),
    vibration:       parseFloat(document.getElementById("p_vib").value),
    machine_speed:   parseFloat(document.getElementById("p_speed").value),
    production_rate: parseFloat(document.getElementById("p_rate").value),
  };
  for (const [k, v] of Object.entries(body)) {
    if (isNaN(v)) { alert(`Invalid value for ${k}`); return; }
  }
  showLoading(true);
  try {
    const data = await apiFetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderReport(data);
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    showLoading(false);
  }
}

async function uploadCSV() {
  const fileInput = document.getElementById("csvFile");
  if (!fileInput.files.length) { alert("Select a CSV file first."); return; }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  showLoading(true);
  try {
    const data = await apiFetch("/api/upload-csv", { method: "POST", body: formData });
    renderReport(data);
    navigateTo("analysis");
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    showLoading(false);
  }
}

function handleFileSelect() {
  const fileInput = document.getElementById("csvFile");
  const hint = document.getElementById("selectedFileName");
  if (fileInput.files.length) {
    hint.textContent = "Selected: " + fileInput.files[0].name;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// renderReport — updates all pages from one pipeline response
// ═══════════════════════════════════════════════════════════════════════════════

function statusCls(s) {
  if (s === "Critical") return "c-critical";
  if (s === "Warning")  return "c-warning";
  return "c-normal";
}

function renderReport(data) {
  const { process_data, monitoring_result, quality_result,
          defect_result, optimization_result, ai_explanation } = data;

  const overall  = monitoring_result.overall_status;    // Normal / Warning / Critical
  const risk     = quality_result.quality_risk;         // Low / Medium / High
  const abnormal = monitoring_result.abnormal_parameters || [];
  const recs     = optimization_result.recommendations || [];
  const defectPred = defect_result.defect_predicted;
  const probPct  = (defect_result.defect_probability * 100).toFixed(1);

  // ── 1. Header badge ───────────────────────────────────────────────────────
  const badge = document.getElementById("systemStatus");
  badge.textContent = "● " + overall.toUpperCase();
  badge.className   = "hdr-badge badge-" + overall.toLowerCase();

  // ── 2. Alert banner (overview page) ──────────────────────────────────────
  const banner = document.getElementById("statusBanner");
  const icons  = { Normal: "✅", Warning: "⚠️", Critical: "🚨" };
  banner.className = "alert-banner alert-" + overall.toLowerCase();
  banner.classList.remove("hidden");
  document.getElementById("bannerIcon").textContent  = icons[overall] || "●";
  document.getElementById("bannerTitle").textContent =
    overall + " — Quality Risk: " + risk;
  document.getElementById("bannerSub").textContent   = monitoring_result.summary;

  // ── 3. KPI cards ─────────────────────────────────────────────────────────
  const kpiStatusCls = overall === "Critical" ? "kpi-critical"
                     : overall === "Warning"  ? "kpi-warning"  : "kpi-normal";
  const kpiRiskCls   = risk === "High"   ? "kpi-critical"
                     : risk === "Medium" ? "kpi-warning"  : "kpi-normal";
  const kpiDefectCls = defectPred ? "kpi-critical" : "kpi-normal";

  setKPI("kpi_status_val",  overall,                kpiStatusCls);
  setKPI("kpi_risk_val",    risk,                   kpiRiskCls);
  setKPI("kpi_defect_val",  probPct + "%",           kpiDefectCls);
  setKPI("kpi_alerts_val",  abnormal.length,         abnormal.length ? "kpi-warning" : "kpi-normal");
  setKPI("kpi_recs_val",    recs.length,             recs.length ? "kpi-warning" : "kpi-normal");

  // ── 4. Overview alert list ────────────────────────────────────────────────
  const alertContainer = document.getElementById("ov_alerts");
  if (!abnormal.length && !defectPred) {
    alertContainer.innerHTML = `<p class="muted-text">✓ No active alerts. Process is operating normally.</p>`;
  } else {
    const items = [];
    abnormal.forEach(p => {
      const sev = p.status === "Critical" ? "sev-critical" : "sev-warning";
      const dot = p.status === "Critical" ? "dot-critical" : "dot-warning";
      items.push(`
        <div class="alert-item ${sev}">
          <div class="alert-item-dot ${dot}"></div>
          <div class="alert-item-body">
            <div class="alert-item-title">${p.parameter.replace(/_/g," ")} — ${p.status}</div>
            <div class="alert-item-sub">Value: ${p.value} | Normal range: ${p.normal_range[0]}–${p.normal_range[1]}</div>
          </div>
        </div>`);
    });
    if (defectPred) {
      items.push(`
        <div class="alert-item sev-critical">
          <div class="alert-item-dot dot-critical"></div>
          <div class="alert-item-body">
            <div class="alert-item-title">Defect Likely — ${probPct}% probability</div>
            <div class="alert-item-sub">ML model predicts a defective part. Inspect immediately.</div>
          </div>
        </div>`);
    }
    alertContainer.innerHTML = items.join("");
  }

  // ── 5. Sensor gauges (Live Analysis page) ────────────────────────────────
  const PARAMS = ["temperature", "pressure", "vibration", "machine_speed", "production_rate"];
  PARAMS.forEach(p => {
    const row = document.getElementById("g_" + p);
    if (!row) return;
    const status = monitoring_result.parameter_statuses[p] || "Normal";
    row.className = "gauge-row " + status.toLowerCase();
    row.querySelector(".gauge-value").textContent = parseFloat(process_data[p]).toFixed(2);
  });

  // ── 6. Agent pipeline cards ───────────────────────────────────────────────
  // Agent 1 — Process Monitoring
  const stepMon = document.getElementById("agent_monitoring");
  stepMon.className = "agent-step step-" + overall.toLowerCase();
  const monStatusEl = document.getElementById("a_monitor_status");
  monStatusEl.textContent = overall;
  monStatusEl.className   = "agent-status-txt " + statusCls(overall);
  document.getElementById("a_monitor_detail").textContent =
    abnormal.length
      ? abnormal.map(p => p.parameter.replace(/_/g," ") + ": " + p.value).join(" · ")
      : "All parameters within normal range.";

  // Agent 2 — Quality Analysis
  const qRiskCls = risk === "High" ? "c-critical" : risk === "Medium" ? "c-warning" : "c-normal";
  const stepQual = document.getElementById("agent_quality");
  stepQual.className = "agent-step step-" + (risk === "High" ? "critical" : risk === "Medium" ? "warning" : "normal");
  const qualStatusEl = document.getElementById("a_quality_status");
  qualStatusEl.textContent = risk + " Risk";
  qualStatusEl.className   = "agent-status-txt " + qRiskCls;
  const contrib = quality_result.contributing_parameters || [];
  document.getElementById("a_quality_detail").textContent =
    contrib.length ? "Issues: " + contrib.join(", ") : "No quality issues detected.";

  // Agent 3 — Defect Prediction
  const stepDef = document.getElementById("agent_defect");
  stepDef.className = "agent-step " + (defectPred ? "step-critical" : "step-normal");
  const defStatusEl = document.getElementById("a_defect_status");
  defStatusEl.textContent = defectPred ? "⚠ DEFECT LIKELY" : "✓ NO DEFECT";
  defStatusEl.className   = "agent-status-txt " + (defectPred ? "c-defect" : "c-good");
  const anomalyTxt = defect_result.is_anomaly ? " · Anomaly detected" : "";
  document.getElementById("a_defect_detail").textContent =
    `Defect probability: ${probPct}%${anomalyTxt}`;

  // Agent 4 — Process Optimization
  const immCount = optimization_result.immediate_actions?.length || 0;
  const stepOpt  = document.getElementById("agent_optim");
  stepOpt.className = "agent-step " + (immCount ? "step-critical" : recs.length ? "step-warning" : "step-normal");
  const optStatusEl = document.getElementById("a_optim_status");
  optStatusEl.textContent = `${recs.length} Recommendation${recs.length !== 1 ? "s" : ""}`;
  optStatusEl.className   = "agent-status-txt " + (immCount ? "c-critical" : recs.length ? "c-warning" : "c-good");
  document.getElementById("a_optim_detail").textContent = optimization_result.summary;

  // ── 7. AI Explanation ─────────────────────────────────────────────────────
  document.getElementById("aiExplanation").textContent =
    ai_explanation || "No explanation available.";

  // ── 8. Recommendations ───────────────────────────────────────────────────
  const recEl = document.getElementById("recommendations");
  if (!recs.length) {
    recEl.innerHTML = `<p class="muted-text">No corrective actions required at this time.</p>`;
  } else {
    recEl.innerHTML = recs.map(r => `
      <div class="rec-item priority-${r.priority}">
        <div class="rec-header">
          <span class="rec-category">${r.category}</span>
          <span class="rec-priority">${r.priority}</span>
        </div>
        <div class="rec-action">${r.action}</div>
        <div class="rec-basis">Source: ${r.basis === "data" ? "Data-driven finding" : "Manufacturing knowledge base"}</div>
      </div>`).join("");
  }

  // ── 9. Feature contribution chart ────────────────────────────────────────
  renderFeatureChart(defect_result.top_features || []);
}

// KPI helper
function setKPI(id, value, colorClass) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  el.className   = "kpi-value " + colorClass;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chart builders
// ═══════════════════════════════════════════════════════════════════════════════

// ── Overview: combined trend chart (last 40 rows) ─────────────────────────────
function renderTrendChart(records) {
  const last40  = records.slice(-40);
  const labels  = last40.map((_, i) => `R${records.length - 40 + i + 1}`);

  const datasets = [
    { label: "Temperature (°C)",   data: last40.map(r => r.temperature),         borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,.07)",    tension: .3, fill: false },
    { label: "Pressure (bar×10)",  data: last40.map(r => r.pressure * 10),       borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,.07)",    tension: .3, fill: false },
    { label: "Vibration (×10)",    data: last40.map(r => r.vibration * 10),      borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,.07)",    tension: .3, fill: false },
    { label: "Speed (RPM/10)",     data: last40.map(r => r.machine_speed / 10),  borderColor: "#8b5cf6", backgroundColor: "rgba(139,92,246,.07)",    tension: .3, fill: false },
  ];

  const ctx = document.getElementById("trendChart").getContext("2d");
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { font: { size: 11 }, boxWidth: 12 } } },
      scales: {
        y: { beginAtZero: false, ticks: { font: { size: 11 } } },
        x: { ticks: { font: { size: 10 }, maxTicksLimit: 10 } },
      },
    },
  });
}

// ── Overview: status doughnut ─────────────────────────────────────────────────
function renderStatusChart(counts) {
  const labels = Object.keys(counts);
  const values = Object.values(counts);
  const colors = { Normal: "#16a34a", Warning: "#d97706", Critical: "#dc2626" };

  const ctx = document.getElementById("statusChart").getContext("2d");
  if (statusChart) statusChart.destroy();
  statusChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: labels.map(l => colors[l] || "#9ca3af"), borderWidth: 2, borderColor: "#fff" }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom", labels: { font: { size: 11 }, boxWidth: 14 } } },
    },
  });
}

// ── Live Analysis: feature bar chart ─────────────────────────────────────────
function renderFeatureChart(features) {
  if (!features || !features.length) return;
  const labels = features.map(f => f.feature.replace(/_/g, " "));
  const values = features.map(f => parseFloat(f.contribution.toFixed(4)));
  const colors = ["#ef4444", "#f59e0b", "#3b82f6"];

  const ctx = document.getElementById("featureChart").getContext("2d");
  if (featureChart) featureChart.destroy();
  featureChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Contribution Score", data: values, backgroundColor: colors, borderRadius: 6 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { font: { size: 11 } } },
        y: { ticks: { font: { size: 12 } } },
      },
    },
  });
}

// ── Analytics: individual parameter trend charts ──────────────────────────────
function buildLineChart(canvasId, label, color, data, labels, ref) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return ref;
  if (ref) ref.destroy();
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label,
        data,
        borderColor: color,
        backgroundColor: color.replace(")", ",.08)").replace("rgb", "rgba"),
        tension: .3,
        fill: true,
        pointRadius: 2,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: false, ticks: { font: { size: 11 } } },
        x: { ticks: { font: { size: 10 }, maxTicksLimit: 12 } },
      },
    },
  });
}

function renderAnalyticsCharts(records) {
  const last60 = records.slice(-60);
  const labels = last60.map((_, i) => `R${records.length - 60 + i + 1}`);

  chartTemp  = buildLineChart("chartTemp",  "Temperature (°C)",   "rgb(239,68,68)",   last60.map(r => r.temperature),    labels, chartTemp);
  chartPres  = buildLineChart("chartPres",  "Pressure (bar)",     "rgb(59,130,246)",  last60.map(r => r.pressure),       labels, chartPres);
  chartVib   = buildLineChart("chartVib",   "Vibration (mm/s)",   "rgb(245,158,11)",  last60.map(r => r.vibration),      labels, chartVib);
  chartSpeed = buildLineChart("chartSpeed", "Machine Speed (RPM)","rgb(139,92,246)",  last60.map(r => r.machine_speed),  labels, chartSpeed);
  chartRate  = buildLineChart("chartRate",  "Production Rate",    "rgb(22,163,74)",   last60.map(r => r.production_rate),labels, chartRate);
}

// ── Data & Models: summary table ──────────────────────────────────────────────
function renderSummaryTable(summary) {
  const params = ["temperature", "pressure", "vibration", "machine_speed", "production_rate"];
  const units  = { temperature: "°C", pressure: "bar", vibration: "mm/s", machine_speed: "RPM", production_rate: "u/min" };

  let html = `<table>
    <thead><tr><th>Parameter</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead>
    <tbody>`;
  params.forEach(p => {
    const s = summary.statistics[p];
    if (!s) return;
    const label = p.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) + (units[p] ? ` (${units[p]})` : "");
    html += `<tr>
      <td>${label}</td>
      <td>${s.mean}</td><td>${s.std}</td><td>${s.min}</td><td>${s.max}</td>
    </tr>`;
  });
  html += `</tbody></table>`;

  // Footer pills
  html += `<div class="summary-footer">`;
  html += `<span class="s-pill pill-blue">Total Records: ${summary.total_records}</span>`;
  html += `<span class="s-pill pill-red">Defect Rate: ${summary.defect_rate_percent}%</span>`;
  const qc = summary.quality_status_counts || {};
  const pillMap = { Normal: "pill-green", Warning: "pill-amber", Critical: "pill-red" };
  Object.entries(qc).forEach(([k, v]) => {
    html += `<span class="s-pill ${pillMap[k] || "pill-blue"}">${k}: ${v}</span>`;
  });
  html += `</div>`;

  document.getElementById("datasetSummary").innerHTML = html;
}

// ── Data & Models: dataset preview table (last 10 rows) ──────────────────────
function renderDatasetPreview(records) {
  if (!records.length) return;
  const last10 = records.slice(-10);
  const cols   = Object.keys(last10[0]);

  let html = `<table><thead><tr>${cols.map(c => `<th>${c.replace(/_/g," ")}</th>`).join("")}</tr></thead><tbody>`;
  last10.forEach(row => {
    const qs = row.quality_status;
    html += `<tr>`;
    cols.forEach(c => {
      let cls = "";
      if (c === "quality_status") {
        cls = qs === "Critical" ? "td-critical" : qs === "Warning" ? "td-warning" : "td-normal";
      }
      if (c === "defect") cls = row[c] === 1 ? "td-critical" : "td-normal";
      html += `<td class="${cls}">${row[c]}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;

  document.getElementById("datasetPreview").innerHTML = html;
  const totalEl = document.getElementById("totalRecords");
  if (totalEl) totalEl.textContent = `${records.length} records`;
}

// ── Analytics: defect & anomaly stats panel ───────────────────────────────────
function renderDefectStats(summary, records) {
  const total    = summary.total_records;
  const defects  = records.filter(r => r.defect === 1).length;
  const criticals= records.filter(r => r.quality_status === "Critical").length;
  const warnings = records.filter(r => r.quality_status === "Warning").length;

  document.getElementById("defectStats").innerHTML = `
    <div class="stat-grid">
      <div class="stat-item">
        <div class="stat-value" style="color:var(--green)">${total}</div>
        <div class="stat-label">Total Records</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color:var(--red)">${defects}</div>
        <div class="stat-label">Defect Records</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color:var(--amber)">${warnings}</div>
        <div class="stat-label">Warning Events</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color:var(--red)">${criticals}</div>
        <div class="stat-label">Critical Events</div>
      </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Initialisation
// ═══════════════════════════════════════════════════════════════════════════════

async function init() {
  try {
    const [datasetRes, summaryRes] = await Promise.all([
      apiFetch("/api/dataset"),
      apiFetch("/api/dataset/summary"),
    ]);

    _datasetRecords = datasetRes.data;

    // Overview page charts
    renderTrendChart(_datasetRecords);
    renderStatusChart(summaryRes.quality_status_counts);

    // Data & Models page
    renderSummaryTable(summaryRes);
    renderDatasetPreview(_datasetRecords);

    // Analytics: defect stats (charts rendered lazily on tab visit)
    renderDefectStats(summaryRes, _datasetRecords);

  } catch (e) {
    console.warn("Could not load dataset on init:", e.message);
  }
}

document.addEventListener("DOMContentLoaded", init);
