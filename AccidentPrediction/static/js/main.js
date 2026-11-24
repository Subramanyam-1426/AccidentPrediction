/* ==========================================================================
   Accident Prediction System - Client Side Application Logic
   ========================================================================== */

let sessionHistory = JSON.parse(localStorage.getItem('accident_pred_history') || '[]');
let chartInstances = {};
let metadata = null;
let modelMetrics = null;
let activeSelectedModel = "Decision Tree";

// On DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  fetchMetadata();
  fetchModelMetrics();
  fetchDashboardData();
  renderHistoryTable();
  setupFormListeners();
});

/* ==================== Navigation ==================== */
function initNav() {
  const navBtns = document.querySelectorAll('.nav-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const pageId = btn.getAttribute('data-page');
      switchPage(pageId);
    });
  });

  document.getElementById('brandLogo').addEventListener('click', (e) => {
    e.preventDefault();
    switchPage('home-page');
  });

  // Model Engine Dropdown change listener
  const modelSelect = document.getElementById('selectModelEngine');
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      activeSelectedModel = e.target.value;
      updateActiveModelBadges(activeSelectedModel);
    });
  }
}

function updateActiveModelBadges(modelName) {
  document.getElementById('statModelName').innerText = modelName;
  const selectEl = document.getElementById('selectModelEngine');
  if (selectEl && selectEl.value !== modelName) {
    selectEl.value = modelName;
  }
  if (modelMetrics) {
    renderModelMetricsTable(modelMetrics);
  }
}

function switchPage(pageId) {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    if (btn.getAttribute('data-page') === pageId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  document.querySelectorAll('.page').forEach(page => {
    if (page.id === pageId) {
      page.classList.add('active');
    } else {
      page.classList.remove('active');
    }
  });

  if (pageId === 'dashboard-page') {
    fetchDashboardData();
  } else if (pageId === 'models-page') {
    fetchModelMetrics();
  }
}

/* ==================== Fetch Metadata ==================== */
async function fetchMetadata() {
  try {
    const res = await fetch('/api/metadata');
    if (!res.ok) return;
    const data = await res.json();
    metadata = data.metadata;
    if (data.best_model) {
      activeSelectedModel = data.best_model;
      updateActiveModelBadges(activeSelectedModel);
    }
  } catch (err) {
    console.error("Error loading metadata:", err);
  }
}

/* ==================== Form Controls & Live Validation ==================== */
function updateFormVal(field, valText) {
  const labelEl = document.getElementById(`val${field}`);
  if (labelEl) labelEl.innerText = valText;
}

function checkExperienceLimit() {
  const age = parseInt(document.getElementById('inputDriverAge').value);
  const expInput = document.getElementById('inputDriverExperience');
  const maxAllowedExp = Math.max(0, age - 16);
  
  if (parseInt(expInput.value) > maxAllowedExp) {
    expInput.value = maxAllowedExp;
    updateFormVal('DriverExperience', maxAllowedExp + ' yrs');
  }
}

function setAlcoholToggle(val) {
  document.getElementById('inputDriverAlcohol').value = val;
  const options = document.querySelectorAll('.toggle-option');
  options.forEach(opt => {
    if (parseInt(opt.getAttribute('data-val')) === val) {
      opt.classList.add('active');
    } else {
      opt.classList.remove('active');
    }
  });
}

function adjustVehicles(delta) {
  const inputHidden = document.getElementById('inputNumberVehicles');
  const valDisplay = document.getElementById('valNumberVehicles');
  let current = parseInt(inputHidden.value) || 1;
  current = Math.max(1, Math.min(20, current + delta));
  inputHidden.value = current;
  valDisplay.innerText = current;
}

function validateFormInputs() {
  const age = parseInt(document.getElementById('inputDriverAge').value);
  const exp = parseInt(document.getElementById('inputDriverExperience').value);
  const speed = parseInt(document.getElementById('inputSpeedLimit').value);
  const vehicles = parseInt(document.getElementById('inputNumberVehicles').value);

  const errors = [];
  if (isNaN(age) || age < 18 || age > 80) errors.push("Driver Age must be between 18 and 80 years.");
  if (isNaN(exp) || exp < 0 || exp > 60) errors.push("Driver Experience must be between 0 and 60 years.");
  if (exp > (age - 16)) errors.push(`Driver Experience (${exp} yrs) cannot exceed Age minus 16 (${age - 16} yrs).`);
  if (isNaN(vehicles) || vehicles < 1) errors.push("Number of Vehicles must be at least 1.");
  if (isNaN(speed) || speed < 10 || speed > 250) errors.push("Speed Limit must be between 10 and 250 km/h.");

  const alertBox = document.getElementById('validationAlert');
  const listEl = document.getElementById('validationList');

  if (errors.length > 0) {
    listEl.innerHTML = errors.map(err => `<li>${err}</li>`).join('');
    alertBox.style.display = 'block';
    return false;
  } else {
    alertBox.style.display = 'none';
    return true;
  }
}

function setupFormListeners() {
  const form = document.getElementById('predictionForm');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateFormInputs()) return;

    await submitPrediction();
  });
}

function resetForm() {
  document.getElementById('inputDriverAge').value = 30;
  updateFormVal('DriverAge', '30 yrs');
  document.getElementById('inputDriverExperience').value = 8;
  updateFormVal('DriverExperience', '8 yrs');
  setAlcoholToggle(0);

  document.getElementById('inputRoadType').value = 'Highway';
  document.getElementById('inputRoadCondition').value = 'Dry';
  document.getElementById('inputSpeedLimit').value = 60;
  updateFormVal('SpeedLimit', '60 km/h');
  document.getElementById('inputNumberVehicles').value = 2;
  document.getElementById('valNumberVehicles').innerText = '2';

  document.getElementById('inputWeather').value = 'Clear';
  document.getElementById('inputTimeOfDay').value = 'Afternoon';
  document.getElementById('inputTrafficDensity').value = '1';
  document.getElementById('inputRoadLightCondition').value = 'Daylight';
  document.getElementById('inputVehicleType').value = 'Car';

  document.getElementById('validationAlert').style.display = 'none';
  document.getElementById('resultSection').classList.remove('active');
}

/* ==================== Submit Prediction ==================== */
async function submitPrediction() {
  const btn = document.getElementById('btnPredict');
  const btnIcon = document.getElementById('btnPredictIcon');
  const btnText = document.getElementById('btnPredictText');

  btn.disabled = true;
  btnIcon.innerHTML = `<span class="spinner"></span>`;
  btnText.innerText = `Analyzing with ${activeSelectedModel}...`;

  const payload = {
    Model: activeSelectedModel,
    Weather: document.getElementById('inputWeather').value,
    Road_Type: document.getElementById('inputRoadType').value,
    Time_of_Day: document.getElementById('inputTimeOfDay').value,
    Traffic_Density: document.getElementById('inputTrafficDensity').value,
    Speed_Limit: parseFloat(document.getElementById('inputSpeedLimit').value),
    Number_of_Vehicles: parseFloat(document.getElementById('inputNumberVehicles').value),
    Driver_Alcohol: parseFloat(document.getElementById('inputDriverAlcohol').value),
    Road_Condition: document.getElementById('inputRoadCondition').value,
    Vehicle_Type: document.getElementById('inputVehicleType').value,
    Driver_Age: parseFloat(document.getElementById('inputDriverAge').value),
    Driver_Experience: parseFloat(document.getElementById('inputDriverExperience').value),
    Road_Light_Condition: document.getElementById('inputRoadLightCondition').value
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
      alert("Prediction Error: " + (data.error || "Failed to generate prediction"));
      return;
    }

    displayPredictionResult(data, payload);
    saveToHistory(data, payload);
  } catch (err) {
    alert("Network or Server error: " + err.message);
  } finally {
    btn.disabled = false;
    btnIcon.innerText = "🔮";
    btnText.innerText = "Predict Accident Risk";
  }
}

/* ==================== Display Prediction Result ==================== */
function displayPredictionResult(res, payload) {
  const resultSec = document.getElementById('resultSection');
  resultSec.classList.add('active');

  // Model Badge
  document.getElementById('resultModelBadge').innerText = `Engine: ${res.selected_model}`;
  document.getElementById('activeModelLabelText').innerText = res.selected_model;

  // Status Badge
  const statusBadge = document.getElementById('resultStatusBadge');
  if (res.risk_color === 'rose') {
    statusBadge.className = "prediction-badge rose";
    statusBadge.innerHTML = `<span>🔴</span> ACCIDENT LIKELY`;
  } else if (res.risk_color === 'amber') {
    statusBadge.className = "prediction-badge amber";
    statusBadge.innerHTML = `<span>🟠</span> ${res.prediction_label}`;
  } else {
    statusBadge.className = "prediction-badge emerald";
    statusBadge.innerHTML = `<span>🟢</span> NO ACCIDENT LIKELY`;
  }

  // Circle Gauge Meter Animation
  const percent = res.probability_percent;
  document.getElementById('gaugePercentText').innerText = `${percent}%`;
  document.getElementById('gaugeRiskLabel').innerText = `Risk Level: ${res.risk_level}`;

  const gaugeCircle = document.getElementById('gaugeCircle');
  const circumference = 2 * Math.PI * 70; // 440
  const offset = circumference - (percent / 100) * circumference;
  
  if (res.risk_color === 'rose') {
    gaugeCircle.style.stroke = '#ef4444';
  } else if (res.risk_color === 'amber') {
    gaugeCircle.style.stroke = '#f59e0b';
  } else {
    gaugeCircle.style.stroke = '#10b981';
  }
  
  gaugeCircle.style.strokeDashoffset = offset;

  // Render Key Conditions Grid
  const condGrid = document.getElementById('conditionsGrid');
  condGrid.innerHTML = res.key_conditions.map(cond => `
    <div class="condition-pill">
      <span class="condition-icon">${cond.icon}</span>
      <div class="condition-info">
        <span class="condition-label">${cond.label}</span>
        <span class="condition-val">${cond.value}</span>
      </div>
    </div>
  `).join('');

  // Feature Importance List
  const featureList = document.getElementById('featureImportanceList');
  if (res.model_top_features && Object.keys(res.model_top_features).length > 0) {
    featureList.innerHTML = Object.entries(res.model_top_features).map(([feat, imp]) => `
      <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 20px; padding: 0.4rem 0.85rem; font-size: 0.825rem; font-family: var(--font-mono); color: #cbd5e1;">
        <span style="color: var(--accent-cyan); font-weight: 700;">${feat}:</span> ${(imp * 100).toFixed(1)}%
      </div>
    `).join('');
  } else {
    featureList.innerHTML = `<span style="color: var(--text-dim);">Feature importances evaluated by ${res.selected_model}.</span>`;
  }

  // Safety Recommendations
  const recList = document.getElementById('recommendationsList');
  recList.innerHTML = res.recommendations.map(rec => `
    <div class="rec-item">${rec}</div>
  `).join('');

  // Smooth Scroll to Result
  resultSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ==================== History Management ==================== */
function saveToHistory(res, payload) {
  const item = {
    id: Date.now(),
    timestamp: new Date().toLocaleString(),
    modelUsed: res.selected_model,
    summary: `${payload.Weather}, ${payload.Road_Type}, Alcohol: ${payload.Driver_Alcohol ? 'Yes' : 'No'}, Speed: ${payload.Speed_Limit}km/h`,
    label: res.prediction_label,
    risk: res.risk_level,
    color: res.risk_color,
    probability: `${res.probability_percent}%`
  };

  sessionHistory.unshift(item);
  localStorage.setItem('accident_pred_history', JSON.stringify(sessionHistory));
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = document.getElementById('historyTableBody');
  if (!tbody) return;

  if (sessionHistory.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No predictions recorded in this session yet.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = sessionHistory.map((item, idx) => `
    <tr>
      <td style="font-family: var(--font-mono);">${sessionHistory.length - idx}</td>
      <td style="font-size: 0.825rem; color: var(--text-muted);">${item.timestamp}</td>
      <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-cyan);">${item.modelUsed || 'Decision Tree'}</td>
      <td>${item.summary}</td>
      <td style="font-weight: 700;">${item.label}</td>
      <td>
        <span class="badge-tag" style="background: ${item.color === 'rose' ? 'rgba(239, 68, 68, 0.2)' : item.color === 'amber' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.2)'}; color: ${item.color === 'rose' ? '#f87171' : item.color === 'amber' ? '#fbbf24' : '#34d399'};">
          ${item.risk}
        </span>
      </td>
      <td style="font-family: var(--font-mono); font-weight: 700;">${item.probability}</td>
    </tr>
  `).join('');
}

function clearHistory() {
  if (confirm("Are you sure you want to clear session prediction history?")) {
    sessionHistory = [];
    localStorage.removeItem('accident_pred_history');
    renderHistoryTable();
  }
}

function exportHistoryJSON() {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(sessionHistory, null, 2));
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute("href", dataStr);
  dlAnchor.setAttribute("download", `prediction_history_${Date.now()}.json`);
  document.body.appendChild(dlAnchor);
  dlAnchor.click();
  dlAnchor.remove();
}

/* ==================== Dashboard Charts (Chart.js) ==================== */
async function fetchDashboardData() {
  try {
    const res = await fetch('/api/dashboard-data');
    if (!res.ok) return;
    const data = await res.json();
    renderDashboardCharts(data);
  } catch (err) {
    console.error("Error fetching dashboard data:", err);
  }
}

function renderDashboardCharts(data) {
  createChart('chartTargetDist', 'doughnut', {
    labels: Object.keys(data.target_distribution),
    datasets: [{
      data: Object.values(data.target_distribution),
      backgroundColor: ['#10b981', '#ef4444'],
      borderColor: '#0f172a',
      borderWidth: 2
    }]
  });

  const weatherLabels = Object.keys(data.by_weather);
  const weatherRates = weatherLabels.map(k => data.by_weather[k].rate);
  createChart('chartWeather', 'bar', {
    labels: weatherLabels,
    datasets: [{
      label: 'Accident Rate (%)',
      data: weatherRates,
      backgroundColor: '#8b5cf6',
      borderRadius: 6
    }]
  }, { scales: { y: { beginAtZero: true, max: 100 } } });

  const roadTypeLabels = Object.keys(data.by_road_type);
  const roadTypeRates = roadTypeLabels.map(k => data.by_road_type[k].rate);
  createChart('chartRoadType', 'bar', {
    labels: roadTypeLabels,
    datasets: [{
      label: 'Accident Rate (%)',
      data: roadTypeRates,
      backgroundColor: '#06b6d4',
      borderRadius: 6
    }]
  }, { scales: { y: { beginAtZero: true, max: 100 } } });

  const trafficLabels = Object.keys(data.by_traffic_density);
  const trafficRates = trafficLabels.map(k => data.by_traffic_density[k].rate);
  createChart('chartTraffic', 'bar', {
    labels: trafficLabels,
    datasets: [{
      label: 'Accident Rate (%)',
      data: trafficRates,
      backgroundColor: '#f59e0b',
      borderRadius: 6
    }]
  }, { scales: { y: { beginAtZero: true, max: 100 } } });

  const alcoholLabels = Object.keys(data.by_alcohol);
  const alcoholRates = alcoholLabels.map(k => data.by_alcohol[k].rate);
  createChart('chartAlcohol', 'bar', {
    labels: alcoholLabels.map(l => l === 'Yes' ? 'Alcohol Present' : 'Sober Driver'),
    datasets: [{
      label: 'Accident Rate (%)',
      data: alcoholRates,
      backgroundColor: ['#10b981', '#ef4444'],
      borderRadius: 6
    }]
  }, { scales: { y: { beginAtZero: true, max: 100 } } });

  const vehicleLabels = Object.keys(data.by_vehicle_type);
  const vehicleRates = vehicleLabels.map(k => data.by_vehicle_type[k].rate);
  createChart('chartVehicle', 'bar', {
    labels: vehicleLabels,
    datasets: [{
      label: 'Accident Rate (%)',
      data: vehicleRates,
      backgroundColor: '#3b82f6',
      borderRadius: 6
    }]
  }, { scales: { y: { beginAtZero: true, max: 100 } } });
}

function createChart(canvasId, type, data, extraOptions = {}) {
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }

  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans' } }
      }
    },
    scales: extraOptions.scales ? {
      x: {
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(255, 255, 255, 0.05)' }
      },
      y: {
        ticks: { color: '#94a3b8' },
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ...extraOptions.scales.y
      }
    } : {}
  };

  chartInstances[canvasId] = new Chart(ctx, {
    type: type,
    data: data,
    options: { ...defaultOptions, ...extraOptions }
  });
}

/* ==================== Model Metrics & Interactive Selection ==================== */
async function fetchModelMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    modelMetrics = data;
    renderModelMetricsTable(data);
    renderConfusionMatrices(data);
  } catch (err) {
    console.error("Error fetching model metrics:", err);
  }
}

function selectModelFromTable(modelName) {
  activeSelectedModel = modelName;
  updateActiveModelBadges(modelName);
  switchPage('predict-page');
}

function renderModelMetricsTable(data) {
  const tbody = document.getElementById('metricsTableBody');
  if (!tbody || !data.metrics) return;

  const rows = [];

  for (const [algoName, m] of Object.entries(data.metrics)) {
    const isActive = algoName === activeSelectedModel;
    rows.push(`
      <tr>
        <td style="font-weight: 700; color: #fff;">
          ${algoName} ${algoName === data.best_model ? '⭐' : ''}
        </td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${(m.accuracy * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${(m.precision * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${(m.recall * 100).toFixed(2)}%</td>
        <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);">${(m.f1 * 100).toFixed(2)}%</td>
        <td>
          ${isActive 
            ? '<span class="badge-tag best" style="background: rgba(16, 185, 129, 0.25); color: #34d399; border-color: rgba(16, 185, 129, 0.5);">ACTIVE ENGINE</span>' 
            : `<button type="button" class="btn-secondary" style="padding: 0.35rem 0.85rem; font-size: 0.8rem;" onclick="selectModelFromTable('${algoName}')">⚡ Use Engine</button>`
          }
        </td>
      </tr>
    `);
  }

  tbody.innerHTML = rows.join('');
}

function renderConfusionMatrices(data) {
  const container = document.getElementById('cmContainer');
  if (!container || !data.metrics) return;

  const cards = [];
  for (const [algoName, m] of Object.entries(data.metrics)) {
    const cm = m.confusion_matrix;
    cards.push(`
      <div class="glass-card" style="text-align: center;">
        <h4 style="font-size: 1rem; color: #fff; margin-bottom: 1rem;">${algoName}</h4>
        <div class="cm-grid">
          <div class="cm-cell" style="background: rgba(16, 185, 129, 0.15);">
            <div class="cm-cell-val" style="color: #34d399;">${cm[0][0]}</div>
            <div class="cm-cell-lbl">True Neg (0,0)</div>
          </div>
          <div class="cm-cell" style="background: rgba(239, 68, 68, 0.15);">
            <div class="cm-cell-val" style="color: #f87171;">${cm[0][1]}</div>
            <div class="cm-cell-lbl">False Pos (0,1)</div>
          </div>
          <div class="cm-cell" style="background: rgba(239, 68, 68, 0.15);">
            <div class="cm-cell-val" style="color: #f87171;">${cm[1][0]}</div>
            <div class="cm-cell-lbl">False Neg (1,0)</div>
          </div>
          <div class="cm-cell" style="background: rgba(16, 185, 129, 0.15);">
            <div class="cm-cell-val" style="color: #34d399;">${cm[1][1]}</div>
            <div class="cm-cell-lbl">True Pos (1,1)</div>
          </div>
        </div>
      </div>
    `);
  }

  container.innerHTML = cards.join('');
}
