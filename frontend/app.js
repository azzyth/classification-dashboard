// Classification ML Dashboard — Frontend Logic
const API = '';
let lastPrediction = null;
let distChart = null;
let pollInterval = null;
let allVisualizations = [];
let activeVizFilter = 'all';

// --- Utilities ---
function showToast(msg, type = 'info') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  setTimeout(() => toast.classList.remove('show'), 3000);
}

function getFeatures() {
  const inputs = document.querySelectorAll('#featureInputs input');
  const features = {};
  inputs.forEach(inp => {
    features[inp.name] = parseFloat(inp.value) || 0;
  });
  return features;
}

// --- CSV Upload ---
const dropZone = document.getElementById('dropZone');
const csvInput = document.getElementById('csvInput');
const browseBtn = document.getElementById('browseBtn');

browseBtn.addEventListener('click', () => csvInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target === dropZone || e.target.closest('.drop-content')) csvInput.click();
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) uploadCSV(files[0]);
});

csvInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) uploadCSV(e.target.files[0]);
});

async function uploadCSV(file) {
  const progress = document.getElementById('uploadProgress');
  const dropContent = document.querySelector('.drop-content');
  dropContent.style.display = 'none';
  progress.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API}/upload-csv`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.detail) throw new Error(data.detail);

    progress.classList.add('hidden');
    dropContent.style.display = 'block';

    showAnalysisSummary(data);
    allVisualizations = data.visualizations || [];
    renderVizGallery();
    buildFeatureInputs(data.feature_names, data.feature_types);
    populateTrueLabelSelect(data.class_distribution);
    await refreshAll();
    showToast(`CSV uploaded: ${data.n_rows} rows, ${data.visualizations.length} visualizations generated`, 'success');
  } catch (e) {
    progress.classList.add('hidden');
    dropContent.style.display = 'block';
    showToast('Upload failed: ' + e.message, 'error');
  }
}

function showAnalysisSummary(data) {
  const summary = document.getElementById('analysisSummary');
  summary.classList.remove('hidden');

  document.getElementById('sumRows').textContent = data.n_rows;
  document.getElementById('sumCols').textContent = data.n_cols;
  document.getElementById('sumFeatures').textContent = data.feature_names.length;
  document.getElementById('sumTarget').textContent = data.target_name;
  document.getElementById('sumViz').textContent = data.visualizations.length;
  document.getElementById('sumModelVer').textContent = 'v' + data.model_version;

  // Feature type tags
  const ftList = document.getElementById('featureTypeList');
  ftList.innerHTML = '';
  for (const [feat, ftype] of Object.entries(data.feature_types)) {
    const tag = document.createElement('span');
    tag.className = `feature-tag ${ftype}`;
    tag.textContent = `${feat} (${ftype})`;
    ftList.appendChild(tag);
  }

  // Class distribution
  const cd = document.getElementById('classDist');
  cd.innerHTML = 'Class distribution: ' + Object.entries(data.class_distribution)
    .map(([k, v]) => `${k}: ${v}`).join(' | ');

  // Target override
  const override = document.getElementById('targetOverride');
  override.classList.remove('hidden');
  const select = document.getElementById('targetSelect');
  select.innerHTML = '';
  // We need column names — fetch from csv-info or just use feature_names + target
  const allCols = [...data.feature_names, data.target_name];
  allCols.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    if (c === data.target_name) opt.selected = true;
    select.appendChild(opt);
  });
}

// Reprocess with different target
document.getElementById('reprocessBtn').addEventListener('click', async () => {
  const target = document.getElementById('targetSelect').value;
  // Re-upload is not possible without the file; instead inform user
  showToast('To change target, re-upload the CSV with target param', 'info');
});

// --- Visualization Gallery ---
function renderVizGallery() {
  const gallery = document.getElementById('vizGallery');
  const filtered = activeVizFilter === 'all'
    ? allVisualizations
    : allVisualizations.filter(v => v.viz_type === activeVizFilter);

  if (filtered.length === 0) {
    gallery.innerHTML = '<div class="empty-state">No visualizations match this filter.</div>';
    return;
  }

  gallery.innerHTML = '';
  filtered.forEach(viz => {
    const card = document.createElement('div');
    card.className = 'viz-card-item';
    if (viz.html) {
      const frame = document.createElement('iframe');
      frame.srcdoc = viz.html;
      frame.style.width = '100%';
      frame.style.height = '260px';
      frame.style.border = '0';
      frame.setAttribute('sandbox', 'allow-scripts allow-same-origin');
      frame.loading = 'lazy';
      card.appendChild(frame);
    } else if (viz.image_base64) {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${viz.image_base64}`;
      img.alt = viz.title;
      img.loading = 'lazy';
      card.appendChild(img);
    }
    const title = document.createElement('div');
    title.className = 'viz-card-title';
    title.textContent = viz.title;
    card.appendChild(title);
    const type = document.createElement('div');
    type.className = 'viz-card-type';
    type.textContent = viz.viz_type.replace('_', ' ');
    card.appendChild(type);
    card.addEventListener('click', () => openModal(viz));
    gallery.appendChild(card);
  });
}

// Viz tab filtering
document.querySelectorAll('.viz-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeVizFilter = tab.dataset.type;
    renderVizGallery();
  });
});

// --- Image Modal ---
let lastFocusBeforeModal = null;
function openModal(viz) {
  const modal = document.getElementById('imgModal');
  const frame = document.getElementById('modalFrame');
  const img = document.getElementById('modalImg');
  if (viz.html) {
    frame.srcdoc = viz.html;
    frame.style.display = 'block';
    img.style.display = 'none';
  } else if (viz.image_base64) {
    img.src = `data:image/png;base64,${viz.image_base64}`;
    img.style.display = 'block';
    frame.style.display = 'none';
  }
  document.getElementById('modalTitle').textContent = viz.title;
  lastFocusBeforeModal = document.activeElement;
  modal.classList.remove('hidden');
  document.getElementById('modalClose').focus();
}
function closeModal() {
  const modal = document.getElementById('imgModal');
  modal.classList.add('hidden');
  if (lastFocusBeforeModal) lastFocusBeforeModal.focus();
}
document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('imgModal').addEventListener('click', (e) => {
  if (e.target.id === 'imgModal') closeModal();
});
document.addEventListener('keydown', (e) => {
  const modal = document.getElementById('imgModal');
  if (modal.classList.contains('hidden')) return;
  if (e.key === 'Escape') { closeModal(); return; }
  if (e.key !== 'Tab') return;
  const focusables = modal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
  if (focusables.length === 0) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
});

// --- Dynamic Feature Inputs ---
function buildFeatureInputs(featureNames, featureTypes) {
  const container = document.getElementById('featureInputs');
  container.innerHTML = '';
  featureNames.forEach(feat => {
    const ftype = featureTypes[feat] || 'numeric';
    const label = document.createElement('label');
    label.innerHTML = `${feat} <span class="feature-tag ${ftype}" style="font-size:0.65rem">${ftype}</span>
      <input type="number" name="${feat}" step="0.01" value="0">`;
    container.appendChild(label);
  });
}

function populateTrueLabelSelect(classDist) {
  const select = document.getElementById('trueLabel');
  select.innerHTML = '<option value="">— none —</option>';
  for (const cls of Object.keys(classDist)) {
    const opt = document.createElement('option');
    opt.value = cls;
    opt.textContent = cls;
    select.appendChild(opt);
  }
}

// --- Predict ---
async function predict() {
  const features = getFeatures();
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    const data = await res.json();
    lastPrediction = { features, ...data };
    showResult(data);
    document.getElementById('saveBtn').disabled = false;
  } catch (e) {
    showToast('Prediction failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Predict';
  }
}

function showResult(data) {
  const card = document.getElementById('resultCard');
  card.classList.remove('hidden');
  document.getElementById('predClass').textContent = data.predicted_class;
  document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const bars = document.getElementById('probBars');
  bars.innerHTML = '';
  const colors = ['a', 'b', 'c', 'd', 'e'];
  data.probabilities.forEach((p, i) => {
    const cls = data.class_names[i];
    const row = document.createElement('div');
    row.className = 'prob-bar-row';
    row.innerHTML = `
      <span class="prob-bar-label">${cls}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${colors[i % colors.length]}" style="width:${(p*100).toFixed(1)}%">
          ${(p*100).toFixed(1)}%
        </div>
      </div>
    `;
    bars.appendChild(row);
  });
}

// --- Save Record ---
async function saveRecord() {
  if (!lastPrediction) return;
  const trueLabel = document.getElementById('trueLabel').value || null;
  const btn = document.getElementById('saveBtn');
  btn.disabled = true;

  try {
    const res = await fetch(`${API}/data`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: lastPrediction.features, true_label: trueLabel }),
    });
    const data = await res.json();
    showToast(`Record #${data.record.id} saved`, 'success');
    await refreshAll();
  } catch (e) {
    showToast('Save failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

// --- Metrics ---
async function refreshMetrics() {
  try {
    const res = await fetch(`${API}/metrics`);
    const data = await res.json();
    document.getElementById('mAccuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
    document.getElementById('mMacro').textContent = (data.macro_f1 * 100).toFixed(1) + '%';
    document.getElementById('mMicro').textContent = (data.micro_f1 * 100).toFixed(1) + '%';
    document.getElementById('mSamples').textContent = data.n_samples;
  } catch (e) { /* silent */ }
}

// --- Confusion Matrix ---
async function refreshConfusionMatrix() {
  try {
    const res = await fetch(`${API}/confusion-matrix`);
    const data = await res.json();
    const container = document.getElementById('cmContainer');
    const classes = data.class_names;
    const matrix = data.matrix;
    const maxVal = Math.max(...matrix.flat(), 1);

    let html = '';
    html += '<div class="cm-cell header"></div>';
    classes.forEach(c => { html += `<div class="cm-cell header">${c}</div>`; });

    matrix.forEach((row, i) => {
      html += `<div class="cm-cell header">${classes[i]}</div>`;
      row.forEach((val, j) => {
        const intensity = val / maxVal;
        let cls = 'zero';
        if (val > 0) {
          if (intensity > 0.66) cls = 'high';
          else if (intensity > 0.33) cls = 'med';
          else cls = 'low';
        }
        html += `<div class="cm-cell ${cls}" title="${classes[i]} → ${classes[j]}: ${val}">${val}</div>`;
      });
    });

    container.innerHTML = html;
    container.style.gridTemplateColumns = `repeat(${classes.length + 1}, 52px)`;
  } catch (e) { /* silent */ }
}

// --- Distribution Chart ---
async function refreshDistribution() {
  try {
    const res = await fetch(`${API}/distribution`);
    const data = await res.json();
    const ctx = document.getElementById('distChart').getContext('2d');
    if (distChart) distChart.destroy();
    distChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.class_names,
        datasets: [
          { label: 'Prior (Train)', data: data.prior.map(v => (v * 100).toFixed(1)), backgroundColor: 'rgba(99,102,241,0.5)', borderColor: '#6366f1', borderWidth: 1 },
          { label: 'Predicted', data: data.predicted.map(v => (v * 100).toFixed(1)), backgroundColor: 'rgba(16,185,129,0.5)', borderColor: '#10b981', borderWidth: 1 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'top', labels: { color: getComputedStyle(document.body).getPropertyValue('--text') } } },
        scales: {
          y: { beginAtZero: true, ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') }, title: { display: true, text: '%' } },
          x: { ticks: { color: getComputedStyle(document.body).getPropertyValue('--text-muted') } },
        },
      },
    });
  } catch (e) { /* silent */ }
}

// --- Data Table ---
async function refreshTable() {
  try {
    const res = await fetch(`${API}/data?limit=200`);
    const records = await res.json();
    const tbody = document.getElementById('tableBody');
    const empty = document.getElementById('emptyState');
    const search = document.getElementById('searchInput').value.toLowerCase();

    // Dynamic table header based on first record's features
    const thead = document.getElementById('tableHead');
    if (records.length > 0) {
      const featKeys = Object.keys(records[0].features);
      let headerHtml = '<tr><th>ID</th>';
      featKeys.forEach(k => { headerHtml += `<th>${k}</th>`; });
      headerHtml += '<th>True</th><th>Predicted</th><th>Conf</th><th>Time</th></tr>';
      thead.innerHTML = headerHtml;
    }

    tbody.innerHTML = '';
    let count = 0;
    records.forEach(r => {
      const featStr = Object.values(r.features).map(v => typeof v === 'number' ? v.toFixed(2) : v).join(', ');
      const searchStr = (r.id + ' ' + featStr + ' ' + (r.true_label || '') + ' ' + r.predicted_label).toLowerCase();
      if (search && !searchStr.includes(search)) return;
      count++;
      const tr = document.createElement('tr');
      const trueTag = r.true_label ? `<span class="tag tag-${String(r.true_label).toLowerCase().charAt(0)}">${r.true_label}</span>` : '—';
      const predTag = `<span class="tag tag-${String(r.predicted_label).toLowerCase().charAt(0)}">${r.predicted_label}</span>`;
      const time = new Date(r.created_at).toLocaleTimeString();
      let rowHtml = `<td>${r.id}</td>`;
      for (const k of Object.keys(r.features)) {
        const v = r.features[k];
        rowHtml += `<td>${typeof v === 'number' ? v.toFixed(2) : v}</td>`;
      }
      rowHtml += `<td>${trueTag}</td><td>${predTag}</td><td>${(r.confidence * 100).toFixed(0)}%</td><td>${time}</td>`;
      tr.innerHTML = rowHtml;
      tbody.appendChild(tr);
    });

    empty.style.display = count === 0 ? 'block' : 'none';
    document.querySelector('.table-wrapper').style.display = count === 0 ? 'none' : 'block';
  } catch (e) { /* silent */ }
}

async function refreshAll() {
  await Promise.all([refreshMetrics(), refreshConfusionMatrix(), refreshDistribution(), refreshTable()]);
}

// --- Retrain ---
async function retrain() {
  const btn = document.getElementById('retrainBtn');
  btn.classList.add('loading');
  btn.innerHTML = '<span class="spinner"></span> Training...';
  try {
    const res = await fetch(`${API}/retrain`, { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      showToast(`Model retrained (v${data.version}) — Macro-F1: ${(data.metrics.macro_f1 * 100).toFixed(1)}%`, 'success');
      await refreshAll();
    } else {
      showToast(data.detail || 'Retrain failed', 'error');
    }
  } catch (e) {
    showToast('Retrain failed: ' + e.message, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = '🔄 Retrain';
  }
}

// --- Clear ---
async function clearData() {
  if (!confirm('Clear all records, uploaded CSV, and reset model?')) return;
  try {
    await fetch(`${API}/csv`, { method: 'DELETE' });
    showToast('All data cleared', 'info');
    lastPrediction = null;
    document.getElementById('resultCard').classList.add('hidden');
    document.getElementById('saveBtn').disabled = true;
    document.getElementById('analysisSummary').classList.add('hidden');
    document.getElementById('targetOverride').classList.add('hidden');
    document.getElementById('vizGallery').innerHTML = '<div class="empty-state">Upload a CSV to generate visualizations.</div>';
    allVisualizations = [];
    document.getElementById('featureInputs').innerHTML = '';
    await refreshAll();
  } catch (e) {
    showToast('Clear failed: ' + e.message, 'error');
  }
}

// --- Health ---
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const badge = document.getElementById('healthBadge');
    badge.textContent = `● v${data.model_version} · ${data.record_count} records`;
    badge.className = 'badge badge-ok';
  } catch {
    document.getElementById('healthBadge').textContent = '● Offline';
    document.getElementById('healthBadge').className = 'badge badge-err';
  }
}

// --- Theme selector ---
function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  const sel = document.getElementById('themeSelect');
  sel.value = saved;
  sel.addEventListener('change', () => {
    const t = sel.value;
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('theme', t);
    refreshDistribution();
  });
}

// --- Density toggle ---
function initDensity() {
  const saved = localStorage.getItem('density');
  if (saved === 'compact') document.body.classList.add('compact');
  document.getElementById('densityToggle').addEventListener('click', () => {
    document.body.classList.toggle('compact');
    localStorage.setItem('density', document.body.classList.contains('compact') ? 'compact' : 'comfortable');
  });
}

// --- Drop zone keyboard support ---
function initDropZone() {
  const dz = document.getElementById('dropZone');
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); document.getElementById('csvInput').click(); }
  });
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initDensity();
  initDropZone();
  document.getElementById('predictBtn').addEventListener('click', predict);
  document.getElementById('saveBtn').addEventListener('click', saveRecord);
  document.getElementById('retrainBtn').addEventListener('click', retrain);
  document.getElementById('clearBtn').addEventListener('click', clearData);
  document.getElementById('searchInput').addEventListener('input', refreshTable);

  document.querySelectorAll('.form-row input').forEach(inp => {
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') predict(); });
  });

  refreshAll();
  checkHealth();
  pollInterval = setInterval(() => { refreshMetrics(); refreshConfusionMatrix(); checkHealth(); }, 5000);
});
