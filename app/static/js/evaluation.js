/* AlignEval – Evaluation page JavaScript */
"use strict";

let _pieChart = null;
let _barChart = null;

async function loadMetrics() {
  const res = await fetch(`/api/evaluate/${SESSION_ID}/metrics`);
  if (!res.ok) {
    document.getElementById("precisionVal").textContent = "N/A";
    document.getElementById("recallVal").textContent = "N/A";
    document.getElementById("f1Val").textContent = "N/A";
    return;
  }
  const m = await res.json();

  document.getElementById("precisionVal").textContent = (m.precision * 100).toFixed(1) + "%";
  document.getElementById("recallVal").textContent    = (m.recall * 100).toFixed(1) + "%";
  document.getElementById("f1Val").textContent        = (m.f1 * 100).toFixed(1) + "%";

  renderPieChart(m);
  renderBarChart(m);
  renderOptimizationSuggestions(m);
}

function renderPieChart(m) {
  const ctx = document.getElementById("pieChart").getContext("2d");
  if (_pieChart) _pieChart.destroy();
  const correct = m.correct_count;
  const missing = m.missing_triples.length;
  const wrong   = m.wrong_triples.length;

  _pieChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Correctly Learned", "Missing (Gaps)", "Wrong (Incorrect)"],
      datasets: [{
        data: [correct, missing, wrong],
        backgroundColor: ["#69db7c", "#ff6b6b", "#ffd43b"],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: ctx => {
              const total = correct + missing + wrong;
              return ` ${ctx.label}: ${ctx.raw} (${total ? ((ctx.raw/total)*100).toFixed(1) : 0}%)`;
            }
          }
        }
      }
    }
  });
}

function renderBarChart(m) {
  const ctx = document.getElementById("barChart").getContext("2d");
  if (_barChart) _barChart.destroy();
  _barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Source KG", "Learned KG", "Correct", "Missing", "Wrong"],
      datasets: [{
        label: "Triple Count",
        data: [m.total_source, m.total_learned, m.correct_count, m.missing_triples.length, m.wrong_triples.length],
        backgroundColor: ["#4dabf7", "#9775fa", "#69db7c", "#ff6b6b", "#ffd43b"],
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
    }
  });
}

async function loadMissingTriples() {
  const res = await fetch(`/api/evaluate/${SESSION_ID}/missing-triples`);
  if (!res.ok) return;
  const data = await res.json();
  const tbody = document.getElementById("missingTable");
  if (!data.missing.length) {
    tbody.innerHTML = `<tr><td colspan="3" class="text-center text-success">No missing triples 🎉</td></tr>`;
    return;
  }
  tbody.innerHTML = data.missing.map(t =>
    `<tr class="table-danger">
      <td>${t.head}</td>
      <td><span class="badge bg-danger">${t.relation.replace(/_/g," ")}</span></td>
      <td>${t.tail}</td>
    </tr>`).join("");
}

async function loadWrongTriples() {
  const res = await fetch(`/api/evaluate/${SESSION_ID}/wrong-triples`);
  if (!res.ok) return;
  const data = await res.json();
  const tbody = document.getElementById("wrongTable");
  if (!data.wrong.length) {
    tbody.innerHTML = `<tr><td colspan="3" class="text-center text-success">No wrong triples 🎉</td></tr>`;
    return;
  }
  tbody.innerHTML = data.wrong.map(t =>
    `<tr class="table-warning">
      <td>${t.head}</td>
      <td><span class="badge bg-warning text-dark">${t.relation.replace(/_/g," ")}</span></td>
      <td>${t.tail}</td>
    </tr>`).join("");
}

async function loadProbeResults() {
  const res = await fetch(`/api/probe/${SESSION_ID}/prompts`);
  if (!res.ok) return;
  const data = await res.json();
  const tbody = document.getElementById("probeTable");
  if (!data.probe_results.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted">No probe results yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = data.probe_results.slice(0, 100).map(r => {
    const levelBadge = {
      factual:    '<span class="badge bg-info text-dark">Factual</span>',
      relational: '<span class="badge bg-primary">Relational</span>',
      reverse:    '<span class="badge bg-secondary">Reverse</span>',
    }[r.level] || r.level;
    const tripleStr = r.triples.map(t => `<code>${t[0]} → ${t[1]} → ${t[2]}</code>`).join(", ") || '<span class="text-muted">—</span>';
    return `<tr>
      <td>${levelBadge}</td>
      <td style="max-width:260px;white-space:normal;">${r.prompt}</td>
      <td style="max-width:320px;white-space:normal;font-size:.85em;">${r.response.slice(0,200)}${r.response.length > 200 ? '…' : ''}</td>
      <td style="max-width:200px;white-space:normal;font-size:.8em;">${tripleStr}</td>
    </tr>`;
  }).join("");
}

function renderOptimizationSuggestions(m) {
  const missingRatio = m.total_source > 0 ? m.missing_triples.length / m.total_source : 0;
  const wrongRatio   = m.total_learned > 0 ? m.wrong_triples.length / m.total_learned : 0;

  const suggestions = [];

  // Data-level suggestions
  if (m.recall < 0.6) {
    suggestions.push({
      layer: "Data",
      color: "primary",
      icon: "database",
      text: `Recall is low (${(m.recall*100).toFixed(1)}%). The training dataset likely lacks coverage of ${m.missing_triples.length} key knowledge triples. <strong>Suggestion:</strong> Augment the fine-tuning dataset with additional Q&A pairs covering the missing triples. Focus on: ${m.missing_triples.slice(0,3).map(t => `<em>${t[0]} → ${t[1]} → ${t[2]}</em>`).join("; ")}.`
    });
  }

  if (m.precision < 0.5) {
    suggestions.push({
      layer: "Data",
      color: "primary",
      icon: "funnel",
      text: `Precision is low (${(m.precision*100).toFixed(1)}%). The model learned ${m.wrong_triples.length} incorrect triples. <strong>Suggestion:</strong> Review and clean training data for noisy or contradictory facts. Apply data deduplication and fact verification.`
    });
  }

  // Model-level suggestions
  if (m.f1 < 0.5) {
    suggestions.push({
      layer: "Model",
      color: "success",
      icon: "cpu",
      text: `F1 score (${(m.f1*100).toFixed(1)}%) indicates significant knowledge misalignment. <strong>Suggestion:</strong> Consider knowledge-aware fine-tuning with explicit entity/relation supervision. Use contrastive learning to distinguish correct vs incorrect triples.`
    });
  }

  // Reasoning-level suggestions
  if (missingRatio > 0.3) {
    suggestions.push({
      layer: "Reasoning",
      color: "warning",
      icon: "lightbulb",
      text: `${(missingRatio*100).toFixed(1)}% of source triples are missing from model knowledge. <strong>Suggestion:</strong> Implement chain-of-thought prompting or retrieval-augmented generation (RAG) to compensate for knowledge gaps at inference time.`
    });
  }

  if (suggestions.length === 0) {
    suggestions.push({
      layer: "Overall",
      color: "success",
      icon: "check-circle",
      text: `Knowledge alignment looks good! Precision: ${(m.precision*100).toFixed(1)}%, Recall: ${(m.recall*100).toFixed(1)}%, F1: ${(m.f1*100).toFixed(1)}%. Continue monitoring with new domain data.`
    });
  }

  document.getElementById("optimizationSuggestions").innerHTML = suggestions.map(s => `
    <div class="alert alert-${s.color} d-flex gap-3 mb-2">
      <i class="bi bi-${s.icon}-fill fs-4 mt-1 flex-shrink-0"></i>
      <div>
        <span class="badge bg-${s.color} opt-layer me-2">${s.layer} Layer</span>
        ${s.text}
      </div>
    </div>`).join("");
}

document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  loadMissingTriples();
  loadWrongTriples();
  loadProbeResults();
});
