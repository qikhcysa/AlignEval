/* AlignEval – index page JavaScript */
"use strict";

let _sampleDataBlob = null;

async function createSession() {
  const name = document.getElementById("sessionName").value.trim();
  const domain = document.getElementById("sessionDomain").value;
  const model = document.getElementById("modelName").value.trim();
  if (!name) { alert("Please enter a session name."); return; }

  const res = await fetch("/api/sessions/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, domain, model_name: model }),
  });
  if (!res.ok) { alert("Failed to create session."); return; }
  await loadSessions();
}

async function loadSessions() {
  const res = await fetch("/api/sessions/");
  if (!res.ok) return;
  const sessions = await res.json();

  const container = document.getElementById("sessionsContainer");
  const wizardSel = document.getElementById("wizardSession");

  if (sessions.length === 0) {
    container.innerHTML = `<div class="text-center text-muted py-4">
      <i class="bi bi-inbox fs-2 d-block mb-2"></i>No sessions yet. Create one to start.</div>`;
    wizardSel.innerHTML = `<option value="">-- pick a session --</option>`;
    return;
  }

  container.innerHTML = sessions.map(s => `
    <div class="session-item status-${s.status} p-3 border-bottom">
      <div class="d-flex justify-content-between align-items-start">
        <div>
          <strong>${s.name}</strong>
          <span class="badge bg-secondary ms-2">${s.domain}</span>
          <span class="badge ${statusBadgeClass(s.status)} ms-1">${s.status}</span>
        </div>
        <div class="d-flex gap-1">
          ${s.status === 'evaluated' ? `<a href="/graph/${s.session_id}" class="btn btn-xs btn-outline-primary p-1"><i class="bi bi-diagram-3"></i></a>
          <a href="/evaluation/${s.session_id}" class="btn btn-xs btn-outline-success p-1"><i class="bi bi-bar-chart"></i></a>` : ''}
          <button class="btn btn-xs btn-outline-danger p-1" onclick="deleteSession('${s.session_id}')"><i class="bi bi-trash"></i></button>
        </div>
      </div>
      <div class="small text-muted mt-1">
        Source: ${s.source_kg_entities} entities / ${s.source_kg_relations} relations &nbsp;|&nbsp;
        Learned: ${s.learned_kg_entities} entities / ${s.learned_kg_relations} relations
      </div>
    </div>`).join("");

  wizardSel.innerHTML = `<option value="">-- pick a session --</option>` +
    sessions.map(s => `<option value="${s.session_id}">${s.name}</option>`).join("");
}

function statusBadgeClass(status) {
  return { evaluated: "bg-success", probed: "bg-warning text-dark",
           dataset_uploaded: "bg-info text-dark", pending: "bg-secondary" }[status] || "bg-secondary";
}

async function deleteSession(id) {
  if (!confirm("Delete this session?")) return;
  await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  await loadSessions();
}

async function loadSampleDataset() {
  setWizardAlert("info", "Loading sample biomedical dataset…");
  const res = await fetch("/api/sample-dataset");
  if (!res.ok) { setWizardAlert("danger", "Failed to load sample dataset."); return; }
  const data = await res.json();
  _sampleDataBlob = new Blob([JSON.stringify(data)], { type: "application/json" });
  setWizardAlert("success", `Sample dataset loaded (${data.data.length} Q&A pairs). Click 'Run Pipeline'.`);
}

async function runWizard() {
  const sessionId = document.getElementById("wizardSession").value;
  const fileInput = document.getElementById("wizardFile");
  const mockMode = document.getElementById("mockMode").checked;

  if (!sessionId) { setWizardAlert("warning", "Please select or create a session first."); return; }

  let fileBlob = null;
  if (fileInput.files.length > 0) {
    fileBlob = fileInput.files[0];
  } else if (_sampleDataBlob) {
    fileBlob = _sampleDataBlob;
  } else {
    setWizardAlert("warning", "Please upload a dataset or load the sample dataset.");
    return;
  }

  document.getElementById("wizardProgress").style.display = "block";
  setWizardProgress(10, "Uploading dataset and building source KG…");

  // Step 1: Upload dataset
  const formData = new FormData();
  formData.append("file", fileBlob, "dataset.json");
  formData.append("domain", "biomedical");

  const uploadRes = await fetch(`/api/sessions/${sessionId}/upload-dataset`, {
    method: "POST", body: formData,
  });
  if (!uploadRes.ok) {
    const err = await uploadRes.json();
    setWizardAlert("danger", "Upload failed: " + (err.detail || uploadRes.statusText));
    return;
  }
  const uploadData = await uploadRes.json();
  setWizardProgress(40, `Source KG built: ${uploadData.entity_count} entities, ${uploadData.relation_count} relations.`);

  // Step 2: Probe
  setWizardProgress(50, "Running multi-level LLM probing…");
  const probeRes = await fetch(`/api/probe/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mock_mode: mockMode, max_entities: 20, max_relations: 20 }),
  });
  if (!probeRes.ok) {
    const err = await probeRes.json();
    setWizardAlert("danger", "Probing failed: " + (err.detail || probeRes.statusText));
    return;
  }
  const probeData = await probeRes.json();
  setWizardProgress(75, `Probing done: ${probeData.prompt_count} prompts, learned KG has ${probeData.learned_relations} relations.`);

  // Step 3: Evaluate
  setWizardProgress(85, "Aligning KGs and computing metrics…");
  const evalRes = await fetch(`/api/evaluate/${sessionId}`, { method: "POST" });
  if (!evalRes.ok) {
    const err = await evalRes.json();
    setWizardAlert("danger", "Evaluation failed: " + (err.detail || evalRes.statusText));
    return;
  }
  const evalData = await evalRes.json();
  setWizardProgress(100, `Done! Precision: ${(evalData.precision*100).toFixed(1)}%, Recall: ${(evalData.recall*100).toFixed(1)}%, F1: ${(evalData.f1*100).toFixed(1)}%`);

  setWizardAlert("success",
    `Pipeline complete! <a href="/evaluation/${sessionId}" class="alert-link">View Evaluation</a> &nbsp;|&nbsp; <a href="/graph/${sessionId}" class="alert-link">View Graph</a>`);
  await loadSessions();
}

function setWizardProgress(pct, msg) {
  document.getElementById("wizardBar").style.width = pct + "%";
  document.getElementById("wizardStatus").textContent = msg;
}

function setWizardAlert(type, html) {
  document.getElementById("wizardAlert").innerHTML =
    `<div class="alert alert-${type} alert-dismissible" role="alert">${html}
     <button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>`;
}

// Load sessions on page load
document.addEventListener("DOMContentLoaded", loadSessions);
