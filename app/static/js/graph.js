/* AlignEval – D3 Knowledge Graph Visualization */
"use strict";

let _graphData = null;
let _currentMode = "source";
let _currentFilter = "all";
let _simulation = null;
let _svg = null;
let _g = null;
let _zoom = null;

const STATUS_COLORS = {
  source:  "#4dabf7",
  matched: "#69db7c",
  missing: "#ff6b6b",
  wrong:   "#ffd43b",
  learned: "#9775fa",
};

const NODE_TYPE_COLORS = {
  PERSON: "#e7c0f6",
  ORGANIZATION: "#74c0fc",
  LOCATION: "#96f2d7",
  DISEASE: "#ff8787",
  DRUG: "#ffa94d",
  AI_CONCEPT: "#a5d8ff",
  PROGRAMMING_LANGUAGE: "#b2f2bb",
  UNKNOWN: "#dee2e6",
};

async function fetchGraph(mode) {
  const url = {
    source:  `/api/evaluate/${SESSION_ID}/source-graph`,
    learned: `/api/evaluate/${SESSION_ID}/learned-graph`,
    aligned: `/api/evaluate/${SESSION_ID}/aligned-graph`,
  }[mode];
  const res = await fetch(url);
  if (!res.ok) return null;
  return await res.json();
}

async function switchGraph(mode, btn) {
  // Update button state
  document.querySelectorAll(".btn-group .btn").forEach(b => {
    if (["source","learned","aligned"].some(m => b.getAttribute("onclick") && b.getAttribute("onclick").includes(m))) {
      b.classList.remove("active");
    }
  });
  btn.classList.add("active");

  _currentMode = mode;
  document.getElementById("graph-loading").style.display = "flex";
  document.getElementById("graph-svg").style.display = "none";

  const data = await fetchGraph(mode);
  if (!data) {
    document.getElementById("graph-loading").innerHTML =
      `<div class="text-danger">Failed to load graph. Run pipeline first.</div>`;
    return;
  }
  _graphData = data;
  renderGraph(data);
}

function renderGraph(data) {
  const container = document.getElementById("graph-container");
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Remove old SVG content
  const svgEl = document.getElementById("graph-svg");
  svgEl.style.display = "block";
  document.getElementById("graph-loading").style.display = "none";
  d3.select("#graph-svg").selectAll("*").remove();

  _svg = d3.select("#graph-svg").attr("width", width).attr("height", height);

  // Defs: arrowhead markers
  const defs = _svg.append("defs");
  Object.entries(STATUS_COLORS).forEach(([key, color]) => {
    defs.append("marker")
      .attr("id", `arrow-${key}`)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", color);
  });

  _zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", e => {
    _g.attr("transform", e.transform);
  });
  _svg.call(_zoom);

  _g = _svg.append("g");

  // Filter data by current filter
  const filteredLinks = filterLinks(data.links);
  const usedNodeIds = new Set(filteredLinks.flatMap(l => [l.source, l.target]));
  const filteredNodes = data.nodes.filter(n => usedNodeIds.has(n.id) || _currentFilter === "all");

  // Build node index
  const nodeById = new Map(data.nodes.map(n => [n.id, n]));

  // Link elements
  const link = _g.append("g").selectAll("line")
    .data(filteredLinks)
    .join("line")
      .attr("class", d => `link ${d.status || d.kg}`)
      .attr("stroke", d => STATUS_COLORS[d.status] || STATUS_COLORS[d.kg] || "#aaa")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", d => (d.status === "missing" ? "6,3" : d.status === "wrong" ? "4,2" : null))
      .attr("marker-end", d => `url(#arrow-${d.status || d.kg})`);

  // Edge labels
  const edgeLabel = _g.append("g").selectAll("text")
    .data(filteredLinks)
    .join("text")
      .attr("class", "link-label")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .text(d => d.relation.replace(/_/g, " "));

  // Node elements
  const node = _g.append("g").selectAll("g.node")
    .data(filteredNodes)
    .join("g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded))
      .on("click", (event, d) => showNodeInfo(d));

  node.append("circle")
    .attr("r", 9)
    .attr("fill", d => NODE_TYPE_COLORS[d.type] || NODE_TYPE_COLORS.UNKNOWN)
    .attr("stroke", d => d.kg === "both" ? "#fff" : (d.kg === "learned" ? "#9775fa" : "#4dabf7"))
    .attr("stroke-width", 2);

  node.append("text")
    .attr("x", 12)
    .attr("y", 4)
    .text(d => d.label ? (d.label.length > 20 ? d.label.slice(0, 18) + "…" : d.label) : d.id);

  // Simulation
  if (_simulation) _simulation.stop();
  _simulation = d3.forceSimulation(filteredNodes)
    .force("link", d3.forceLink(filteredLinks).id(d => d.id).distance(120).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(18))
    .on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      edgeLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);

      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

  // Stats
  updateStats(data, filteredLinks, filteredNodes);
}

function filterLinks(links) {
  if (_currentFilter === "all") return links;
  return links.filter(l => l.status === _currentFilter || (!l.status && l.kg === _currentFilter));
}

function filterStatus(status, btn) {
  document.querySelectorAll(".btn-group .btn").forEach(b => {
    if (b.getAttribute("onclick") && b.getAttribute("onclick").includes("filterStatus")) {
      b.classList.remove("active");
    }
  });
  btn.classList.add("active");
  _currentFilter = status;
  if (_graphData) renderGraph(_graphData);
}

function updateStats(data, links, nodes) {
  const counts = {};
  links.forEach(l => { counts[l.status || l.kg] = (counts[l.status || l.kg] || 0) + 1; });
  document.getElementById("graphStats").innerHTML = `
    <div class="row g-2 text-center">
      <div class="col-6"><div class="fw-bold fs-5">${nodes.length}</div><div class="small text-muted">Nodes</div></div>
      <div class="col-6"><div class="fw-bold fs-5">${links.length}</div><div class="small text-muted">Edges</div></div>
      ${Object.entries(counts).map(([k,v]) =>
        `<div class="col-6"><div class="fw-bold" style="color:${STATUS_COLORS[k]||'#aaa'}">${v}</div>
         <div class="small text-muted">${k}</div></div>`).join("")}
    </div>`;
}

function showNodeInfo(d) {
  document.getElementById("nodeInfo").innerHTML = `
    <dl class="mb-0 row g-1 small">
      <dt class="col-4 text-muted">Label</dt><dd class="col-8">${d.label || d.id}</dd>
      <dt class="col-4 text-muted">Type</dt><dd class="col-8"><span class="badge" style="background:${NODE_TYPE_COLORS[d.type]||'#dee2e6'};color:#333">${d.type||'UNKNOWN'}</span></dd>
      <dt class="col-4 text-muted">Source</dt><dd class="col-8">${d.kg || '—'}</dd>
    </dl>`;
}

function resetZoom() {
  if (_svg && _zoom) _svg.transition().call(_zoom.transform, d3.zoomIdentity);
}

function dragStarted(event, d) {
  if (!event.active) _simulation.alphaTarget(0.3).restart();
  d.fx = d.x; d.fy = d.y;
}
function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
function dragEnded(event, d) {
  if (!event.active) _simulation.alphaTarget(0);
  d.fx = null; d.fy = null;
}

// Init: load source graph
document.addEventListener("DOMContentLoaded", () => {
  switchGraph("source", document.querySelector(".btn-group .btn.active"));
});
