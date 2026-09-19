const state = {
  data: null,
  cases: null,
  selectedSite: null,
  selectedWorld: null,
  selectedEffort: 6,
  layer: "probability",
  busy: false,
};

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const fmtPct = (value) => value == null ? "—" : `${Math.round(value * 100)}%`;
const fmtPp = (value) => {
  const pp = Math.round(value * 100);
  return `${pp >= 0 ? "+" : ""}${pp} pp`;
};

async function api(path, method = "GET", body = null) {
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Request failed");
  return payload;
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function setBusy(value) {
  state.busy = value;
  document.body.classList.toggle("is-busy", value);
  $("deploy-btn").disabled = value || !state.selectedSite || !canSpend(state.selectedEffort);
  $("reset-btn").disabled = value;
  $("next-case-btn").disabled = value;
  $("case-select").disabled = value;
  $("reveal-btn").disabled = value || !state.data?.can_reveal;
}

function canSpend(effort) {
  return Boolean(
    state.data &&
    effort <= state.data.resources.remaining_budget &&
    state.data.resources.remaining_budget > 0
  );
}

function showTransition(kicker, title, detail) {
  $("transition-kicker").textContent = kicker;
  $("transition-title").textContent = title;
  $("transition-detail").textContent = detail;
  const layer = $("transition-layer");
  layer.classList.remove("hidden");
  requestAnimationFrame(() => layer.classList.add("show"));
}

function hideTransition() {
  const layer = $("transition-layer");
  layer.classList.remove("show");
  setTimeout(() => layer.classList.add("hidden"), 180);
}

function worldById(worldId) {
  return state.data?.top_worlds?.items?.find(w => w.world_id === worldId) || null;
}

function recommendationBySite(siteId) {
  return state.data?.global_recommendations?.find(r => r.site_id === siteId) || null;
}

function nodeById(siteId) {
  return state.data?.nodes?.find(n => n.id === siteId) || null;
}

function ensureSelection() {
  const recommendations = state.data?.global_recommendations || [];
  const known = new Set(state.data?.nodes?.map(n => n.id) || []);
  if (!state.selectedSite || !known.has(state.selectedSite)) {
    state.selectedSite = recommendations[0]?.site_id || null;
  }
  if (state.selectedWorld && !worldById(state.selectedWorld)) {
    state.selectedWorld = null;
  }

  const allowed = state.data?.resources?.effort_levels || [1, 3, 6];
  const affordable = allowed.filter(e => e <= (state.data?.resources?.remaining_budget || 0));
  if (!affordable.includes(state.selectedEffort)) {
    state.selectedEffort = affordable.at(-1) || allowed[0];
  }
}

async function bootstrap() {
  try {
    state.cases = await api("/api/cases");
    renderCaseSelect();
    const data = await api("/api/state");
    render(data);
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

function renderCaseSelect() {
  const select = $("case-select");
  const rows = state.cases?.cases || [];
  select.innerHTML = rows.map(row =>
    `<option value="${row.case_id}">${String(row.index).padStart(3, "0")} · ${row.label}</option>`
  ).join("");
}

function render(data, previous = state.data) {
  state.data = data;
  ensureSelection();

  if (data.case?.case_id) {
    $("case-select").value = data.case.case_id;
  }
  $("initial-detection").textContent = data.incident.initial_detection;
  $("case-number").textContent = data.case.index
    ? `${String(data.case.index).padStart(3, "0")} / ${data.case.count}`
    : data.case.case_id;
  $("budget-left").textContent = `${data.resources.remaining_budget} / ${data.resources.initial_budget}`;
  $("round").textContent = data.resources.round;
  $("truth-state").textContent = data.revealed ? "REVEALED" : "LOCKED";
  $("truth-state").classList.toggle("revealed", data.revealed);

  const budgetRatio = data.resources.initial_budget
    ? data.resources.remaining_budget / data.resources.initial_budget
    : 0;
  $("budget-fill").style.width = `${Math.max(0, Math.round(budgetRatio * 100))}%`;

  renderWorlds();
  renderRecommendations();
  renderSelectedSite();
  renderEffortControls();
  renderMap(previous);
  renderEvidence();
  renderPropagation();
  renderStress();
  renderQ();
  renderReplan();
  renderPerformance();
  renderTimeline();
  renderControls();
}

function renderWorlds() {
  const bundle = state.data.top_worlds;
  $("world-count").textContent = `${bundle.unique_world_count} EXTENTS`;
  $("world-list").innerHTML = bundle.items.map(world => {
    const selected = state.selectedWorld === world.world_id ? "selected" : "";
    const family = world.families.length ? world.families.join(" · ") : "mixed prior";
    const delta = Math.abs(world.delta) < 0.0005 ? "stable" : fmtPp(world.delta);
    return `
      <button class="world-card ${selected}" data-world="${world.world_id}">
        <div class="world-rank">#${world.rank}</div>
        <div class="world-main">
          <strong>Possible extent ${world.rank}</strong>
          <span>${world.occupied_count} occupied sites · ${family}</span>
        </div>
        <div class="world-mass">
          <b>${fmtPct(world.posterior)}</b>
          <small>${delta}</small>
        </div>
      </button>`;
  }).join("") + `
    <div class="other-worlds">
      <span>All other extents</span><b>${fmtPct(bundle.remaining_mass)}</b>
    </div>`;

  document.querySelectorAll("[data-world]").forEach(button => {
    button.addEventListener("click", () => {
      state.selectedWorld = button.dataset.world;
      render(state.data);
    });
  });

  $("clear-world-btn").classList.toggle("hidden", !state.selectedWorld);
  $("scenario-banner").classList.toggle("hidden", !state.selectedWorld);
  if (state.selectedWorld) {
    const world = worldById(state.selectedWorld);
    $("scenario-banner").textContent =
      `WHAT-IF VIEW · POSSIBLE EXTENT #${world.rank} · POSTERIOR ${fmtPct(world.posterior)} · NOT HIDDEN TRUTH`;
  }
}

function renderRecommendations() {
  const rows = state.data.global_recommendations || [];
  const container = $("recommendation-list");
  if (!rows.length) {
    container.innerHTML = '<div class="empty-copy">No new recommendation. Field work is complete.</div>';
    return;
  }

  container.innerHTML = rows.map(row => {
    const selected = row.site_id === state.selectedSite ? "selected" : "";
    return `
      <button class="recommendation-row ${selected}" data-site="${row.site_id}">
        <span class="rec-rank">#${row.rank}</span>
        <span class="rec-main">
          <b>Site ${row.site_id}</b>
          <small>${row.frontier ? "frontier" : "fallback"} · belief ${fmtPct(row.belief)}</small>
        </span>
        <span class="rec-score">${fmtPct(row.predictive_detection["6"])}</span>
      </button>`;
  }).join("");

  container.querySelectorAll("[data-site]").forEach(button => {
    button.addEventListener("click", () => {
      state.selectedSite = button.dataset.site;
      render(state.data);
    });
  });
}

function renderSelectedSite() {
  const node = nodeById(state.selectedSite);
  const rec = recommendationBySite(state.selectedSite);
  if (!node) {
    $("selected-site").textContent = "—";
    $("selected-metrics").innerHTML = '<span class="muted">Choose a site on the map.</span>';
    return;
  }

  $("selected-site").textContent = `SITE ${node.id}`;
  const rank = node.marine_rank ? `#${node.marine_rank}` : "OVERRIDE";
  const pd = node.predictive_detection || rec?.predictive_detection || {};
  $("selected-metrics").innerHTML = `
    <div><span>Marine rank</span><b>${rank}</b></div>
    <div><span>Occupancy belief</span><b>${fmtPct(node.belief)}</b></div>
    <div><span>Uncertainty</span><b>${fmtPct(node.uncertainty)}</b></div>
    <div><span>Frontier</span><b>${node.frontier ? "YES" : "NO"}</b></div>
    <div class="wide"><span>Predicted detection</span>
      <b>e1 ${fmtPct(pd["1"])} · e3 ${fmtPct(pd["3"])} · e6 ${fmtPct(pd["6"])}</b>
    </div>
  `;
  $("follow-marine-btn").disabled = state.data.global_recommendations?.[0]?.site_id === state.selectedSite;
}

function renderEffortControls() {
  document.querySelectorAll(".effort-btn").forEach(button => {
    const effort = Number(button.dataset.effort);
    button.classList.toggle("active", effort === state.selectedEffort);
    button.disabled = !canSpend(effort) || state.data.revealed || state.data.can_reveal;
  });
}

function renderControls() {
  const deploy = $("deploy-btn");
  const done = state.data.can_reveal || state.data.revealed;
  deploy.disabled = state.busy || !state.selectedSite || !canSpend(state.selectedEffort) || done;
  deploy.textContent = done
    ? "FIELD WORK COMPLETE"
    : `DEPLOY SITE ${state.selectedSite || "—"} · EFFORT ${state.selectedEffort}`;
  $("reveal-btn").disabled = state.busy || !state.data.can_reveal;
}

function activeWorldOccupancy() {
  const world = worldById(state.selectedWorld);
  return new Set(world?.occupied_site_ids || []);
}

function nodeLayerValue(node) {
  if (state.selectedWorld) {
    return activeWorldOccupancy().has(node.id) ? 1 : 0;
  }
  if (state.layer === "habitat") return Math.max(0, Math.min(1, node.habitat || 0));
  if (state.layer === "history") return Math.min(1, (node.effort || 0) / 12);
  return Math.max(0, Math.min(1, node.belief || 0));
}

function renderMap(previous) {
  const svg = $("graph");
  svg.innerHTML = "";
  const width = 1100, height = 650;
  const nodes = state.data.nodes;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = x => 60 + ((x - minX) / Math.max(1e-9, maxX - minX)) * 980;
  const sy = y => 60 + ((y - minY) / Math.max(1e-9, maxY - minY)) * 520;
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const prevById = Object.fromEntries((previous?.nodes || []).map(n => [n.id, n]));

  const defs = svgEl("defs");
  defs.innerHTML = `
    <filter id="glow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8df7d4"></path>
    </marker>`;
  svg.appendChild(defs);

  svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, class: "map-water" }));

  state.data.edges.forEach(edge => {
    const a = byId[edge.src], b = byId[edge.dst];
    if (!a || !b) return;
    svg.appendChild(svgEl("line", {
      x1: sx(a.x), y1: sy(a.y), x2: sx(b.x), y2: sy(b.y),
      class: "graph-edge",
    }));
  });

  if (state.data.mission_changed && state.data.replan?.from && state.data.replan?.to) {
    const from = state.data.replan.from.allocations?.[0]?.site_id;
    const to = state.data.replan.to.allocations?.[0]?.site_id;
    if (byId[from] && byId[to]) {
      const a = byId[from], b = byId[to];
      svg.appendChild(svgEl("line", {
        x1: sx(a.x), y1: sy(a.y), x2: sx(b.x), y2: sy(b.y),
        class: "replan-vector",
        "marker-end": "url(#arrow)",
      }));
    }
  }

  nodes.forEach(node => {
    const cx = sx(node.x), cy = sy(node.y);
    const group = svgEl("g", { class: "node-group", "data-site": node.id });
    const value = nodeLayerValue(node);
    group.style.setProperty("--node-intensity", value.toFixed(3));

    if (node.frontier && !state.selectedWorld) {
      group.appendChild(svgEl("circle", { cx, cy, r: 26, class: "frontier-ring" }));
    }
    if (state.selectedSite === node.id) {
      group.appendChild(svgEl("circle", { cx, cy, r: 31, class: "selected-ring" }));
    }
    if (state.data.global_recommendations?.[0]?.site_id === node.id && !state.data.can_reveal) {
      group.appendChild(svgEl("circle", { cx, cy, r: 35, class: "marine-ring" }));
    }

    const coreClass = [
      "node-core",
      node.status === "confirmed_detection" ? "confirmed" : "",
      node.detections > 1 || node.status === "detected" ? "detected" : "",
      node.effort > 0 ? "surveyed" : "",
      state.data.revealed && node.true_occupied ? "true-occupied" : "",
      state.data.revealed && node.true_occupied && node.detections === 0 ? "true-missed" : "",
    ].filter(Boolean).join(" ");

    const outer = svgEl("circle", { cx, cy, r: 20, class: "node-outer" });
    outer.style.opacity = String(0.22 + 0.78 * value);
    group.appendChild(outer);

    const core = svgEl("circle", { cx, cy, r: 12, class: coreClass });
    core.style.opacity = String(0.35 + 0.65 * Math.max(value, 0.12));
    group.appendChild(core);

    const prevBelief = prevById[node.id]?.belief ?? node.belief;
    if (!state.selectedWorld && state.layer === "probability" && Math.abs(prevBelief - node.belief) > 0.005) {
      const delta = svgEl("text", {
        x: cx, y: cy - 31, "text-anchor": "middle", class: "belief-delta",
      });
      delta.textContent = `${fmtPct(prevBelief)}→${fmtPct(node.belief)}`;
      group.appendChild(delta);
    }

    const label = svgEl("text", {
      x: cx, y: cy + 34, "text-anchor": "middle", class: "node-label",
    });
    label.textContent = node.id;
    group.appendChild(label);

    const valueLabel = svgEl("text", {
      x: cx, y: cy + 4, "text-anchor": "middle", class: "node-value",
    });
    valueLabel.textContent = state.selectedWorld
      ? (activeWorldOccupancy().has(node.id) ? "●" : "○")
      : state.layer === "history"
        ? String(node.effort || 0)
        : String(Math.round(value * 100));
    group.appendChild(valueLabel);

    group.style.cursor = "pointer";
    group.addEventListener("click", () => {
      if (node.id === state.data.incident.initial_detection) return;
      state.selectedSite = node.id;
      render(state.data);
    });
    group.addEventListener("mouseenter", event => showTooltip(event, node));
    group.addEventListener("mousemove", moveTooltip);
    group.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(group);
  });

  const title = state.selectedWorld
    ? `Possible extent #${worldById(state.selectedWorld)?.rank || "?"}`
    : state.layer === "probability"
      ? "Posterior occupancy probability"
      : state.layer === "habitat"
        ? "Habitat suitability context"
        : "Observed survey effort";
  $("map-title").textContent = title;
}

function showTooltip(event, node) {
  const rec = recommendationBySite(node.id);
  const tip = $("node-tooltip");
  tip.innerHTML = `
    <strong>SITE ${node.id}</strong>
    <div class="tip-grid">
      <span>Occupancy belief</span><b>${fmtPct(node.belief)}</b>
      <span>Habitat</span><b>${fmtPct(node.habitat)}</b>
      <span>Observed effort</span><b>${node.effort}</b>
      <span>Detections</span><b>${node.detections}</b>
      <span>Marine rank</span><b>${node.marine_rank ? "#" + node.marine_rank : "—"}</b>
    </div>
    ${state.data.revealed ? `<div class="truth-line">TRUE OCCUPANCY: <b>${node.true_occupied ? "PRESENT" : "ABSENT"}</b></div>` : ""}
  `;
  tip.classList.remove("hidden");
  moveTooltip(event);
}

function moveTooltip(event) {
  const stage = document.querySelector(".map-stage");
  const rect = stage.getBoundingClientRect();
  const tip = $("node-tooltip");
  tip.style.left = `${Math.min(rect.width - 240, event.clientX - rect.left + 14)}px`;
  tip.style.top = `${Math.min(rect.height - 175, event.clientY - rect.top + 14)}px`;
}

function hideTooltip() {
  $("node-tooltip").classList.add("hidden");
}

function renderEvidence() {
  const panel = $("evidence-panel");
  const last = state.data.last_round;
  if (!last) {
    panel.className = "insight-body muted";
    panel.textContent = "Awaiting field evidence.";
    return;
  }
  panel.className = "insight-body";
  panel.innerHTML = last.observations.map(obs => {
    const outcome = obs.detection ? "DETECTION" : "NO DETECTION";
    return `
      <div class="evidence-line ${obs.detection ? "positive" : ""}">
        <div><b>SITE ${obs.site_id}</b><span>${outcome} · effort ${obs.effort}</span></div>
        <strong>${fmtPct(obs.belief_before)} → ${fmtPct(obs.belief_after)}</strong>
      </div>`;
  }).join("");
}

function renderPropagation() {
  const panel = $("propagation-panel");
  const rows = state.data.last_round?.propagated_belief_changes || [];
  if (!rows.length) {
    panel.className = "insight-body muted";
    panel.textContent = "No propagated update yet.";
    return;
  }
  panel.className = "insight-body";
  panel.innerHTML = rows.slice(0, 4).map(row => `
    <div class="prop-row">
      <b>Site ${row.site_id}</b>
      <span>${fmtPct(row.belief_before)} → ${fmtPct(row.belief_after)}</span>
      <strong>${fmtPp(row.delta)}</strong>
    </div>`
  ).join("");
}

function renderStress() {
  const panel = $("stress-panel");
  const stress = state.data.model_stress;
  if (!stress) {
    panel.className = "insight-body muted";
    panel.textContent = "No field result yet.";
    return;
  }
  panel.className = "insight-body";
  const impossible = stress.impossible_under_current_ensemble;
  panel.innerHTML = `
    <div class="stress-value ${impossible ? "warning" : ""}">
      <span>Observed-result probability</span>
      <strong>${impossible ? "0%" : fmtPct(stress.observation_probability)}</strong>
    </div>
    <p>${impossible
      ? "The current scenario ensemble did not support this result."
      : "Lower probability means the field return was less expected under the current scenario ensemble."}</p>
    <small>${stress.surprise_bits == null ? "Infinite surprise" : stress.surprise_bits.toFixed(2) + " bits surprise"}</small>
  `;
}

function renderQ() {
  const posterior = state.data.q_posterior || {};
  const rows = Object.entries(posterior);
  $("q-panel").innerHTML = rows.map(([q, weight]) => `
    <div class="q-row">
      <span>q = ${Number(q).toFixed(2)}</span>
      <div class="q-track"><i style="width:${Math.round(weight * 100)}%"></i></div>
      <b>${fmtPct(weight)}</b>
    </div>`
  ).join("") + `<div class="q-mean">posterior mean q = <b>${Number(state.data.q_mean).toFixed(3)}</b></div>`;
}

function renderReplan() {
  const callout = $("replan-callout");
  if (!state.data.mission_changed || !state.data.replan?.from || !state.data.replan?.to || state.data.revealed) {
    callout.classList.add("hidden");
    return;
  }
  const from = state.data.replan.from.allocations?.[0]?.site_id || "—";
  const to = state.data.replan.to.allocations?.[0]?.site_id || "—";
  $("replan-route").textContent = `SITE ${from} → SITE ${to}`;
  callout.classList.remove("hidden");
}

function renderPerformance() {
  const content = $("performance-content");
  const locked = $("performance-locked");
  if (!state.data.performance) {
    content.classList.add("hidden");
    locked.classList.remove("hidden");
    locked.textContent = state.data.can_reveal
      ? "Field work complete. Reveal the hidden extent to score all three tracks."
      : "Outcome comparison stays locked until the field budget is exhausted.";
    return;
  }

  locked.classList.add("hidden");
  content.classList.remove("hidden");
  const p = state.data.performance;
  const entries = [
    ["MARINE", p.marine],
    ["STATIC RESPONSE", p.static],
    ["YOU", p.you],
  ];
  $("scorecards").innerHTML = entries.map(([label, row]) => `
    <div class="scorecard">
      <span>${label}</span>
      <strong>${row.detected_occupied} / ${row.occupied_total}</strong>
      <small>occupied sites detected</small>
    </div>`
  ).join("");

  drawPerformanceChart(entries);
}

function drawPerformanceChart(entries) {
  const svg = $("performance-chart");
  svg.innerHTML = "";
  const width = 900, height = 250;
  const margin = { left: 58, right: 24, top: 24, bottom: 42 };
  const maxEffort = Math.max(18, ...entries.flatMap(([, row]) => row.curve.map(p => p.effort)));
  const sx = x => margin.left + (x / maxEffort) * (width - margin.left - margin.right);
  const sy = y => height - margin.bottom - y * (height - margin.top - margin.bottom);

  for (let i = 0; i <= 4; i++) {
    const y = i / 4;
    svg.appendChild(svgEl("line", {
      x1: margin.left, x2: width - margin.right,
      y1: sy(y), y2: sy(y), class: "chart-grid",
    }));
    const label = svgEl("text", {
      x: margin.left - 10, y: sy(y) + 4,
      "text-anchor": "end", class: "chart-label",
    });
    label.textContent = `${Math.round(y * 100)}%`;
    svg.appendChild(label);
  }

  [0, 6, 12, 18].forEach(x => {
    const label = svgEl("text", {
      x: sx(x), y: height - 14, "text-anchor": "middle", class: "chart-label",
    });
    label.textContent = String(x);
    svg.appendChild(label);
  });

  entries.forEach(([label, row], index) => {
    const points = row.curve.map(p => [sx(p.effort), sy(p.detected_fraction)]);
    const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
    svg.appendChild(svgEl("path", {
      d,
      class: `chart-line series-${index}`,
    }));
    points.forEach(([x, y]) => {
      svg.appendChild(svgEl("circle", {
        cx: x, cy: y, r: 4, class: `chart-dot series-${index}`,
      }));
    });
    const last = points.at(-1);
    if (last) {
      const text = svgEl("text", {
        x: last[0] - 4, y: last[1] - 10, "text-anchor": "end",
        class: `chart-series-label series-${index}`,
      });
      text.textContent = label;
      svg.appendChild(text);
    }
  });
}

function renderTimeline() {
  const recent = (state.data.events || []).slice(-9);
  $("timeline").innerHTML = recent.map(event => `
    <div class="tape-event kind-${event.kind}">
      <span>R${event.round}</span>
      <div><b>${event.title}</b><small>${event.detail}</small></div>
    </div>`
  ).join("");
  $("timeline").scrollLeft = $("timeline").scrollWidth;
}

async function resetCase(caseId = null) {
  try {
    setBusy(true);
    state.selectedSite = null;
    state.selectedWorld = null;
    showTransition("NEW INCIDENT", "Initializing response scenario", "Loading the real monitoring graph and a blinded hidden extent.");
    const data = await api("/api/reset", "POST", { case_id: caseId || $("case-select").value });
    render(data, null);
    setTimeout(hideTransition, 260);
  } catch (error) {
    hideTransition();
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function deploy() {
  if (!state.selectedSite || state.busy) return;
  const before = state.data;
  try {
    setBusy(true);
    showTransition(
      "FIELD TEAM ACTIVE",
      `Surveying site ${state.selectedSite}`,
      `Allocating ${state.selectedEffort} effort units. Hidden truth remains locked.`
    );
    const data = await api("/api/deploy", "POST", {
      site_id: state.selectedSite,
      effort: state.selectedEffort,
    });
    const obs = data.last_round?.observations?.[0];
    if (obs) {
      $("transition-kicker").textContent = "FIELD RETURN";
      $("transition-title").textContent = obs.detection
        ? `Detection at site ${obs.site_id}`
        : `No detection at site ${obs.site_id}`;
      $("transition-detail").textContent =
        `Effort ${obs.effort} · belief ${fmtPct(obs.belief_before)} → ${fmtPct(obs.belief_after)}`;
    }
    await new Promise(resolve => setTimeout(resolve, 420));
    state.selectedSite = data.global_recommendations?.[0]?.site_id || state.selectedSite;
    render(data, before);
    setTimeout(hideTransition, 500);
  } catch (error) {
    hideTransition();
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function reveal() {
  if (!state.data?.can_reveal || state.busy) return;
  try {
    setBusy(true);
    showTransition("EVALUATE", "Revealing the hidden extent", "Scoring Marine, the static response and your own decisions on the same incident.");
    const before = state.data;
    const data = await api("/api/reveal", "POST");
    await new Promise(resolve => setTimeout(resolve, 350));
    render(data, before);
    $("transition-kicker").textContent = "TRUE EXTENT REVEALED";
    $("transition-title").textContent = "Same incident. Same budget. Different decisions.";
    $("transition-detail").textContent = "The comparison now uses the simulator-only hidden occupancy that was locked during planning.";
    setTimeout(hideTransition, 900);
  } catch (error) {
    hideTransition();
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

$("clear-world-btn").addEventListener("click", () => {
  state.selectedWorld = null;
  render(state.data);
});

$("follow-marine-btn").addEventListener("click", () => {
  const first = state.data.global_recommendations?.[0];
  if (first) {
    state.selectedSite = first.site_id;
    render(state.data);
  }
});

document.querySelectorAll(".effort-btn").forEach(button => {
  button.addEventListener("click", () => {
    const effort = Number(button.dataset.effort);
    if (!canSpend(effort)) return;
    state.selectedEffort = effort;
    render(state.data);
  });
});

document.querySelectorAll(".layer-btn").forEach(button => {
  button.addEventListener("click", () => {
    state.layer = button.dataset.layer;
    state.selectedWorld = null;
    document.querySelectorAll(".layer-btn").forEach(b => b.classList.toggle("active", b === button));
    render(state.data);
  });
});

$("deploy-btn").addEventListener("click", deploy);
$("reveal-btn").addEventListener("click", reveal);
$("reset-btn").addEventListener("click", () => resetCase(state.data?.case?.case_id));
$("case-select").addEventListener("change", () => resetCase($("case-select").value));

$("next-case-btn").addEventListener("click", () => {
  const rows = state.cases?.cases || [];
  if (!rows.length) return;
  const current = rows.findIndex(row => row.case_id === state.data?.case?.case_id);
  const next = rows[(current + 1 + rows.length) % rows.length];
  resetCase(next.case_id);
});

bootstrap();
