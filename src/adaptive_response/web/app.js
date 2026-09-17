const state = { data: null, busy: false };
const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

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

function phaseLabel(phase) {
  return {
    ready_to_plan: "READY TO PLAN",
    mission_planned: "MISSION ACTIVE",
    complete: "FIELD WORK COMPLETE",
    revealed: "TRUE EXTENT REVEALED",
  }[phase] || phase.replaceAll("_", " ").toUpperCase();
}

function fmtBelief(value) {
  return `${Math.round(value * 100)}%`;
}

function shortSite(siteId) {
  return siteId.replace("site_", "").toUpperCase();
}

function siteName(data, siteId) {
  const node = data.nodes.find(n => n.id === siteId);
  return node ? node.label : siteId;
}

function missionTargets(mission) {
  if (!mission) return [];
  return mission.allocations.map(a => a.site_id);
}

function missionTargetCopy(data, mission) {
  if (!mission) return "No active mission";
  return mission.allocations.map(a => `${shortSite(a.site_id)} · ${a.effort_units} checks`).join("  /  ");
}

// "Why this mission" copy is derived from the active Planner's own ranking
// diagnostics (e.g. FrontierPlanner's ranked_selected_sites), never invented text.
function missionReasonText(mission, siteId) {
  const row = mission?.diagnostics?.ranked_selected_sites?.find(r => r.site_id === siteId);
  if (!row) return "observable-state assignment";
  const bits = [];
  if (row.frontier !== undefined) bits.push(row.frontier ? "frontier site" : "non-frontier pick");
  if (row.belief !== undefined) bits.push(`belief ${fmtBelief(row.belief)}`);
  if (row.uncertainty !== undefined) bits.push(`uncertainty ${fmtBelief(row.uncertainty)}`);
  return bits.length ? bits.join(" · ") : "observable-state assignment";
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("is-busy", busy);
  $("primary-btn").disabled = busy || $("primary-btn").dataset.action === "none";
  $("reset-btn").disabled = busy;
  $("reveal-btn").disabled = busy || !state.data?.can_reveal;
}

function showTransition(kicker, title, detail, tone = "mint") {
  const layer = $("transition-layer");
  $("transition-kicker").textContent = kicker;
  $("transition-title").textContent = title;
  $("transition-detail").textContent = detail;
  layer.dataset.tone = tone;
  layer.classList.remove("hidden");
  requestAnimationFrame(() => layer.classList.add("show"));
}

function hideTransition() {
  const layer = $("transition-layer");
  layer.classList.remove("show");
  setTimeout(() => layer.classList.add("hidden"), 220);
}

function render(data, previous = state.data) {
  state.data = data;
  $("phase-pill").textContent = phaseLabel(data.phase);
  $("scenario-name").textContent = data.incident.scenario_name || data.incident.label;
  $("initial-detection").textContent = data.incident.initial_detection;
  $("budget-left").textContent = `${data.resources.remaining_budget} / ${data.resources.initial_budget}`;
  $("teams").textContent = data.resources.teams;
  $("round").textContent = data.resources.round;
  $("truth-status").textContent = data.revealed ? "TRUE EXTENT REVEALED" : "HIDDEN EXTENT LOCKED";
  $("truth-lock-copy").textContent = data.revealed ? "REVEALED" : "LOCKED";

  const budgetRatio = data.resources.initial_budget > 0
    ? data.resources.remaining_budget / data.resources.initial_budget
    : 0;
  $("budget-fill").style.width = `${Math.round(budgetRatio * 100)}%`;

  renderGraph(data, previous);
  renderMission(data);
  renderEvidence(data);
  renderCausalLoop(data);
  renderTimeline(data.events);
  renderControls(data);
  renderReplanCallout(data);
}

function renderControls(data) {
  const primary = $("primary-btn");
  const reveal = $("reveal-btn");
  primary.disabled = false;

  if (data.can_plan) {
    primary.textContent = data.resources.round === 0 ? "GENERATE MISSION 1" : "GENERATE NEXT MISSION";
    primary.dataset.action = "plan";
  } else if (data.can_execute) {
    primary.textContent = `DEPLOY MISSION ${data.resources.round + 1}`;
    primary.dataset.action = "execute";
  } else if (data.can_reveal) {
    primary.textContent = "FIELD WORK COMPLETE";
    primary.dataset.action = "none";
    primary.disabled = true;
  } else {
    primary.textContent = "INCIDENT REVEALED";
    primary.dataset.action = "none";
    primary.disabled = true;
  }

  reveal.disabled = !data.can_reveal;
  if (state.busy) setBusy(true);
}

function renderMission(data) {
  const body = $("mission-body");
  const plannerChip = $("planner-chip");
  const mission = data.mission;
  if (!mission) {
    plannerChip.textContent = "NO PLAN";
    body.className = "mission-body empty-state";
    body.innerHTML = data.can_reveal
      ? "Budget exhausted.<br><strong>Evaluator gate available.</strong>"
      : "No field allocation yet.<br><span>Generate a mission from the current belief state.</span>";
    return;
  }

  plannerChip.textContent = mission.planner.toUpperCase();
  body.className = "mission-body";
  body.innerHTML = mission.allocations.map((a, idx) => {
    const node = data.nodes.find(n => n.id === a.site_id);
    const effortBlocks = Array.from({ length: a.effort_units }, () => "<i></i>").join("");
    return `
      <div class="dispatch-row">
        <div class="dispatch-index">T${idx + 1}</div>
        <div class="dispatch-main">
          <div class="dispatch-site"><b>${a.site_id}</b><span>${node?.label || "Survey site"}</span></div>
          <div class="dispatch-meta">${missionReasonText(mission, a.site_id).toUpperCase()}</div>
        </div>
        <div class="dispatch-effort"><strong>${a.effort_units}</strong><span>checks</span><div class="effort-blocks">${effortBlocks}</div></div>
      </div>`;
  }).join("");
}

function renderEvidence(data) {
  const body = $("evidence-body");
  const last = data.last_round;
  if (!last) {
    body.className = "evidence-body empty-state";
    body.innerHTML = "Awaiting first field return.<br><span>Evidence will update belief, never hidden truth.</span>";
    return;
  }

  body.className = "evidence-body";
  body.innerHTML = last.observations.map(obs => {
    const delta = obs.belief_after - obs.belief_before;
    const deltaPct = Math.round(Math.abs(delta) * 100);
    const outcome = obs.detection ? "DETECTION" : "NO DETECTION";
    const direction = delta > 0 ? "+" : "−";
    const node = data.nodes.find(n => n.id === obs.site_id);
    return `
      <div class="evidence-item ${obs.detection ? "positive" : "negative"}">
        <div class="evidence-head">
          <div><b>${obs.site_id}</b><span>${node?.label || ""}</span></div>
          <strong>${outcome}</strong>
        </div>
        <div class="evidence-equation">
          <span class="belief-old">${fmtBelief(obs.belief_before)}</span>
          <span class="evidence-arrow">→</span>
          <span class="belief-new">${fmtBelief(obs.belief_after)}</span>
          <em>${direction}${deltaPct} pp</em>
        </div>
        <div class="evidence-foot">${obs.effort} checks · q-aware Bayesian update</div>
      </div>`;
  }).join("");
}

function renderCausalLoop(data) {
  const body = $("causal-body");
  if (!data.last_round) {
    body.innerHTML = `
      <div class="causal-step muted"><span>01</span><b>Field evidence</b><em>pending</em></div>
      <div class="causal-arrow">↓</div>
      <div class="causal-step muted"><span>02</span><b>Belief changed</b><em>pending</em></div>
      <div class="causal-arrow">↓</div>
      <div class="causal-step muted"><span>03</span><b>Mission changed</b><em>pending</em></div>`;
    return;
  }

  const observations = data.last_round.observations;
  const evidenceText = observations.map(o => `${shortSite(o.site_id)} ${o.detection ? "DET" : "0/" + o.effort}`).join(" · ");
  const deltaText = observations.map(o => `${shortSite(o.site_id)} ${fmtBelief(o.belief_before)}→${fmtBelief(o.belief_after)}`).join(" · ");
  const missionText = data.mission_changed && data.replan?.to
    ? `${missionTargets(data.replan.from).map(shortSite).join("+")} → ${missionTargets(data.replan.to).map(shortSite).join("+")}`
    : "mission stable under current evidence";

  body.innerHTML = `
    <div class="causal-step active"><span>01</span><b>Field evidence</b><em>${evidenceText}</em></div>
    <div class="causal-arrow active">↓</div>
    <div class="causal-step active"><span>02</span><b>Belief changed</b><em>${deltaText}</em></div>
    <div class="causal-arrow ${data.mission_changed ? "active" : ""}">↓</div>
    <div class="causal-step ${data.mission_changed ? "changed" : "active"}"><span>03</span><b>${data.mission_changed ? "Mission changed" : "Mission evaluated"}</b><em>${missionText}</em></div>`;
}

function renderTimeline(events) {
  const timeline = $("timeline");
  const recent = events.slice(-7);
  timeline.innerHTML = recent.map((event, index) => `
    <div class="tape-event kind-${event.kind}" style="--delay:${index * 35}ms">
      <div class="tape-index">R${event.round}</div>
      <div class="tape-copy"><b>${event.title}</b><span>${event.detail}</span></div>
    </div>`).join("");
  timeline.scrollLeft = timeline.scrollWidth;
}

function renderReplanCallout(data) {
  const callout = $("replan-callout");
  if (!data.mission_changed || !data.replan?.from || !data.replan?.to || data.revealed) {
    callout.classList.add("hidden");
    return;
  }
  const from = missionTargets(data.replan.from).map(shortSite).join(" + ");
  const to = missionTargets(data.replan.to).map(shortSite).join(" + ");
  $("replan-route").textContent = `${from}  →  ${to}`;
  $("replan-reason").textContent = "New field evidence changed occupancy belief; effort was physically reallocated.";
  callout.classList.remove("hidden");
  callout.classList.remove("flash");
  requestAnimationFrame(() => callout.classList.add("flash"));
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function addDefs(svg) {
  const defs = svgEl("defs");
  defs.innerHTML = `
    <linearGradient id="waterGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#08211f"/>
      <stop offset="100%" stop-color="#041513"/>
    </linearGradient>
    <linearGradient id="landGradient" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#183029"/>
      <stop offset="100%" stop-color="#101f1b"/>
    </linearGradient>
    <filter id="softGlow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="cloudBlur" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="13"/>
    </filter>
    <marker id="replanArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#76f0c8"/>
    </marker>`;
  svg.appendChild(defs);
}

function smoothPath(points) {
  if (points.length < 2) return "";
  let d = `M ${points[0][0]} ${points[0][1]}`;
  for (let i = 0; i < points.length - 1; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[i + 1];
    const mx = (x0 + x1) / 2;
    const my = (y0 + y1) / 2;
    d += ` Q ${x0} ${y0} ${mx} ${my}`;
    if (i === points.length - 2) d += ` T ${x1} ${y1}`;
  }
  return d;
}

function shortestPath(start, goal, edges) {
  if (start === goal) return [start];
  const adj = {};
  edges.forEach(e => {
    (adj[e.src] ||= []).push(e.dst);
    (adj[e.dst] ||= []).push(e.src);
  });
  const queue = [[start]];
  const seen = new Set([start]);
  while (queue.length) {
    const path = queue.shift();
    const tail = path[path.length - 1];
    for (const next of adj[tail] || []) {
      if (seen.has(next)) continue;
      const candidate = [...path, next];
      if (next === goal) return candidate;
      seen.add(next);
      queue.push(candidate);
    }
  }
  return [];
}

function edgeKey(a, b) {
  return [a, b].sort().join("|");
}

function renderGraph(data, previous) {
  const svg = $("graph");
  svg.innerHTML = "";
  addDefs(svg);

  const width = 1100, height = 650;
  const nodes = data.nodes;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = x => 72 + ((x - minX) / Math.max(1e-6, maxX - minX)) * 940;
  const sy = y => 82 + ((y - minY) / Math.max(1e-6, maxY - minY)) * 490;
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const prevById = Object.fromEntries((previous?.nodes || []).map(n => [n.id, n]));
  const obsById = Object.fromEntries((data.last_round?.observations || []).map(o => [o.site_id, o]));

  svg.appendChild(svgEl("rect", { x: 0, y: 0, width, height, fill: "url(#waterGradient)", class: "water-field" }));

  // Build a schematic land edge from the coastal chain and carve a harbor basin back into it.
  const coastNodes = nodes.filter(n => n.zone === "coast").sort((a, b) => a.id.localeCompare(b.id));
  const coastPoints = coastNodes.map(n => [sx(n.x), sy(n.y)]);
  const coastD = smoothPath(coastPoints);
  const reversed = [...coastPoints].reverse().map(([x, y]) => `${x},${Math.max(0, y - 22)}`).join(" L ");
  const land = svgEl("path", {
    d: `M 0 0 L ${width} 0 L ${width} ${Math.max(20, coastPoints.at(-1)[1] - 22)} L ${reversed} L 0 ${Math.max(20, coastPoints[0][1] - 22)} Z`,
    class: "landmass",
  });
  svg.appendChild(land);

  // The M5 harbor-bay basin was a decorative path hand-fit to that demo's
  // fixed toy x/y coordinates (~0-10 range). Real site coordinates are
  // locally-projected km around the incident median (real_incident_source.py)
  // and have no corresponding real inlet, so redrawing it here would place an
  // arbitrary, meaningless shape rather than any real geography - dropped
  // rather than ported.
  const coast = svgEl("path", { d: coastD, class: "coastline" });
  svg.appendChild(coast);
  for (const offset of [26, 52, 82]) {
    const contour = svgEl("path", { d: coastD, class: "depth-contour" });
    contour.setAttribute("transform", `translate(0 ${offset})`);
    svg.appendChild(contour);
  }

  const current1 = svgEl("path", {
    d: `M 120 545 C 330 485, 470 610, 650 535 S 940 480, 1040 555`,
    class: "current-line current-a",
  });
  const current2 = svgEl("path", {
    d: `M 180 585 C 380 525, 520 635, 735 570 S 930 545, 1010 605`,
    class: "current-line current-b",
  });
  svg.append(current1, current2);

  const routeEdges = new Set();
  missionTargets(data.mission).forEach(target => {
    const path = shortestPath(data.incident.initial_detection, target, data.edges);
    for (let i = 0; i < path.length - 1; i++) routeEdges.add(edgeKey(path[i], path[i + 1]));
  });

  data.edges.forEach(edge => {
    const a = byId[edge.src], b = byId[edge.dst];
    if (!a || !b) return;
    const zones = new Set([a.zone, b.zone]);
    const kind = zones.has("offshore") ? "offshore" : zones.has("harbor") ? "harbor" : "coast";
    const line = svgEl("line", {
      x1: sx(a.x), y1: sy(a.y), x2: sx(b.x), y2: sy(b.y),
      class: `graph-edge edge-${kind} ${routeEdges.has(edgeKey(edge.src, edge.dst)) ? "mission-path" : ""}`,
    });
    svg.appendChild(line);
  });

  // Reallocation vectors make the policy change physically visible on the map.
  if (data.mission_changed && data.replan?.from && data.replan?.to) {
    const from = missionTargets(data.replan.from);
    const to = missionTargets(data.replan.to);
    to.forEach((target, idx) => {
      const source = from[idx % Math.max(1, from.length)];
      const a = byId[source], b = byId[target];
      if (!a || !b) return;
      const x1 = sx(a.x), y1 = sy(a.y), x2 = sx(b.x), y2 = sy(b.y);
      const cx = (x1 + x2) / 2;
      const cy = Math.min(y1, y2) - 52 - idx * 12;
      const p = svgEl("path", {
        d: `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`,
        class: "replan-vector",
        "marker-end": "url(#replanArrow)",
      });
      svg.appendChild(p);
    });
  }

  nodes.forEach(node => {
    const cx = sx(node.x), cy = sy(node.y);
    const group = svgEl("g", { "data-site": node.id, class: `node-group zone-${node.zone}` });
    group.style.cursor = "pointer";

    const cloud = svgEl("circle", {
      cx, cy,
      r: 26 + node.uncertainty * 30,
      class: `uncertainty-cloud ${node.detections > 0 ? "positive" : ""}`,
    });
    group.appendChild(cloud);

    if (node.frontier) {
      group.appendChild(svgEl("circle", { cx, cy, r: 26, class: "frontier-ring" }));
    }

    const wasMission = data.last_round?.previous_mission?.allocations?.some(a => a.site_id === node.id);
    if (wasMission && data.mission_changed && !node.mission_effort) {
      group.appendChild(svgEl("circle", { cx, cy, r: 31, class: "previous-mission-ring" }));
    }

    if (node.mission_effort > 0) {
      group.appendChild(svgEl("circle", { cx, cy, r: 34, class: "node-mission-pulse" }));
      group.appendChild(svgEl("circle", { cx, cy, r: 25, class: "node-mission-halo" }));
    }

    const circumference = 2 * Math.PI * 18;
    group.appendChild(svgEl("circle", { cx, cy, r: 18, class: "node-ring-bg" }));
    const ring = svgEl("circle", { cx, cy, r: 18, class: "node-belief-ring" });
    const prevBelief = prevById[node.id]?.belief ?? node.belief;
    ring.style.strokeDasharray = `${circumference * prevBelief} ${circumference}`;
    group.appendChild(ring);
    requestAnimationFrame(() => {
      ring.style.strokeDasharray = `${circumference * node.belief} ${circumference}`;
    });

    const coreClasses = ["node-core"];
    if (node.status === "confirmed_detection" || node.detections > 0) coreClasses.push("confirmed");
    else if (node.effort > 0) coreClasses.push("surveyed");
    if (node.mission_effort > 0) coreClasses.push("mission");
    if (data.revealed && node.true_occupied) coreClasses.push("true-occupied");
    group.appendChild(svgEl("circle", { cx, cy, r: 11, class: coreClasses.join(" ") }));

    const idLabel = svgEl("text", { x: cx, y: cy + 37, "text-anchor": "middle", class: "node-id" });
    idLabel.textContent = shortSite(node.id);
    group.appendChild(idLabel);

    const pLabel = svgEl("text", { x: cx, y: cy - 31, "text-anchor": "middle", class: "node-belief-label" });
    pLabel.textContent = fmtBelief(node.belief);
    group.appendChild(pLabel);

    if (node.mission_effort > 0) {
      const tag = svgEl("g", { class: "mission-tag" });
      const rect = svgEl("rect", { x: cx + 16, y: cy - 28, width: 42, height: 19, rx: 3 });
      const text = svgEl("text", { x: cx + 37, y: cy - 15, "text-anchor": "middle" });
      text.textContent = `${node.mission_effort}×`;
      tag.append(rect, text);
      group.appendChild(tag);
    }

    const obs = obsById[node.id];
    if (obs) {
      const badge = svgEl("g", { class: `evidence-badge ${obs.detection ? "positive" : "negative"}` });
      const w = obs.detection ? 70 : 58;
      badge.appendChild(svgEl("rect", { x: cx - w / 2, y: cy + 44, width: w, height: 21, rx: 3 }));
      const t = svgEl("text", { x: cx, y: cy + 59, "text-anchor": "middle" });
      t.textContent = obs.detection ? `DET / ${obs.effort}` : `0 / ${obs.effort}`;
      badge.appendChild(t);
      group.appendChild(badge);

      const delta = svgEl("text", { x: cx, y: cy + 81, "text-anchor": "middle", class: "belief-delta-label" });
      delta.textContent = `${fmtBelief(obs.belief_before)} → ${fmtBelief(obs.belief_after)}`;
      group.appendChild(delta);
    }

    group.addEventListener("mouseenter", e => showTooltip(e, node, data.revealed));
    group.addEventListener("mousemove", moveTooltip);
    group.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(group);
  });
}

function showTooltip(event, node, revealed) {
  const tip = $("node-tooltip");
  const truth = revealed ? `<div class="tip-truth">True occupancy: <b>${node.true_occupied ? "PRESENT" : "ABSENT"}</b></div>` : "";
  tip.innerHTML = `
    <div class="tip-zone">${node.zone.toUpperCase()}</div>
    <strong>${node.id}</strong>
    <span>${node.label}</span>
    <div class="tip-grid"><span>Belief</span><b>${fmtBelief(node.belief)}</b><span>Uncertainty</span><b>${fmtBelief(node.uncertainty)}</b><span>Effort</span><b>${node.effort}</b><span>Detections</span><b>${node.detections}</b></div>
    ${truth}`;
  tip.classList.remove("hidden");
  moveTooltip(event);
}

function moveTooltip(event) {
  const wrap = document.querySelector(".map-stage");
  const rect = wrap.getBoundingClientRect();
  const tip = $("node-tooltip");
  tip.style.left = `${Math.min(rect.width - 230, event.clientX - rect.left + 14)}px`;
  tip.style.top = `${Math.min(rect.height - 190, event.clientY - rect.top + 14)}px`;
}

function hideTooltip() {
  $("node-tooltip").classList.add("hidden");
}

async function refresh() {
  render(await api("/api/state"));
}

$("primary-btn").addEventListener("click", async () => {
  const action = $("primary-btn").dataset.action;
  if (!action || action === "none" || state.busy) return;
  const before = state.data;
  try {
    setBusy(true);
    if (action === "plan") {
      showTransition("DECIDE", "Mission generated from current belief", "Selecting feasible survey sites under the remaining field budget.");
      const next = await api("/api/plan", "POST");
      await sleep(420);
      render(next, before);
      $("transition-title").textContent = "Field teams assigned";
      $("transition-detail").textContent = missionTargetCopy(next, next.mission);
      await sleep(650);
      hideTransition();
    } else if (action === "execute") {
      showTransition("SURVEY", "Field teams deployed", missionTargetCopy(before, before.mission));
      await sleep(420);
      const next = await api("/api/execute", "POST");
      const obs = next.last_round?.observations || [];
      const obsCopy = obs.map(o => `${shortSite(o.site_id)}: ${o.detection ? "DETECTION" : `0/${o.effort}`}`).join("  /  ");
      showTransition("FIELD RETURN", "Evidence received", obsCopy, obs.some(o => o.detection) ? "amber" : "mint");
      await sleep(560);
      render(next, before);
      const beliefCopy = obs.map(o => `${shortSite(o.site_id)} ${fmtBelief(o.belief_before)}→${fmtBelief(o.belief_after)}`).join("  /  ");
      showTransition("INFER", "Bayesian belief updated", beliefCopy);
      await sleep(650);
      if (next.mission_changed && next.mission) {
        const from = missionTargets(next.replan.from).map(shortSite).join(" + ");
        const to = missionTargets(next.replan.to).map(shortSite).join(" + ");
        showTransition("REPLAN", "MISSION UPDATED", `${from}  →  ${to} · field effort reallocated`, "mint");
        await sleep(920);
      }
      hideTransition();
    }
  } catch (error) {
    hideTransition();
    alert(error.message);
  } finally {
    setBusy(false);
    renderControls(state.data);
  }
});

$("reveal-btn").addEventListener("click", async () => {
  if (state.busy) return;
  const before = state.data;
  try {
    setBusy(true);
    showTransition("EVALUATOR GATE", "Revealing hidden extent", "Only now does the latent incident become visible to compare belief and mission history.", "amber");
    const next = await api("/api/reveal", "POST");
    await sleep(650);
    render(next, before);
    $("transition-title").textContent = "TRUE EXTENT REVEALED";
    $("transition-detail").textContent = "Evaluator-only view. The planner never received this information.";
    await sleep(900);
    hideTransition();
  } catch (error) {
    hideTransition();
    alert(error.message);
  } finally {
    setBusy(false);
    renderControls(state.data);
  }
});

$("reset-btn").addEventListener("click", async () => {
  if (state.busy) return;
  try {
    setBusy(true);
    const next = await api("/api/reset", "POST", { seed: null });
    render(next, state.data);
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
    renderControls(state.data);
  }
});

refresh().catch(error => {
  console.error(error);
  alert(`Mission Control failed to initialize: ${error.message}`);
});
