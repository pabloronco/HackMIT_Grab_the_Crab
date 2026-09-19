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
  if (value == null) return "—";
  const pp = Math.round(value * 100);
  return `${pp >= 0 ? "+" : ""}${pp} pp`;
};
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

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
  renderControls();
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
  return state.data?.top_worlds?.items?.find((world) => world.world_id === worldId) || null;
}

function recommendationBySite(siteId) {
  return state.data?.global_recommendations?.find((row) => row.site_id === siteId) || null;
}

function nodeById(siteId) {
  return state.data?.nodes?.find((node) => node.id === siteId) || null;
}

function ensureSelection() {
  const recommendations = state.data?.global_recommendations || [];
  const known = new Set(state.data?.nodes?.map((node) => node.id) || []);
  if (!state.selectedSite || !known.has(state.selectedSite) || state.selectedSite === state.data?.incident?.initial_detection) {
    state.selectedSite = recommendations[0]?.site_id || null;
  }
  if (state.selectedWorld && !worldById(state.selectedWorld)) {
    state.selectedWorld = null;
  }

  const allowed = state.data?.resources?.effort_levels || [1, 3, 6];
  const affordable = allowed.filter((effort) => effort <= (state.data?.resources?.remaining_budget || 0));
  if (!affordable.includes(state.selectedEffort)) {
    state.selectedEffort = affordable.at(-1) || allowed[0];
  }
}

async function bootstrap() {
  try {
    state.cases = await api("/api/cases");
    renderCaseSelect();
    const data = await api("/api/state");
    render(data, null);
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

function renderCaseSelect() {
  const rows = state.cases?.cases || [];
  $("case-select").innerHTML = rows.map((row) => {
    const demo = row.case_id === "incident_097" ? "★ DEMO · " : "";
    return `<option value="${row.case_id}">${demo}${String(row.index).padStart(3, "0")} · ${row.label}</option>`;
  }).join("");
}

function render(data, previous = state.data) {
  state.data = data;
  ensureSelection();

  if (data.case?.case_id) $("case-select").value = data.case.case_id;
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
  $("budget-fill").style.width = `${Math.round(clamp01(budgetRatio) * 100)}%`;

  renderMissionPanel();
  renderWorlds();
  renderQ();
  renderWhyMission();
  renderMap(previous);
  renderEvidence();
  renderPropagation();
  renderStress();
  renderReplan();
  renderLiveChart();
  renderPerformance();
  renderTimeline();
  renderControls();
}

function renderMissionPanel() {
  const recommendations = state.data.global_recommendations || [];
  const top = recommendations[0] || null;

  $("marine-site").textContent = top ? `Site ${top.site_id}` : "Field work complete";
  $("marine-belief").textContent = top ? fmtPct(top.belief) : "—";
  $("marine-priority").textContent = top
    ? (top.frontier ? "Frontier priority" : "Fallback priority")
    : "Complete";
  $("marine-priority").classList.toggle("pill-green", Boolean(top?.frontier));
  $("marine-copy").textContent = top
    ? "Frontier-first ranking: frontier status, then occupancy belief, uncertainty, and deterministic site ID."
    : "No further Marine recommendation is required before reveal.";

  const staticSites = state.data.static_response?.plan_sites || [];
  $("static-site").textContent = staticSites.length ? `Site ${staticSites[0]}` : "—";
  $("static-route").innerHTML = staticSites.map((siteId, index) => {
    const arrow = index < staticSites.length - 1 ? '<i>→</i>' : "";
    return `<span class="route-chip"><b>${siteId}</b>${arrow}</span>`;
  }).join("");

  const selectable = (state.data.nodes || [])
    .filter((node) => node.id !== state.data.incident.initial_detection)
    .slice()
    .sort((a, b) => {
      const ar = a.marine_rank ?? 9999;
      const br = b.marine_rank ?? 9999;
      return ar - br || String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
    });

  $("site-select").innerHTML = selectable.map((node) => {
    const rank = node.marine_rank ? ` · Marine #${node.marine_rank}` : "";
    const effort = node.effort > 0 ? " · surveyed" : "";
    return `<option value="${node.id}" ${node.id === state.selectedSite ? "selected" : ""}>Site ${node.id}${rank}${effort}</option>`;
  }).join("");

  const selected = nodeById(state.selectedSite);
  const predictive = selected?.predictive_detection?.[String(state.selectedEffort)];
  $("effort-caption").textContent = selected
    ? `At Site ${selected.id}, posterior-predictive detection at effort ${state.selectedEffort}: ${fmtPct(predictive)}.`
    : "Higher effort increases the chance of detecting an occupied site.";

  $("follow-marine-btn").disabled = !top || top.site_id === state.selectedSite;
}

function renderWorlds() {
  const bundle = state.data.top_worlds;
  $("world-count").textContent = `${bundle.unique_world_count} extents`;

  const expandedId = state.selectedWorld || bundle.items?.[0]?.world_id || null;
  $("world-list").innerHTML = bundle.items.map((world) => {
    const selected = expandedId === world.world_id;
    const family = world.families.length ? world.families.join(" · ") : "mixed provenance";
    const delta = Math.abs(world.delta) < 0.0005 ? "stable" : fmtPp(world.delta);
    const changeText = Math.abs(world.delta) < 0.0005
      ? "Posterior mass is stable after the latest field return."
      : `Posterior mass moved ${fmtPp(world.delta)} after the latest field return.`;
    return `
      <div class="world-card ${selected ? "selected" : ""}" data-world="${world.world_id}" role="button" tabindex="0">
        <span class="world-rank">${world.rank}</span>
        <span class="world-main">
          <strong>Possible extent ${world.rank}</strong>
          <span>${world.occupied_count} occupied sites · ${family}</span>
        </span>
        <span class="world-mass"><b>${fmtPct(world.posterior)}</b><small>${delta}</small></span>
        ${selected ? `
          <div class="world-expanded">
            <b>Why it remains plausible</b>
            <ul>
              <li>${world.occupied_count} sites are occupied in this hypothesis.</li>
              <li>Ecological provenance: ${family}.</li>
              <li>${changeText}</li>
            </ul>
          </div>` : ""}
      </div>`;
  }).join("") + `
    <div class="other-worlds">
      <span>All other unique extents</span><b>${fmtPct(bundle.remaining_mass)}</b>
    </div>`;

  document.querySelectorAll("[data-world]").forEach((button) => {
    const toggle = () => {
      const worldId = button.dataset.world;
      state.selectedWorld = state.selectedWorld === worldId ? null : worldId;
      render(state.data);
    };
    button.addEventListener("click", toggle);
    button.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle();
      }
    });
  });

  $("clear-world-btn").classList.toggle("hidden", !state.selectedWorld);
  $("scenario-banner").classList.toggle("hidden", !state.selectedWorld);
  if (state.selectedWorld) {
    const world = worldById(state.selectedWorld);
    $("scenario-banner").textContent =
      `WHAT-IF EXTENT #${world.rank} · posterior ${fmtPct(world.posterior)} · this is not hidden truth`;
  }
}

function renderQ() {
  const posterior = state.data.q_posterior || {};
  const rows = Object.entries(posterior).sort(([a], [b]) => Number(a) - Number(b));
  const maxWeight = Math.max(0.0001, ...rows.map(([, weight]) => Number(weight)));
  $("q-panel").innerHTML = rows.map(([q, weight]) => {
    const height = Math.max(3, Math.round((Number(weight) / maxWeight) * 100));
    return `
      <div class="q-column">
        <b>${fmtPct(weight)}</b>
        <div class="q-bar-shell"><i class="q-bar" style="height:${height}%"></i></div>
        <span>q = ${Number(q).toFixed(2)}</span>
      </div>`;
  }).join("") + `<div class="q-mean">posterior mean q = <b>${Number(state.data.q_mean).toFixed(3)}</b></div>`;
}

function renderWhyMission() {
  const node = nodeById(state.selectedSite);
  if (!node) {
    $("why-subtitle").textContent = "Choose a survey site to inspect the current evidence.";
    $("why-frontier-value").textContent = "—";
    $("why-belief-value").textContent = "—";
    $("why-detection-value").textContent = "—";
    return;
  }

  const rankText = node.marine_rank ? `Marine rank #${node.marine_rank}` : "Operator override";
  $("why-subtitle").textContent = `Site ${node.id} · ${rankText} · current observable state`;
  $("why-frontier-value").textContent = node.frontier ? "YES" : "NO";
  $("why-frontier-copy").textContent = node.frontier
    ? "This site sits on the current response frontier beside public detection evidence."
    : "This site is not currently a frontier node; select it only as an operator override or fallback.";
  $("why-belief-value").textContent = fmtPct(node.belief);
  const predictive = node.predictive_detection?.[String(state.selectedEffort)];
  $("why-detection-value").textContent = fmtPct(predictive);
  $("why-detection-copy").textContent =
    `Posterior-predictive probability at effort ${state.selectedEffort}; it marginalizes detectability uncertainty.`;
}

function activeWorldOccupancy() {
  const world = worldById(state.selectedWorld);
  return new Set(world?.occupied_site_ids || []);
}

function nodeLayerValue(node) {
  if (state.selectedWorld) return activeWorldOccupancy().has(node.id) ? 1 : 0;
  if (state.layer === "habitat") return clamp01(node.habitat);
  if (state.layer === "history") return Math.min(1, (node.effort || 0) / 12);
  return clamp01(node.belief);
}

function mercatorWorld(lon, lat, zoom) {
  const scale = 256 * (2 ** zoom);
  const x = ((Number(lon) + 180) / 360) * scale;
  const sin = Math.sin(Number(lat) * Math.PI / 180);
  const y = (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * scale;
  return { x, y };
}

function chooseMapProjection(nodes, width, height) {
  const geo = nodes.filter((node) => Number.isFinite(Number(node.latitude)) && Number.isFinite(Number(node.longitude)));
  if (geo.length < 2) {
    const xs = nodes.map((node) => Number(node.x));
    const ys = nodes.map((node) => Number(node.y));
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    return {
      geo: false,
      project: (node) => ({
        x: 70 + ((Number(node.x) - minX) / Math.max(1e-9, maxX - minX)) * (width - 140),
        y: height - 65 - ((Number(node.y) - minY) / Math.max(1e-9, maxY - minY)) * (height - 130),
      }),
    };
  }

  let chosen = null;
  for (let zoom = 13; zoom >= 5; zoom -= 1) {
    const pixels = geo.map((node) => mercatorWorld(node.longitude, node.latitude, zoom));
    const xs = pixels.map((p) => p.x), ys = pixels.map((p) => p.y);
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    if (spanX <= width - 170 && spanY <= height - 140) {
      chosen = { zoom, pixels, minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
      break;
    }
  }
  if (!chosen) {
    const zoom = 5;
    const pixels = geo.map((node) => mercatorWorld(node.longitude, node.latitude, zoom));
    const xs = pixels.map((p) => p.x), ys = pixels.map((p) => p.y);
    chosen = { zoom, pixels, minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }

  const centerX = (chosen.minX + chosen.maxX) / 2;
  const centerY = (chosen.minY + chosen.maxY) / 2;
  const originX = centerX - width / 2;
  const originY = centerY - height / 2;

  return {
    geo: true,
    zoom: chosen.zoom,
    originX,
    originY,
    project: (node) => {
      const p = mercatorWorld(node.longitude, node.latitude, chosen.zoom);
      return { x: p.x - originX, y: p.y - originY };
    },
  };
}

function appendMapTiles(svg, projection, width, height) {
  svg.appendChild(svgEl("rect", { x:0, y:0, width, height, class:"map-fallback-water" }));
  if (!projection.geo) return;

  const z = projection.zoom;
  const tileSize = 256;
  const minTileX = Math.floor(projection.originX / tileSize) - 1;
  const maxTileX = Math.floor((projection.originX + width) / tileSize) + 1;
  const minTileY = Math.floor(projection.originY / tileSize) - 1;
  const maxTileY = Math.floor((projection.originY + height) / tileSize) + 1;
  const maxIndex = (2 ** z) - 1;

  for (let tx = minTileX; tx <= maxTileX; tx += 1) {
    for (let ty = minTileY; ty <= maxTileY; ty += 1) {
      if (ty < 0 || ty > maxIndex) continue;
      const wrappedX = ((tx % (maxIndex + 1)) + (maxIndex + 1)) % (maxIndex + 1);
      const image = svgEl("image", {
        href: `https://tile.openstreetmap.org/${z}/${wrappedX}/${ty}.png`,
        x: tx * tileSize - projection.originX,
        y: ty * tileSize - projection.originY,
        width: tileSize,
        height: tileSize,
        class: "map-tile",
        preserveAspectRatio: "none",
      });
      image.addEventListener("error", () => image.remove());
      svg.appendChild(image);
    }
  }
  svg.appendChild(svgEl("rect", { x:0, y:0, width, height, class:"map-tile-fade" }));
}

function beliefColor(value) {
  const v = clamp01(value);
  if (v < 0.25) return "#2877d6";
  if (v < 0.5) return "#32b7d5";
  if (v < 0.72) return "#f4d94c";
  return "#ef4048";
}

function renderMap(previous) {
  const svg = $("graph");
  svg.innerHTML = "";
  const width = 1100, height = 620;
  const nodes = state.data.nodes || [];
  if (!nodes.length) return;

  const projection = chooseMapProjection(nodes, width, height);
  const pointById = Object.fromEntries(nodes.map((node) => [node.id, projection.project(node)]));
  const nodeBy = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const prevBy = Object.fromEntries((previous?.nodes || []).map((node) => [node.id, node]));

  const defs = svgEl("defs");
  defs.innerHTML = `
    <filter id="heatBlur" x="-120%" y="-120%" width="340%" height="340%">
      <feGaussianBlur stdDeviation="18"/>
    </filter>
    <marker id="replanArrow" viewBox="0 0 10 10" refX="8.2" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#0a7ed1"></path>
    </marker>`;
  svg.appendChild(defs);

  appendMapTiles(svg, projection, width, height);

  if (state.layer !== "history" || state.selectedWorld) {
    nodes.forEach((node) => {
      const p = pointById[node.id];
      const value = nodeLayerValue(node);
      if (value <= 0.015) return;
      const radius = state.selectedWorld ? 48 : 34 + 58 * value;
      const halo = svgEl("circle", {
        cx:p.x, cy:p.y, r:radius,
        fill:beliefColor(value),
        opacity: state.selectedWorld ? .28 : (.10 + .24 * value),
        class:"heat-halo",
        filter:"url(#heatBlur)",
      });
      svg.appendChild(halo);
    });
  }

  (state.data.edges || []).forEach((edge) => {
    const a = pointById[edge.src], b = pointById[edge.dst];
    if (!a || !b) return;
    svg.appendChild(svgEl("line", { x1:a.x, y1:a.y, x2:b.x, y2:b.y, class:"graph-edge" }));
  });

  if (state.data.mission_changed && state.data.replan?.from && state.data.replan?.to) {
    const from = state.data.replan.from.allocations?.[0]?.site_id;
    const to = state.data.replan.to.allocations?.[0]?.site_id;
    const a = pointById[from], b = pointById[to];
    if (a && b) {
      svg.appendChild(svgEl("line", {
        x1:a.x, y1:a.y, x2:b.x, y2:b.y,
        class:"replan-vector", "marker-end":"url(#replanArrow)",
      }));
    }
  }

  const staticNext = state.data.static_response?.plan_sites?.[Math.min(state.data.resources.round, 2)] || null;
  const topSite = state.data.global_recommendations?.[0]?.site_id || null;

  nodes.forEach((node) => {
    const p = pointById[node.id];
    const value = nodeLayerValue(node);
    const group = svgEl("g", { class:"node-group", "data-site":node.id });

    if (node.frontier && !state.selectedWorld) {
      group.appendChild(svgEl("circle", { cx:p.x, cy:p.y, r:22, class:"frontier-ring" }));
    }
    if (state.selectedSite === node.id) {
      group.appendChild(svgEl("circle", { cx:p.x, cy:p.y, r:27, class:"selected-ring" }));
    }
    if (topSite === node.id && !state.data.can_reveal && !state.data.revealed) {
      group.appendChild(svgEl("circle", { cx:p.x, cy:p.y, r:32, class:"marine-ring" }));
    }

    const coreClass = [
      "node-core",
      node.status === "confirmed_detection" ? "confirmed" : "",
      node.status === "detected" || node.detections > 1 ? "detected" : "",
      node.effort > 0 ? "surveyed" : "",
      state.data.revealed && node.true_occupied ? "true-occupied" : "",
      state.data.revealed && node.true_occupied && node.detections === 0 ? "true-missed" : "",
    ].filter(Boolean).join(" ");
    group.appendChild(svgEl("circle", { cx:p.x, cy:p.y, r:12, class:coreClass }));

    const prevBelief = prevBy[node.id]?.belief ?? node.belief;
    if (!state.selectedWorld && state.layer === "probability" && Math.abs(prevBelief - node.belief) > 0.015) {
      const delta = svgEl("text", { x:p.x, y:p.y - 25, "text-anchor":"middle", class:"belief-delta" });
      delta.textContent = `${fmtPct(prevBelief)}→${fmtPct(node.belief)}`;
      group.appendChild(delta);
    }

    const important = (
      node.id === state.data.incident.initial_detection ||
      node.id === topSite ||
      node.id === state.selectedSite ||
      node.id === staticNext ||
      node.detections > 0
    );
    if (important) {
      const labelText = `Site ${node.id}`;
      const labelWidth = 48 + String(node.id).length * 5;
      const lx = p.x + 17, ly = p.y - 12;
      group.appendChild(svgEl("rect", {
        x:lx - 5, y:ly - 12, width:labelWidth, height:22, rx:5, class:"node-label-bg"
      }));
      const label = svgEl("text", { x:lx, y:ly + 3, class:"node-label" });
      label.textContent = labelText;
      group.appendChild(label);
    }

    group.addEventListener("click", () => {
      if (node.id === state.data.incident.initial_detection) return;
      state.selectedSite = node.id;
      render(state.data);
    });
    group.addEventListener("mouseenter", (event) => showTooltip(event, node));
    group.addEventListener("mousemove", moveTooltip);
    group.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(group);
  });

  const title = state.selectedWorld
    ? `Possible extent #${worldById(state.selectedWorld)?.rank || "?"}`
    : state.layer === "probability"
      ? "Posterior occupancy belief"
      : state.layer === "habitat"
        ? "Real-data habitat context"
        : "Observed field effort";
  $("map-title").textContent = title;
}

function showTooltip(event, node) {
  const tip = $("node-tooltip");
  const predictive = node.predictive_detection?.[String(state.selectedEffort)];
  tip.innerHTML = `
    <strong>Site ${node.id}</strong>
    <div class="tip-grid">
      <span>Occupancy belief</span><b>${fmtPct(node.belief)}</b>
      <span>Habitat proxy</span><b>${fmtPct(node.habitat)}</b>
      <span>Habitat label</span><b>${node.habitat_label || "—"}</b>
      <span>Observed effort</span><b>${node.effort}</b>
      <span>Detections</span><b>${node.detections}</b>
      <span>Marine rank</span><b>${node.marine_rank ? "#" + node.marine_rank : "—"}</b>
      <span>P(detect) @ e${state.selectedEffort}</span><b>${fmtPct(predictive)}</b>
      <span>Coordinates</span><b>${Number(node.latitude).toFixed(3)}, ${Number(node.longitude).toFixed(3)}</b>
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
  tip.style.left = `${Math.max(6, Math.min(rect.width - 240, event.clientX - rect.left + 14))}px`;
  tip.style.top = `${Math.max(6, Math.min(rect.height - 205, event.clientY - rect.top + 14))}px`;
}

function hideTooltip() {
  $("node-tooltip").classList.add("hidden");
}

function renderEvidence() {
  const panel = $("evidence-panel");
  const last = state.data.last_round;
  if (!last) {
    panel.className = "evidence-content muted";
    panel.textContent = "Awaiting field evidence.";
    return;
  }
  panel.className = "evidence-content";
  panel.innerHTML = last.observations.map((obs) => `
    <div class="evidence-line ${obs.detection ? "positive" : ""}">
      <div><b>Site ${obs.site_id}</b><span>${obs.detection ? "DETECTION" : "NO DETECTION"} · effort ${obs.effort}</span></div>
      <strong>${fmtPct(obs.belief_before)} → ${fmtPct(obs.belief_after)}</strong>
    </div>`
  ).join("");
}

function renderPropagation() {
  const panel = $("propagation-panel");
  const rows = state.data.last_round?.propagated_belief_changes || [];
  if (!rows.length) {
    panel.className = "evidence-content muted";
    panel.textContent = "No propagated update yet.";
    return;
  }
  panel.className = "evidence-content";
  panel.innerHTML = rows.slice(0, 3).map((row) => `
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
    panel.className = "evidence-content muted";
    panel.textContent = "No field result yet.";
    return;
  }
  panel.className = "evidence-content";
  const impossible = stress.impossible_under_current_ensemble;
  panel.innerHTML = `
    <div class="stress-value ${impossible ? "warning" : ""}">
      <span>Observed-result probability</span>
      <strong>${impossible ? "0%" : fmtPct(stress.observation_probability)}</strong>
    </div>
    <p>${impossible ? "Outside current ensemble support." : "Posterior-predictive fit of the result."}</p>
    <small>${stress.surprise_bits == null ? "Infinite surprise" : stress.surprise_bits.toFixed(2) + " bits surprise"}</small>
  `;
}

function renderReplan() {
  const callout = $("replan-callout");
  if (!state.data.mission_changed || !state.data.replan?.from || !state.data.replan?.to || state.data.revealed) {
    callout.classList.add("hidden");
    return;
  }
  const from = state.data.replan.from.allocations?.[0]?.site_id || "—";
  const to = state.data.replan.to.allocations?.[0]?.site_id || "—";
  $("replan-route").textContent = `Without result: Site ${from}  →  With evidence: Site ${to}`;
  callout.classList.remove("hidden");
}

function renderLiveChart() {
  const svg = $("live-chart");
  svg.innerHTML = "";
  const rows = state.data.live_curve || [];
  const width = 980, height = 250;
  const margin = { left:50, right:52, top:24, bottom:38 };
  const xMax = Math.max(18, state.data.resources.initial_budget || 18);
  const maxDetections = Math.max(3, ...rows.map((row) => Number(row.field_detections || 0))) + 1;
  const sx = (x) => margin.left + (Number(x) / xMax) * (width - margin.left - margin.right);
  const syLeft = (y) => height - margin.bottom - (Number(y) / maxDetections) * (height - margin.top - margin.bottom);
  const syRight = (fraction) => height - margin.bottom - clamp01(fraction) * (height - margin.top - margin.bottom);

  for (let i = 0; i <= 4; i += 1) {
    const fraction = i / 4;
    const y = height - margin.bottom - fraction * (height - margin.top - margin.bottom);
    svg.appendChild(svgEl("line", { x1:margin.left, x2:width-margin.right, y1:y, y2:y, class:"chart-grid" }));
    const leftLabel = svgEl("text", { x:margin.left-9, y:y+3, "text-anchor":"end", class:"chart-label" });
    leftLabel.textContent = String(Math.round(maxDetections * fraction));
    svg.appendChild(leftLabel);
    const rightLabel = svgEl("text", { x:width-margin.right+9, y:y+3, class:"chart-label" });
    rightLabel.textContent = `${Math.round(fraction * 100)}%`;
    svg.appendChild(rightLabel);
  }

  [0, 6, 12, 18].forEach((effort) => {
    const x = sx(effort);
    const label = svgEl("text", { x, y:height-13, "text-anchor":"middle", class:"chart-label" });
    label.textContent = String(effort);
    svg.appendChild(label);
  });

  const effortLabel = svgEl("text", { x:width/2, y:height-1, "text-anchor":"middle", class:"chart-title-label" });
  effortLabel.textContent = "Cumulative field effort";
  svg.appendChild(effortLabel);

  const leftTitle = svgEl("text", { x:margin.left, y:12, class:"chart-title-label" });
  leftTitle.textContent = "Confirmed detections";
  svg.appendChild(leftTitle);
  const rightTitle = svgEl("text", { x:width-margin.right, y:12, "text-anchor":"end", class:"chart-title-label" });
  rightTitle.textContent = "Budget used";
  svg.appendChild(rightTitle);

  const detectionPoints = rows.map((row) => [sx(row.effort), syLeft(row.field_detections)]);
  const budgetPoints = rows.map((row) => [sx(row.effort), syRight(row.budget_used_fraction)]);
  const path = (points) => points.map((point, index) => `${index === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");

  if (detectionPoints.length) {
    svg.appendChild(svgEl("path", { d:path(detectionPoints), class:"live-detection-line" }));
    detectionPoints.forEach(([x,y]) => svg.appendChild(svgEl("circle", { cx:x, cy:y, r:4, class:"live-dot-detection" })));
  }
  if (budgetPoints.length) {
    svg.appendChild(svgEl("path", { d:path(budgetPoints), class:"live-budget-line" }));
    budgetPoints.forEach(([x,y]) => svg.appendChild(svgEl("circle", { cx:x, cy:y, r:4, class:"live-dot-budget" })));
  }
}

function renderPerformance() {
  const content = $("performance-content");
  const locked = $("performance-locked");
  if (!state.data.performance) {
    content.classList.add("hidden");
    locked.classList.remove("hidden");
    const copy = locked.querySelector("small");
    if (copy) {
      copy.textContent = state.data.can_reveal
        ? "Field campaign complete. Reveal the hidden extent to score all three tracks."
        : "Complete the available field budget to open the evaluator receipt.";
    }
    return;
  }

  locked.classList.add("hidden");
  content.classList.remove("hidden");
  const p = state.data.performance;
  const entries = [["MARINE", p.marine], ["STATIC RESPONSE", p.static], ["YOU", p.you]];
  $("scorecards").innerHTML = entries.map(([label, row]) => `
    <div class="scorecard">
      <span>${label}</span>
      <strong>${row.detected_occupied} / ${row.occupied_total}</strong>
      <small>true occupied sites confirmed / detected</small>
    </div>`
  ).join("");
  drawPerformanceChart(entries);
}

function drawPerformanceChart(entries) {
  const svg = $("performance-chart");
  svg.innerHTML = "";
  const width = 900, height = 250;
  const margin = { left:58, right:30, top:24, bottom:40 };
  const maxEffort = Math.max(18, ...entries.flatMap(([, row]) => row.curve.map((point) => point.effort)));
  const sx = (x) => margin.left + (x / maxEffort) * (width - margin.left - margin.right);
  const sy = (y) => height - margin.bottom - y * (height - margin.top - margin.bottom);

  for (let i = 0; i <= 4; i += 1) {
    const yValue = i / 4;
    const y = sy(yValue);
    svg.appendChild(svgEl("line", { x1:margin.left, x2:width-margin.right, y1:y, y2:y, class:"chart-grid" }));
    const label = svgEl("text", { x:margin.left-10, y:y+3, "text-anchor":"end", class:"chart-label" });
    label.textContent = `${Math.round(yValue*100)}%`;
    svg.appendChild(label);
  }

  [0,6,12,18].forEach((effort) => {
    const label = svgEl("text", { x:sx(effort), y:height-13, "text-anchor":"middle", class:"chart-label" });
    label.textContent = String(effort);
    svg.appendChild(label);
  });

  entries.forEach(([label, row], index) => {
    const points = row.curve.map((point) => [sx(point.effort), sy(point.detected_fraction)]);
    const d = points.map((point, i) => `${i === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");
    svg.appendChild(svgEl("path", { d, class:`chart-line series-${index}` }));
    points.forEach(([x,y]) => svg.appendChild(svgEl("circle", { cx:x, cy:y, r:4, class:`chart-dot series-${index}` })));
    const last = points.at(-1);
    if (last) {
      const text = svgEl("text", { x:last[0]-5, y:last[1]-9, "text-anchor":"end", class:`chart-series-label series-${index}` });
      text.textContent = label;
      svg.appendChild(text);
    }
  });
}

function renderTimeline() {
  const recent = (state.data.events || []).slice(-9);
  $("timeline").innerHTML = recent.map((event) => `
    <div class="tape-event kind-${event.kind}">
      <span>R${event.round}</span>
      <div><b>${event.title}</b><small>${event.detail}</small></div>
    </div>`
  ).join("");
  $("timeline").scrollLeft = $("timeline").scrollWidth;
}

function renderEffortControls() {
  document.querySelectorAll(".effort-btn").forEach((button) => {
    const effort = Number(button.dataset.effort);
    button.classList.toggle("active", effort === state.selectedEffort);
    button.disabled = !canSpend(effort) || state.data.revealed || state.data.can_reveal;
  });
}

function renderControls() {
  if (!state.data) return;
  renderEffortControls();
  const done = state.data.can_reveal || state.data.revealed;
  const deploy = $("deploy-btn");
  deploy.disabled = state.busy || !state.selectedSite || !canSpend(state.selectedEffort) || done;
  const label = deploy.querySelector("b");
  const sub = deploy.querySelector("small");
  if (label) label.textContent = done ? "Field Work Complete" : `Deploy Site ${state.selectedSite || "—"}`;
  if (sub) sub.textContent = done ? "Evaluator reveal is now available" : `Effort ${state.selectedEffort} · send field team and update beliefs`;

  $("reset-btn").disabled = state.busy;
  $("next-case-btn").disabled = state.busy;
  $("case-select").disabled = state.busy;
  $("site-select").disabled = state.busy || done;
  $("reveal-btn").disabled = state.busy || !state.data.can_reveal;
}

async function resetCase(caseId = null) {
  try {
    setBusy(true);
    state.selectedSite = null;
    state.selectedWorld = null;
    showTransition("NEW INCIDENT", "Initializing coastal response", "Loading the real monitoring network and a blinded hidden ecological extent.");
    const data = await api("/api/reset", "POST", { case_id:caseId || $("case-select").value });
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
      `Surveying Site ${state.selectedSite}`,
      `Allocating ${state.selectedEffort} effort units. Hidden truth remains locked.`
    );
    const data = await api("/api/deploy", "POST", {
      site_id:state.selectedSite,
      effort:state.selectedEffort,
    });

    const obs = data.last_round?.observations?.[0];
    if (obs) {
      $("transition-kicker").textContent = "FIELD RETURN";
      $("transition-title").textContent = obs.detection
        ? `Detection at Site ${obs.site_id}`
        : `No detection at Site ${obs.site_id}`;
      $("transition-detail").textContent =
        `Effort ${obs.effort} · local belief ${fmtPct(obs.belief_before)} → ${fmtPct(obs.belief_after)} · posterior recomputed across the graph.`;
    }

    await new Promise((resolve) => setTimeout(resolve, 440));
    state.selectedSite = data.global_recommendations?.[0]?.site_id || state.selectedSite;
    render(data, before);
    setTimeout(hideTransition, 560);
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
    showTransition(
      "EVALUATE",
      "Revealing the hidden extent",
      "Marine, the precommitted Static Response, and your choices are scored on the same blinded incident."
    );
    const before = state.data;
    const data = await api("/api/reveal", "POST");
    await new Promise((resolve) => setTimeout(resolve, 380));
    render(data, before);
    $("transition-kicker").textContent = "TRUE EXTENT REVEALED";
    $("transition-title").textContent = "Same incident. Same budget. Different decisions.";
    $("transition-detail").textContent = "The evaluator now exposes simulator-only occupancy that remained unavailable to every planner and operator action.";
    setTimeout(hideTransition, 1000);
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
  const top = state.data.global_recommendations?.[0];
  if (!top) return;
  state.selectedSite = top.site_id;
  render(state.data);
});

$("site-select").addEventListener("change", () => {
  state.selectedSite = $("site-select").value;
  render(state.data);
});

document.querySelectorAll(".effort-btn").forEach((button) => {
  button.addEventListener("click", () => {
    const effort = Number(button.dataset.effort);
    if (!canSpend(effort)) return;
    state.selectedEffort = effort;
    render(state.data);
  });
});

document.querySelectorAll(".layer-btn").forEach((button) => {
  button.addEventListener("click", () => {
    state.layer = button.dataset.layer;
    state.selectedWorld = null;
    document.querySelectorAll(".layer-btn").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
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
  const current = rows.findIndex((row) => row.case_id === state.data?.case?.case_id);
  const next = rows[(current + 1 + rows.length) % rows.length];
  resetCase(next.case_id);
});

bootstrap();
