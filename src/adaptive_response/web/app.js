const state = {
  data: null,
  cases: null,
  selectedSite: null,
  selectedWorld: null,
  selectedEffort: 6,
  layer: "belief",
  busy: false,
  mapZoom: 0,
  mapPanX: 0,
  mapPanY: 0,
  drag: null,
  mapFrame: null,
  zoomPreviewScale: 1,
  zoomPreviewAnchorX: 550,
  zoomPreviewAnchorY: 320,
  zoomCommitTimer: null,
};

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));
const fmtPct = (value, digits = 0) => value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
const fmtPp = (value) => {
  if (value == null) return "—";
  const pp = Math.round(Number(value) * 100);
  return `${pp >= 0 ? "+" : ""}${pp} pp`;
};
const fmtNum = (value, digits = 1) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);

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

function nodeById(siteId) {
  return state.data?.nodes?.find((node) => node.id === siteId) || null;
}

function recommendationBySite(siteId) {
  return state.data?.global_recommendations?.find((row) => row.site_id === siteId) || null;
}

function worldById(worldId) {
  return state.data?.top_worlds?.items?.find((world) => world.world_id === worldId) || null;
}

function canSpend(effort) {
  return Boolean(
    state.data &&
    Number(effort) > 0 &&
    Number(effort) <= Number(state.data.resources.remaining_budget)
  );
}

function effortOptionFor(siteId, effort) {
  const rec = recommendationBySite(siteId);
  const options = rec?.effort_recommendation?.options || nodeById(siteId)?.effort_recommendation?.options || [];
  return options.find((row) => Number(row.effort) === Number(effort)) || null;
}

function suggestedEffortForSite(siteId) {
  return recommendationBySite(siteId)?.recommended_effort ?? nodeById(siteId)?.recommended_effort ?? null;
}

function applyRecommendedEffort(siteId) {
  const recommended = suggestedEffortForSite(siteId);
  if (recommended && canSpend(recommended)) state.selectedEffort = Number(recommended);
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

function setBusy(value) {
  state.busy = value;
  document.body.classList.toggle("is-busy", value);
  renderControls();
}

function ensureSelection() {
  const recommendations = state.data?.global_recommendations || [];
  const valid = new Set((state.data?.nodes || []).map((node) => node.id));
  const initial = state.data?.incident?.initial_detection;

  if (!state.selectedSite || !valid.has(state.selectedSite) || state.selectedSite === initial) {
    state.selectedSite = recommendations[0]?.site_id || null;
    if (state.selectedSite) applyRecommendedEffort(state.selectedSite);
  }

  if (state.selectedWorld && !worldById(state.selectedWorld)) state.selectedWorld = null;

  const allowed = state.data?.resources?.effort_levels || [1, 3, 6];
  const affordable = allowed.filter((effort) => canSpend(effort));
  if (!affordable.includes(state.selectedEffort)) {
    state.selectedEffort = affordable.at(-1) || allowed[0] || 1;
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
  $("case-select").innerHTML = (state.cases?.cases || []).map((row) => {
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

  const horizon = data.resources.mission_horizon || 3;
  $("round").textContent = `${Math.min(data.resources.round, horizon)} / ${horizon}`;
  $("budget-left").textContent = `${data.resources.remaining_budget} / ${data.resources.initial_budget}`;
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
  renderEvidence();
  renderDecisionReceipt();
  renderPropagation();
  renderStress();
  renderResourceCard();
  renderMap(previous);
  renderReplan();
  renderLiveCharts();
  renderPerformance();
  renderTimeline();
  renderControls();
}

function renderMissionPanel() {
  const top = state.data.global_recommendations?.[0] || null;
  $("marine-site").textContent = top ? `Site ${top.site_id}` : "Complete";
  $("marine-effort").textContent = top?.recommended_effort ?? "—";
  $("marine-belief").textContent = top ? fmtPct(top.belief) : "—";
  $("marine-pdetect").textContent = top
    ? fmtPct(top.predictive_detection?.[String(top.recommended_effort)])
    : "—";
  $("marine-saved").textContent = top
    ? `${top.effort_recommendation?.effort_saved_vs_max ?? 0} units`
    : `${state.data.resources.capacity_preserved ?? 0} units`;
  $("marine-rank-badge").textContent = top ? "#1 current" : "complete";

  $("marine-copy").textContent = top
    ? "Site priority stays Frontier-first. Effort is chosen separately from the Bayesian occupancy/q posterior to preserve field capacity when extra checks add limited value."
    : "Three-deployment response window complete. Any unused effort remains preserved capacity.";

  const staticSites = state.data.static_response?.plan_sites || [];
  $("static-site").textContent = staticSites.length ? `Site ${staticSites[0]}` : "—";
  $("static-route").innerHTML = staticSites.map((siteId, index) => {
    const arrow = index < staticSites.length - 1 ? "<i>→</i>" : "";
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
    const rank = node.marine_rank ? ` · Recommended #${node.marine_rank}` : "";
    const surveyed = node.effort > 0 ? " · surveyed" : "";
    return `<option value="${node.id}" ${node.id === state.selectedSite ? "selected" : ""}>Site ${node.id}${rank}${surveyed}</option>`;
  }).join("");

  const selected = nodeById(state.selectedSite);
  const rec = recommendationBySite(state.selectedSite);
  const recommendedEffort = rec?.recommended_effort ?? selected?.recommended_effort;
  $("effort-rec-badge").textContent = recommendedEffort ? `Suggested: e${recommendedEffort}` : "Suggested: —";

  document.querySelectorAll(".effort-btn").forEach((button) => {
    const effort = Number(button.dataset.effort);
    button.classList.toggle("active", effort === state.selectedEffort);
    button.classList.toggle("recommended", effort === Number(recommendedEffort));
  });

  const options = rec?.effort_recommendation?.options || selected?.effort_recommendation?.options || [];
  $("effort-options").innerHTML = options.map((row) => {
    const chosen = Number(row.effort) === Number(recommendedEffort);
    return `<div class="effort-option ${chosen ? "chosen" : ""}">
      <b>e${row.effort}</b>
      <span>Pdetect ${fmtPct(row.predictive_detection_probability)}</span>
      <span>Info ${fmtNum(row.expected_information_gain_bits, 2)} bits</span>
    </div>`;
  }).join("");

  const predictive = selected?.predictive_detection?.[String(state.selectedEffort)];
  const power = effortOptionFor(state.selectedSite, state.selectedEffort)?.conditional_detection_if_occupied;
  $("effort-caption").textContent = selected
    ? `At Site ${selected.id}: P(detection next survey)=${fmtPct(predictive)}; if the site is truly occupied, detection power at e${state.selectedEffort}=${fmtPct(power)}.`
    : "";

  $("follow-marine-btn").disabled = !top || (
    state.selectedSite === top.site_id &&
    Number(state.selectedEffort) === Number(top.recommended_effort)
  );
}

function renderWorlds() {
  const bundle = state.data.top_worlds;
  $("world-count").textContent = `${bundle.unique_world_count} extents`;
  const expanded = state.selectedWorld || bundle.items?.[0]?.world_id || null;

  $("world-list").innerHTML = bundle.items.map((world) => {
    const selected = state.selectedWorld === world.world_id;
    const isExpanded = world.world_id === expanded;
    const family = world.families.length ? world.families.join(" · ") : "mixed provenance";
    const delta = Math.abs(world.delta) < 0.0005 ? "stable" : fmtPp(world.delta);
    return `<button class="world-card ${selected ? "selected" : ""}" data-world="${world.world_id}">
      <span class="world-rank">${world.rank}</span>
      <span class="world-main">
        <strong>Possible extent ${world.rank}</strong>
        <span>${world.occupied_count} occupied sites · ${family}</span>
      </span>
      <span class="world-mass"><b>${fmtPct(world.posterior)}</b><small>${delta}</small></span>
      ${isExpanded ? `<span class="world-expanded"><b>${selected ? "What-if view active" : "Highest-posterior extent"} — not hidden truth.</b><br>
        ${world.occupied_count} sites are occupied in this extent. Latest evidence changed its posterior mass by ${fmtPp(world.delta)}.</span>` : ""}
    </button>`;
  }).join("") + `<div class="other-worlds"><span>All other unique extents</span><b>${fmtPct(bundle.remaining_mass)}</b></div>`;

  document.querySelectorAll("[data-world]").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.dataset.world;
      state.selectedWorld = state.selectedWorld === id ? null : id;
      render(state.data);
    });
  });

  $("clear-world-btn").classList.toggle("hidden", !state.selectedWorld);
  $("scenario-banner").classList.toggle("hidden", !state.selectedWorld);
  if (state.selectedWorld) {
    const world = worldById(state.selectedWorld);
    $("scenario-banner").textContent = `WHAT-IF EXTENT #${world.rank} · posterior ${fmtPct(world.posterior)} · NOT HIDDEN TRUTH`;
  }
}

function renderQ() {
  const posterior = state.data.q_posterior || {};
  const rows = Object.entries(posterior).sort(([a], [b]) => Number(a) - Number(b));
  const maxWeight = Math.max(0.0001, ...rows.map(([, weight]) => Number(weight)));

  $("q-panel").innerHTML = rows.map(([q, weight]) => {
    const height = Math.max(3, Math.round((Number(weight) / maxWeight) * 100));
    return `<div class="q-column">
      <b>${fmtPct(weight)}</b>
      <div class="q-bar-shell"><i class="q-bar" style="height:${height}%"></i></div>
      <span>q=${Number(q).toFixed(2)}</span>
    </div>`;
  }).join("") + `<div class="q-mean">posterior mean q = <b>${Number(state.data.q_mean).toFixed(3)}</b></div>`;

  const qValues = rows.map(([q]) => Number(q));
  const qLow = qValues.length ? Math.min(...qValues) : 0;
  const qHigh = qValues.length ? Math.max(...qValues) : 0;
  const e = Number(state.selectedEffort || 1);
  const lowPower = 1 - ((1 - qLow) ** e);
  const highPower = 1 - ((1 - qHigh) ** e);
  const selectedOption = effortOptionFor(state.selectedSite, e);
  const posteriorPower = selectedOption?.conditional_detection_if_occupied;

  $("q-effort-preview").innerHTML = `At selected effort <b>e${e}</b>, if a site is occupied, detection power ranges from
    <b>${fmtPct(lowPower)}</b> to <b>${fmtPct(highPower)}</b> across the tested q support.
    ${posteriorPower == null ? "" : `For the selected site, the current posterior-weighted value is <b>${fmtPct(posteriorPower)}</b>.`}`;

  const qd = state.data.q_diagnostics;
  const edge = qd?.dominant_edge === "high" ? "upper" : "lower";
  const pressure = qd?.boundary_pressure ?? qd?.dominant_edge_mass ?? 0;
  $("q-boundary-copy").innerHTML = `Boundary pressure: <b>${fmtPct(pressure)}</b> on the ${edge} edge of the tested q support.
    This can mean q support is narrow or q and occupancy remain confounded. It does <b>not</b> automatically mean model stress is high.
    Model stress is the separate posterior-predictive surprise of the actual field return.
    <div class="boundary-meter"><div class="boundary-meter-row"><span>Interior</span><span>Edge pressure ${fmtPct(pressure)}</span></div><div class="boundary-track"><i style="width:${Math.round(clamp01(pressure) * 100)}%"></i></div></div>`;
}

function renderWhyMission() {
  const node = nodeById(state.selectedSite);
  if (!node) return;

  const rec = recommendationBySite(node.id);
  const recommendedEffort = rec?.recommended_effort ?? node.recommended_effort;
  const option = effortOptionFor(node.id, state.selectedEffort);
  const predictive = node.predictive_detection?.[String(state.selectedEffort)];
  const power = option?.conditional_detection_if_occupied;
  const rankText = node.marine_rank ? `recommendation rank #${node.marine_rank}` : "operator override";
  const distance = node.distance_from_detection_km == null ? "distance unavailable" : `${fmtNum(node.distance_from_detection_km, 1)} km from first detection`;

  $("why-subtitle").textContent = `Site ${node.id} · ${rankText} · ${distance}`;
  $("why-frontier-value").textContent = node.frontier ? "YES" : "NO";
  $("why-frontier-copy").textContent = node.frontier
    ? "This site sits on the current response frontier defined by public detection evidence."
    : "This site is not currently on the response frontier; selecting it is an operator override or fallback.";

  $("why-belief-value").textContent = fmtPct(node.belief);
  $("why-belief-copy").textContent = "Belief is posterior mass over ecological extents, not a distance-decay score. A nearby site can legitimately have lower belief than a farther site.";

  $("why-power-value").textContent = fmtPct(power);
  $("why-power-copy").textContent = `If this site is truly occupied, this is the chance of at least one detection with effort ${state.selectedEffort}, integrating q uncertainty.`;

  $("why-detection-value").textContent = fmtPct(predictive);
  $("why-detection-copy").textContent = "Predictive detection combines both uncertainties: whether the site is occupied and whether the protocol detects it if occupied.";

  $("why-effort-value").textContent = recommendedEffort ? `e${recommendedEffort}` : "—";
  const effortDiag = rec?.effort_recommendation ?? node.effort_recommendation;
  if (effortDiag) {
    $("why-effort-copy").textContent = `Occupancy band: ${effortDiag.occupancy_band}. The planner chooses the smallest effort retaining at least ${fmtPct(effortDiag.information_retention_required)} of max-effort information value and ${fmtPct(effortDiag.detection_power_retention_required)} of max-effort detection power for this occupancy band. This is a transparent product rule, not an ecological constant.`;
  } else {
    $("why-effort-copy").textContent = "No effort recommendation after campaign completion.";
  }
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
  panel.innerHTML = last.observations.map((obs) => `<div class="evidence-line ${obs.detection ? "positive" : ""}">
    <div><b>Site ${obs.site_id}</b><span>${obs.detection ? "DETECTION" : "NO DETECTION"} · effort ${obs.effort}</span></div>
    <strong>${fmtPct(obs.belief_before)} → ${fmtPct(obs.belief_after)}</strong>
  </div>`).join("");
}

function renderDecisionReceipt() {
  const panel = $("decision-receipt");
  const receipt = state.data.decision_receipt;
  if (!receipt) {
    panel.className = "evidence-content muted";
    panel.textContent = "No mission executed yet.";
    return;
  }

  panel.className = "evidence-content";
  const status = receipt.marine_aligned ? "Recommendation-aligned before outcome" : "Operator override";
  const miss = !receipt.detection && receipt.conditional_miss_if_occupied != null
    ? ` If occupied, the selected effort still had a ${fmtPct(receipt.conditional_miss_if_occupied)} miss probability.`
    : "";
  panel.innerHTML = `<div class="receipt-head"><b>${status}</b><span>${receipt.detection ? "detection" : "no detection"}</span></div>
    <p>${receipt.interpretation}${miss}</p>`;
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
  panel.innerHTML = rows.slice(0, 3).map((row) => `<div class="prop-row">
    <b>Site ${row.site_id}</b><span>${fmtPct(row.belief_before)} → ${fmtPct(row.belief_after)}</span><strong>${fmtPp(row.delta)}</strong>
  </div>`).join("");
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
  panel.innerHTML = `<div class="stress-value ${stress.impossible_under_current_ensemble ? "warning" : ""}">
    <span>P(observed result)</span><strong>${fmtPct(stress.observation_probability)}</strong>
  </div>
  <p>${stress.impossible_under_current_ensemble ? "Outside current ensemble support." : "How surprising the observed result was under the pre-update posterior."}</p>
  <small>${stress.surprise_bits == null ? "∞" : Number(stress.surprise_bits).toFixed(2)} bits surprise</small>`;
}

function renderResourceCard() {
  const r = state.data.resource_summary;
  const initial = Number(state.data.resources.initial_budget || 18);
  const remaining = Number(r.capacity_preserved ?? r.budget_remaining ?? 0);
  const fraction = initial ? remaining / initial : 0;

  $("resource-spent").textContent = `${r.effort_spent} / ${initial}`;
  $("resource-left").textContent = remaining;
  $("resource-missions").textContent = `${r.missions_completed} / ${r.mission_horizon || 3}`;
  $("resource-avoided").textContent = r.effort_avoided_vs_always_high_for_completed_missions;
  $("capacity-percent").textContent = fmtPct(fraction);
  $("capacity-ring").style.setProperty("--capacity-angle", `${Math.round(clamp01(fraction) * 360)}deg`);

  renderConversion();
}

function renderConversion() {
  const avoided = Number(state.data?.resource_summary?.effort_avoided_vs_always_high_for_completed_missions || 0);
  const minutes = Number($("minutes-per-unit").value || 0);
  const cost = Number($("cost-per-unit").value || 0);
  const pieces = [];
  if (minutes > 0) pieces.push(`${Math.round(avoided * minutes)} minutes of field work`);
  if (cost > 0) pieces.push(`$${(avoided * cost).toFixed(0)} operational cost`);
  $("conversion-output").textContent = pieces.length
    ? `At your local rates, the currently avoided effort corresponds to approximately ${pieces.join(" and ")}.`
    : "Enter local rates to estimate operational savings. Values are operator-supplied and are not used by the ecological model.";
}

function ordinalValue(raw) {
  if (!raw) return null;
  const value = String(raw).trim().toUpperCase();
  const map = {
    ABSENT: 0,
    NONE: 0,
    PATCHY: 0.5,
    SPARSE: 0.35,
    CONTINUOUS: 1,
    PRESENT: 1,
    EXPOSED: 1,
    "SEMI-EXPOSED": 0.75,
    "SEMI EXPOSED": 0.75,
    PROTECTED: 0.35,
    "VERY PROTECTED": 0.1,
  };
  return Object.prototype.hasOwnProperty.call(map, value) ? map[value] : null;
}

function activeWorldOccupancy() {
  const world = worldById(state.selectedWorld);
  return new Set(world?.occupied_site_ids || []);
}

function layerDescriptor() {
  const nodes = state.data.nodes || [];

  if (state.selectedWorld) {
    const occupied = activeWorldOccupancy();
    return {
      title: "Possible extent occupancy",
      low: "Not occupied",
      high: "Occupied",
      provenance: "What-if posterior extent; not hidden truth",
      value: (node) => occupied.has(node.id) ? 1 : 0,
      display: (node) => occupied.has(node.id) ? "present" : "absent",
    };
  }

  if (state.layer === "uncertainty") return {
    title: "Occupancy uncertainty",
    low: "Low",
    high: "High",
    provenance: "Posterior Bernoulli entropy at monitoring sites",
    value: (node) => clamp01(node.uncertainty),
    display: (node) => fmtPct(node.uncertainty),
  };

  if (state.layer === "habitat") return {
    title: "Habitat suitability",
    low: "Lower",
    high: "Higher",
    provenance: "Real-data-constrained site habitat proxy",
    value: (node) => clamp01(node.habitat),
    display: (node) => fmtPct(node.habitat),
  };

  if (state.layer === "temperature") {
    const vals = nodes
      .map((node) => node.temperature_median_c)
      .filter((value) => value != null)
      .map(Number);
    if (!vals.length) return {
      title: "Historical logger temperature",
      low: "No data",
      high: "No data",
      provenance: "No direct logger rows for this incident subgraph; no imputation is performed.",
      value: () => null,
      display: () => "no data",
    };
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    return {
      title: "Historical logger temperature",
      low: `${min.toFixed(1)}°C`,
      high: `${max.toFixed(1)}°C`,
      provenance: `Direct Crab Team logger summaries at ${vals.length}/${nodes.length} incident sites; median by site; missing sites are not imputed.`,
      value: (node) => node.temperature_median_c == null
        ? null
        : (Number(node.temperature_median_c) - min) / Math.max(0.001, max - min),
      display: (node) => node.temperature_median_c == null ? "no data" : `${fmtNum(node.temperature_median_c, 1)}°C`,
    };
  }

  if (state.layer === "exposure") return {
    title: "Shoreline exposure",
    low: "Protected",
    high: "Exposed",
    provenance: "Documented site exposure category; categorical values are mapped only for display.",
    value: (node) => ordinalValue(node.exposure),
    display: (node) => node.exposure || "no data",
  };

  if (state.layer === "eelgrass") return {
    title: "Eelgrass",
    low: "Absent",
    high: "Continuous",
    provenance: "Documented site eelgrass category; categorical values are mapped only for display.",
    value: (node) => ordinalValue(node.eelgrass),
    display: (node) => node.eelgrass || "no data",
  };

  if (state.layer === "salt_marsh") return {
    title: "Salt marsh",
    low: "Absent",
    high: "Continuous",
    provenance: "Documented site salt-marsh category; categorical values are mapped only for display.",
    value: (node) => ordinalValue(node.salt_marsh),
    display: (node) => node.salt_marsh || "no data",
  };

  if (state.layer === "field_effort") return {
    title: "Field effort this incident",
    low: "0",
    high: "6+",
    provenance: "Observed effort spent during the current blinded response campaign",
    value: (node) => Math.min(1, Number(node.effort || 0) / 6),
    display: (node) => `e${node.effort || 0}`,
  };

  return {
    title: "Posterior occupancy belief",
    low: "Low",
    high: "High",
    provenance: "Posterior P(occupied) from the explicit spatial Bayesian belief engine",
    value: (node) => clamp01(node.belief),
    display: (node) => fmtPct(node.belief),
  };
}

function mercatorWorld(lon, lat, zoom) {
  const scale = 256 * (2 ** zoom);
  const x = ((Number(lon) + 180) / 360) * scale;
  const sinLat = Math.sin(Number(lat) * Math.PI / 180);
  const y = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale;
  return { x, y };
}

function chooseMapProjection(nodes, width, height) {
  const geo = nodes.filter((node) =>
    Number.isFinite(Number(node.latitude)) &&
    Number.isFinite(Number(node.longitude))
  );
  if (geo.length < 2) return null;

  let baseZoom = 5;
  for (let zoom = 14; zoom >= 5; zoom -= 1) {
    const points = geo.map((node) => mercatorWorld(node.longitude, node.latitude, zoom));
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    if (
      Math.max(...xs) - Math.min(...xs) <= width - 190 &&
      Math.max(...ys) - Math.min(...ys) <= height - 150
    ) {
      baseZoom = zoom;
      break;
    }
  }

  const zoom = Math.max(4, Math.min(18, baseZoom + state.mapZoom));
  const points = geo.map((node) => mercatorWorld(node.longitude, node.latitude, zoom));
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const originX = centerX - width / 2 - state.mapPanX;
  const originY = centerY - height / 2 - state.mapPanY;

  return {
    zoom,
    originX,
    originY,
    project: (node) => {
      const point = mercatorWorld(node.longitude, node.latitude, zoom);
      return { x: point.x - originX, y: point.y - originY };
    },
  };
}

function appendMapTiles(root, projection, width, height) {
  root.appendChild(svgEl("rect", { x: 0, y: 0, width, height, class: "map-fallback-water" }));
  if (!projection) return;

  const size = 256;
  const z = projection.zoom;
  const max = (2 ** z) - 1;
  const minX = Math.floor(projection.originX / size) - 1;
  const maxX = Math.floor((projection.originX + width) / size) + 1;
  const minY = Math.floor(projection.originY / size) - 1;
  const maxY = Math.floor((projection.originY + height) / size) + 1;

  for (let tx = minX; tx <= maxX; tx += 1) {
    for (let ty = minY; ty <= maxY; ty += 1) {
      if (ty < 0 || ty > max) continue;
      const wrappedX = ((tx % (max + 1)) + (max + 1)) % (max + 1);
      const esri = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${ty}/${wrappedX}`;
      const osm = `https://tile.openstreetmap.org/${z}/${wrappedX}/${ty}.png`;
      const image = svgEl("image", {
        href: esri,
        x: tx * size - projection.originX,
        y: ty * size - projection.originY,
        width: size,
        height: size,
        class: "map-tile",
        preserveAspectRatio: "none",
      });
      image.setAttribute("data-fallback", "0");
      image.addEventListener("error", () => {
        if (image.getAttribute("data-fallback") === "0") {
          image.setAttribute("data-fallback", "1");
          image.setAttribute("href", osm);
        } else {
          image.remove();
        }
      });
      root.appendChild(image);
    }
  }

  root.appendChild(svgEl("rect", { x: 0, y: 0, width, height, class: "map-tile-fade" }));
}

function interpolateRgb(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function heatColor(value) {
  if (value == null) return "#aebbc6";
  const x = clamp01(value);
  const stops = [
    [0.00, [95, 147, 106]],
    [0.34, [200, 173, 79]],
    [0.67, [201, 119, 66]],
    [1.00, [185, 80, 80]],
  ];
  let left = stops[0];
  let right = stops[1];
  for (let i = 0; i < stops.length - 1; i += 1) {
    if (x >= stops[i][0] && x <= stops[i + 1][0]) {
      left = stops[i];
      right = stops[i + 1];
      break;
    }
  }
  const t = (x - left[0]) / Math.max(0.0001, right[0] - left[0]);
  const rgb = interpolateRgb(left[1], right[1], Math.max(0, Math.min(1, t)));
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

function renderMap(previous) {
  const svg = $("graph");
  svg.innerHTML = "";
  const width = 1100;
  const height = 640;
  const nodes = state.data.nodes || [];
  if (!nodes.length) return;

  const projection = chooseMapProjection(nodes, width, height);
  if (!projection) return;
  const points = Object.fromEntries(nodes.map((node) => [node.id, projection.project(node)]));
  const previousById = Object.fromEntries((previous?.nodes || []).map((node) => [node.id, node]));
  const descriptor = layerDescriptor();

  $("layer-legend-title").textContent = descriptor.title;
  $("legend-low").textContent = descriptor.low;
  $("legend-high").textContent = descriptor.high;
  $("layer-provenance").textContent = descriptor.provenance;
  $("map-title").textContent = descriptor.title;
  $("zoom-readout").textContent = state.mapZoom === 0
    ? "Fit"
    : `Zoom ${state.mapZoom > 0 ? "+" : ""}${state.mapZoom}`;

  const defs = svgEl("defs");
  const influenceBlur = svgEl("filter", {
    id: "influence-blur",
    x: "-35%",
    y: "-35%",
    width: "170%",
    height: "170%",
  });
  influenceBlur.appendChild(svgEl("feGaussianBlur", { stdDeviation: "13" }));
  defs.appendChild(influenceBlur);

  const heatSoften = svgEl("filter", {
    id: "heat-soften",
    x: "-40%",
    y: "-40%",
    width: "180%",
    height: "180%",
  });
  heatSoften.appendChild(svgEl("feGaussianBlur", { stdDeviation: "3.8" }));
  defs.appendChild(heatSoften);
  svg.appendChild(defs);

  const mapRoot = svgEl("g", { id: "map-root" });
  svg.appendChild(mapRoot);
  appendMapTiles(mapRoot, projection, width, height);

  const continuousInfluenceLayer = ["belief", "uncertainty", "habitat"].includes(state.layer);
  if (continuousInfluenceLayer && !state.selectedWorld) {
    (state.data.edges || []).forEach((edge, index) => {
      const aNode = nodeById(edge.src);
      const bNode = nodeById(edge.dst);
      const a = points[edge.src];
      const b = points[edge.dst];
      if (!aNode || !bNode || !a || !b) return;
      const av = descriptor.value(aNode);
      const bv = descriptor.value(bNode);
      if (av == null || bv == null) return;

      const gradientId = `influence-${index}`;
      const gradient = svgEl("linearGradient", {
        id: gradientId,
        gradientUnits: "userSpaceOnUse",
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
      });
      gradient.appendChild(svgEl("stop", {
        offset: "0%",
        "stop-color": heatColor(av),
        "stop-opacity": 0.13,
      }));
      gradient.appendChild(svgEl("stop", {
        offset: "100%",
        "stop-color": heatColor(bv),
        "stop-opacity": 0.13,
      }));
      defs.appendChild(gradient);

      const influence = clamp01((Number(av) + Number(bv)) / 2);
      mapRoot.appendChild(svgEl("line", {
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        stroke: `url(#${gradientId})`,
        "stroke-width": 22 + 18 * influence,
        opacity: 0.07 + 0.08 * influence,
        class: "heat-influence",
      }));
    });
  }

  nodes.forEach((node, index) => {
    const value = descriptor.value(node);
    if (value == null) return;
    const point = points[node.id];
    const color = heatColor(value);
    const intensity = 0.48 + 0.52 * clamp01(value);
    const gradientId = `heat-gradient-${index}`;
    const gradient = svgEl("radialGradient", { id: gradientId, cx: "50%", cy: "50%", r: "50%" });
    gradient.appendChild(svgEl("stop", { offset: "0%", "stop-color": color, "stop-opacity": 0.34 * intensity }));
    gradient.appendChild(svgEl("stop", { offset: "28%", "stop-color": color, "stop-opacity": 0.22 * intensity }));
    gradient.appendChild(svgEl("stop", { offset: "62%", "stop-color": color, "stop-opacity": 0.09 * intensity }));
    gradient.appendChild(svgEl("stop", { offset: "100%", "stop-color": color, "stop-opacity": 0 }));
    defs.appendChild(gradient);

    mapRoot.appendChild(svgEl("circle", {
      cx: point.x,
      cy: point.y,
      r: 96,
      fill: `url(#${gradientId})`,
      class: "heat-ring",
    }));
    mapRoot.appendChild(svgEl("circle", {
      cx: point.x,
      cy: point.y,
      r: 34,
      fill: color,
      opacity: 0.045 + 0.055 * intensity,
      class: "heat-core-glow",
    }));
  });

  (state.data.edges || []).forEach((edge) => {
    const a = points[edge.src];
    const b = points[edge.dst];
    if (!a || !b) return;
    mapRoot.appendChild(svgEl("line", {
      x1: a.x,
      y1: a.y,
      x2: b.x,
      y2: b.y,
      class: "graph-edge",
    }));
  });

  const topSite = state.data.global_recommendations?.[0]?.site_id || null;

  if (state.data.mission_changed && state.data.replan?.from && state.data.replan?.to) {
    const fromId = state.data.replan.from.allocations?.[0]?.site_id;
    const toId = state.data.replan.to.allocations?.[0]?.site_id;
    const a = points[fromId];
    const b = points[toId];
    if (a && b) {
      mapRoot.appendChild(svgEl("line", {
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        class: "replan-vector",
      }));
    }
  }

  nodes.forEach((node) => {
    const point = points[node.id];
    const group = svgEl("g", { class: "node-group", "data-site": node.id });
    const layerValue = descriptor.value(node);
    const layerColor = heatColor(layerValue);

    group.appendChild(svgEl("circle", { cx: point.x, cy: point.y, r: 14, class: "node-outline" }));
    if (node.frontier && !state.selectedWorld) {
      group.appendChild(svgEl("circle", { cx: point.x, cy: point.y, r: 19, class: "frontier-ring" }));
    }
    if (state.selectedSite === node.id) {
      group.appendChild(svgEl("circle", { cx: point.x, cy: point.y, r: 24, class: "selected-ring" }));
    }
    if (topSite === node.id && !state.data.revealed) {
      group.appendChild(svgEl("circle", { cx: point.x, cy: point.y, r: 29, class: "marine-ring" }));
    }

    const classes = [
      "node-core",
      layerValue == null ? "missing-layer" : "layer-colored",
      node.status === "confirmed_detection" ? "confirmed" : "",
      node.status === "detected" || node.detections > 1 ? "detected" : "",
      node.effort > 0 ? "surveyed" : "",
      state.data.revealed && node.true_occupied ? "true-occupied" : "",
      state.data.revealed && node.true_occupied && node.detections === 0 ? "true-missed" : "",
    ].filter(Boolean).join(" ");

    const core = svgEl("circle", { cx: point.x, cy: point.y, r: 8.5, class: classes });
    if (
      layerValue != null &&
      node.status !== "confirmed_detection" &&
      node.status !== "detected" &&
      !(node.detections > 1)
    ) {
      core.setAttribute("fill", layerColor);
    }
    group.appendChild(core);

    const layerLabel = svgEl("text", {
      x: point.x,
      y: point.y + 3,
      "text-anchor": "middle",
      class: "node-layer-value",
    });
    layerLabel.textContent = layerValue == null ? "—" : (
      state.mapZoom >= 2 ? descriptor.display(node).replace("%", "") : ""
    );
    group.appendChild(layerLabel);

    const prev = previousById[node.id]?.belief ?? node.belief;
    if (
      state.layer === "belief" &&
      !state.selectedWorld &&
      Math.abs(prev - node.belief) > 0.015
    ) {
      const label = svgEl("text", {
        x: point.x,
        y: point.y - 25,
        "text-anchor": "middle",
        class: "belief-delta",
      });
      label.textContent = `${fmtPct(prev)}→${fmtPct(node.belief)}`;
      group.appendChild(label);
    }

    const important = (
      node.id === state.data.incident.initial_detection ||
      node.id === topSite ||
      node.id === state.selectedSite ||
      node.detections > 0
    );
    if (important) {
      const text = `Site ${node.id}`;
      const labelWidth = 47 + String(node.id).length * 5;
      const lx = point.x + 16;
      const ly = point.y - 11;
      group.appendChild(svgEl("rect", {
        x: lx - 5,
        y: ly - 13,
        width: labelWidth,
        height: 22,
        rx: 5,
        class: "node-label-bg",
      }));
      const label = svgEl("text", { x: lx, y: ly + 2, class: "node-label" });
      label.textContent = text;
      group.appendChild(label);
    }

    group.addEventListener("click", (event) => {
      event.stopPropagation();
      if (node.id === state.data.incident.initial_detection) return;
      state.selectedSite = node.id;
      applyRecommendedEffort(node.id);
      render(state.data);
    });
    group.addEventListener("mouseenter", (event) => showTooltip(event, node));
    group.addEventListener("mousemove", moveTooltip);
    group.addEventListener("mouseleave", hideTooltip);
    mapRoot.appendChild(group);
  });

  renderMapInspector();
}

function renderMapInspector() {
  const node = nodeById(state.selectedSite) || nodeById(state.data.incident.initial_detection);
  if (!node) {
    $("map-inspector").innerHTML = "";
    return;
  }

  const predictive = node.predictive_detection?.[String(state.selectedEffort)];
  const distance = node.distance_from_detection_km;
  const nearbyLow = distance != null && Number(distance) <= 5 && Number(node.belief) < 0.25;
  const recommendation = recommendationBySite(node.id);
  const recEffort = recommendation?.recommended_effort ?? node.recommended_effort;

  $("map-inspector").innerHTML = `
    <strong>Site ${node.id}</strong>
    <div class="map-inspector-grid">
      <span>Occupancy belief</span><b>${fmtPct(node.belief)}</b>
      <span>Distance from first detection</span><b>${fmtNum(distance, 1)} km</b>
      <span>Frontier</span><b>${node.frontier ? "yes" : "no"}</b>
      <span>Recommendation rank</span><b>${node.marine_rank ? "#" + node.marine_rank : "—"}</b>
      <span>Recommended effort</span><b>${recEffort ? "e" + recEffort : "—"}</b>
      <span>P(detect) at selected effort</span><b>${fmtPct(predictive)}</b>
    </div>
    ${nearbyLow ? `<div class="nearby-note"><b>Why can a nearby site be only ${fmtPct(node.belief)}?</b><br>The planner does not use distance as probability. Belief comes from the posterior over whole ecological extents plus all field evidence. A low-belief nearby frontier can still be worth checking because it helps distinguish plausible spread patterns.</div>` : ""}
  `;
}

function showTooltip(event, node) {
  const predictive = node.predictive_detection?.[String(state.selectedEffort)];
  const option = effortOptionFor(node.id, state.selectedEffort);
  const tip = $("node-tooltip");

  tip.innerHTML = `<strong>Site ${node.id}</strong>
    <div class="tip-grid">
      <span>Occupancy belief</span><b>${fmtPct(node.belief)}</b>
      <span>Uncertainty</span><b>${fmtPct(node.uncertainty)}</b>
      <span>Distance</span><b>${fmtNum(node.distance_from_detection_km, 1)} km</b>
      <span>Habitat</span><b>${node.habitat_label || "—"}</b>
      <span>Exposure</span><b>${node.exposure || "—"}</b>
      <span>Eelgrass</span><b>${node.eelgrass || "—"}</b>
      <span>Salt marsh</span><b>${node.salt_marsh || "—"}</b>
      <span>Temperature</span><b>${node.temperature_median_c == null ? "No direct logger data" : fmtNum(node.temperature_median_c, 1) + "°C"}</b>
      <span>Recommendation rank</span><b>${node.marine_rank ? "#" + node.marine_rank : "—"}</b>
      <span>Suggested effort</span><b>${node.recommended_effort ? "e" + node.recommended_effort : "—"}</b>
      <span>P(detect)</span><b>${fmtPct(predictive)}</b>
      <span>Detection power if occupied</span><b>${fmtPct(option?.conditional_detection_if_occupied)}</b>
    </div>
    ${state.data.revealed ? `<div class="truth-line">TRUE OCCUPANCY: <b>${node.true_occupied ? "PRESENT" : "ABSENT"}</b></div>` : ""}`;

  tip.classList.remove("hidden");
  moveTooltip(event);
}

function moveTooltip(event) {
  const stage = $("map-stage");
  const rect = stage.getBoundingClientRect();
  const tip = $("node-tooltip");
  tip.style.left = `${Math.max(8, Math.min(rect.width - 255, event.clientX - rect.left + 14))}px`;
  tip.style.top = `${Math.max(8, Math.min(rect.height - 310, event.clientY - rect.top + 14))}px`;
}

function hideTooltip() {
  $("node-tooltip").classList.add("hidden");
}

function renderReplan() {
  const box = $("replan-callout");
  if (
    !state.data.mission_changed ||
    !state.data.replan?.from ||
    !state.data.replan?.to ||
    state.data.revealed
  ) {
    box.classList.add("hidden");
    return;
  }

  const from = state.data.replan.from.allocations?.[0];
  const to = state.data.replan.to.allocations?.[0];
  $("replan-route").textContent = `Without new evidence: Site ${from?.site_id} · e${from?.effort_units}  →  With evidence: Site ${to?.site_id} · e${to?.effort_units}`;
  box.classList.remove("hidden");
}

function renderLiveCharts() {
  const rows = state.data.resource_curve || [];
  const summary = state.data.resource_summary;

  $("live-kpis").innerHTML = `
    <div><small>Confirmed detections</small><b>${summary.confirmed_detections}</b></div>
    <div><small>Effort spent</small><b>${summary.effort_spent}</b></div>
    <div><small>Capacity left</small><b>${summary.capacity_preserved}</b></div>
    <div><small>Deployments</small><b>${summary.missions_completed}/${summary.mission_horizon}</b></div>`;

  drawEffortChart(rows, summary.mission_horizon || 3);
  drawDetectionChart(rows, summary.mission_horizon || 3);
}

function drawEffortChart(rows, horizon) {
  const svg = $("effort-chart");
  svg.innerHTML = "";
  const width = 560;
  const height = 250;
  const margin = { left: 42, right: 18, top: 20, bottom: 38 };
  const plotH = height - margin.top - margin.bottom;
  const sx = (round) => margin.left + ((round - 0.5) / horizon) * (width - margin.left - margin.right);
  const sy = (effort) => height - margin.bottom - (Number(effort) / 6) * plotH;

  [0, 3, 6].forEach((effort) => {
    const y = sy(effort);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "chart-grid" }));
    const label = svgEl("text", { x: margin.left - 8, y: y + 3, "text-anchor": "end", class: "chart-label" });
    label.textContent = String(effort);
    svg.appendChild(label);
  });

  for (let round = 1; round <= horizon; round += 1) {
    const x = sx(round);
    const row = rows.find((item) => Number(item.round) === round);
    const actual = Number(row?.mission_effort || 0);

    svg.appendChild(svgEl("rect", {
      x: x - 25,
      y: sy(6),
      width: 50,
      height: height - margin.bottom - sy(6),
      rx: 7,
      class: "static-effort-bar",
    }));
    if (actual > 0) {
      svg.appendChild(svgEl("rect", {
        x: x - 16,
        y: sy(actual),
        width: 32,
        height: height - margin.bottom - sy(actual),
        rx: 7,
        class: "effort-bar",
      }));
      const value = svgEl("text", { x, y: sy(actual) - 7, "text-anchor": "middle", class: "chart-title-label" });
      value.textContent = `e${actual}`;
      svg.appendChild(value);
    }

    const xLabel = svgEl("text", { x, y: height - 14, "text-anchor": "middle", class: "chart-label" });
    xLabel.textContent = `Mission ${round}`;
    svg.appendChild(xLabel);
  }

  const staticLabel = svgEl("text", { x: width - margin.right, y: sy(6) - 6, "text-anchor": "end", class: "chart-label" });
  staticLabel.textContent = "grey = static e6";
  svg.appendChild(staticLabel);
}

function drawDetectionChart(rows, horizon) {
  const svg = $("detection-chart");
  svg.innerHTML = "";
  const width = 560;
  const height = 250;
  const margin = { left: 42, right: 18, top: 20, bottom: 38 };
  const maxDetections = Math.max(3, ...rows.map((row) => Number(row.field_detections || 0))) + 1;
  const sx = (round) => margin.left + (Number(round) / horizon) * (width - margin.left - margin.right);
  const sy = (count) => height - margin.bottom - (Number(count) / maxDetections) * (height - margin.top - margin.bottom);

  for (let i = 0; i <= maxDetections; i += 1) {
    const y = sy(i);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "chart-grid" }));
    const label = svgEl("text", { x: margin.left - 8, y: y + 3, "text-anchor": "end", class: "chart-label" });
    label.textContent = String(i);
    svg.appendChild(label);
  }

  const points = rows.map((row) => [sx(row.round), sy(row.field_detections)]);
  if (points.length) {
    const d = points.map((point, index) => `${index ? "L" : "M"} ${point[0]} ${point[1]}`).join(" ");
    svg.appendChild(svgEl("path", { d, class: "live-detection-line" }));
    points.forEach(([x, y]) => svg.appendChild(svgEl("circle", { cx: x, cy: y, r: 4, class: "live-dot-detection" })));
  }

  for (let round = 0; round <= horizon; round += 1) {
    const label = svgEl("text", { x: sx(round), y: height - 14, "text-anchor": "middle", class: "chart-label" });
    label.textContent = round === 0 ? "Start" : `M${round}`;
    svg.appendChild(label);
  }
}

function renderPerformance() {
  const content = $("performance-content");
  const locked = $("performance-locked");

  if (!state.data.performance) {
    content.classList.add("hidden");
    locked.classList.remove("hidden");
    const small = locked.querySelector("small");
    if (small) {
      small.textContent = state.data.can_reveal
        ? "Three deployments complete. Reveal to compare hidden-truth outcomes."
        : `Complete ${state.data.resources.missions_remaining} more deployment(s) to unlock hidden truth.`;
    }
    return;
  }

  locked.classList.add("hidden");
  content.classList.remove("hidden");

  const performance = state.data.performance;
  const receipt = performance.resource_receipt;
  const saved = Number(receipt.effort_saved_vs_static || 0);
  const detectionDelta = Number(receipt.detected_delta_marine_minus_static || 0);

  if (saved > 0 && detectionDelta >= 0) {
    $("resource-receipt").innerHTML = `<strong>Adaptive response preserved ${saved} effort units vs Static</strong> while confirming ${detectionDelta === 0 ? "the same number of" : detectionDelta + " more"} occupied site${detectionDelta === 1 ? "" : "s"} in this blinded incident. <span>Illustrative case, not a universal claim.</span>`;
  } else if (saved > 0) {
    $("resource-receipt").innerHTML = `<strong>Adaptive response preserved ${saved} effort units vs Static</strong>, but this stochastic realization confirmed ${Math.abs(detectionDelta)} fewer occupied site${Math.abs(detectionDelta) === 1 ? "" : "s"}. This is an explicit resource/performance trade-off, not hidden by the UI.`;
  } else {
    $("resource-receipt").innerHTML = `<strong>No field-effort saving versus Static in this incident.</strong> The outcome receipt below shows the realized detection result without forcing a win.`;
  }

  const entries = [
    ["ADAPTIVE", performance.marine, "primary"],
    ["STATIC RESPONSE", performance.static, "primary"],
    ["YOUR PATH", performance.you, "secondary"],
  ];

  $("scorecards").innerHTML = entries.map(([label, row, kind]) => `<div class="scorecard ${kind}">
    <span>${label}</span>
    <strong>${row.detected_occupied} / ${row.occupied_total}</strong>
    <small>true occupied sites confirmed</small>
    <div class="score-meta">
      <span>${row.effort_spent} effort</span>
      <span>${row.capacity_preserved} preserved</span>
      <span>${row.missions_completed} deployments</span>
    </div>
  </div>`).join("");

  const you = performance.you;
  const marine = performance.marine;
  let note = performance.interpretation;
  if (you.detected_occupied > marine.detected_occupied) {
    note = `Your path happened to confirm more occupied sites in this one stochastic realization (${you.detected_occupied} vs ${marine.detected_occupied}). That does not make it a better policy: compare the pre-outcome decision receipts and aggregate case audit rather than one lucky draw.`;
  }
  $("performance-note").textContent = note;

  drawPerformanceChart([
    ["Adaptive", performance.marine],
    ["Static", performance.static],
    ["You", performance.you],
  ]);
  renderLuckReceipts(performance.marine);
}

function drawPerformanceChart(entries) {
  const svg = $("performance-chart");
  svg.innerHTML = "";
  const width = 920;
  const height = 270;
  const margin = { left: 56, right: 30, top: 25, bottom: 42 };
  const maxEffort = Math.max(18, ...entries.flatMap(([, row]) => row.curve.map((point) => point.effort)));
  const sx = (x) => margin.left + (Number(x) / maxEffort) * (width - margin.left - margin.right);
  const sy = (y) => height - margin.bottom - Number(y) * (height - margin.top - margin.bottom);

  for (let i = 0; i <= 4; i += 1) {
    const value = i / 4;
    const y = sy(value);
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "chart-grid" }));
    const label = svgEl("text", { x: margin.left - 8, y: y + 3, "text-anchor": "end", class: "chart-label" });
    label.textContent = `${Math.round(value * 100)}%`;
    svg.appendChild(label);
  }

  entries.forEach(([label, row], index) => {
    const points = row.curve.map((point) => [sx(point.effort), sy(point.detected_fraction)]);
    if (!points.length) return;
    const d = points.map((point, i) => `${i ? "L" : "M"} ${point[0]} ${point[1]}`).join(" ");
    svg.appendChild(svgEl("path", { d, class: `chart-line series-${index}` }));
    points.forEach(([x, y]) => svg.appendChild(svgEl("circle", { cx: x, cy: y, r: 4, class: `chart-dot series-${index}` })));
    const last = points.at(-1);
    if (last) {
      const text = svgEl("text", { x: last[0] - 4, y: last[1] - 9, "text-anchor": "end", class: `chart-series-label series-${index}` });
      text.textContent = label;
      svg.appendChild(text);
    }
  });

  [0, 3, 6, 9, 12, 15, 18].filter((value) => value <= maxEffort).forEach((effort) => {
    const text = svgEl("text", { x: sx(effort), y: height - 15, "text-anchor": "middle", class: "chart-label" });
    text.textContent = String(effort);
    svg.appendChild(text);
  });

  const axis = svgEl("text", { x: width / 2, y: height - 1, "text-anchor": "middle", class: "chart-title-label" });
  axis.textContent = "Cumulative field effort";
  svg.appendChild(axis);
}

function renderLuckReceipts(marine) {
  const receipts = marine?.mission_receipts || [];
  if (!receipts.length) {
    $("luck-receipts").innerHTML = "";
    return;
  }

  $("luck-receipts").innerHTML = receipts.map((receipt, index) => {
    if (receipt.realization === "occupied_but_missed") {
      return `<div class="luck-card miss"><strong>Mission ${index + 1} · Site ${receipt.site_id}: occupied but missed</strong>
        The adaptive response did survey a truly occupied site. The field outcome was a stochastic non-detection; given occupancy, the miss probability at e${receipt.effort} was ${fmtPct(receipt.conditional_miss_if_occupied)}.</div>`;
    }
    if (receipt.realization === "occupied_and_detected") {
      return `<div class="luck-card detected"><strong>Mission ${index + 1} · Site ${receipt.site_id}: occupied and detected</strong>
        Pre-survey predictive detection probability was ${fmtPct(receipt.predictive_detection_probability)} at effort e${receipt.effort}.</div>`;
    }
    return `<div class="luck-card"><strong>Mission ${index + 1} · Site ${receipt.site_id}: not occupied in hidden truth</strong>
      It was selected from the observable belief state before truth was known; expected information gain was ${fmtNum(receipt.expected_information_gain_bits, 2)} bits.</div>`;
  }).join("");
}

function renderTimeline() {
  const recent = (state.data.events || []).slice(-12);
  $("timeline").innerHTML = recent.map((event) => `<div class="tape-event kind-${event.kind}">
    <span>R${event.round}</span><div><b>${event.title}</b><small>${event.detail}</small></div>
  </div>`).join("");
  $("timeline").scrollLeft = $("timeline").scrollWidth;
}

function renderControls() {
  if (!state.data) return;

  document.querySelectorAll(".effort-btn").forEach((button) => {
    const effort = Number(button.dataset.effort);
    button.disabled = !canSpend(effort) || state.data.can_reveal || state.data.revealed;
    button.classList.toggle("active", effort === state.selectedEffort);
  });

  const layerMeta = state.data.environment_layers || {};
  $("layer-select").value = state.layer;
  Array.from($("layer-select").options).forEach((option) => {
    const meta = layerMeta[option.value];
    option.disabled = meta?.available === false;
    option.title = meta?.available === false ? (meta.provenance || "Layer unavailable") : "";
  });
  if ($("layer-select").selectedOptions[0]?.disabled) {
    state.layer = "belief";
    $("layer-select").value = "belief";
  }

  const done = state.data.can_reveal || state.data.revealed;
  const deploy = $("deploy-btn");
  deploy.disabled = state.busy || !state.selectedSite || !canSpend(state.selectedEffort) || done;
  deploy.querySelector("b").textContent = done
    ? "Response window complete"
    : `Deploy Site ${state.selectedSite || "—"} · e${state.selectedEffort}`;
  deploy.querySelector("small").textContent = done
    ? `${state.data.resources.capacity_preserved} effort units remain preserved`
    : "Survey → field return → Bayes update → replan";

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
    state.mapZoom = 0;
    state.mapPanX = 0;
    state.mapPanY = 0;
    state.zoomPreviewScale = 1;
    if (state.zoomCommitTimer !== null) {
      clearTimeout(state.zoomCommitTimer);
      state.zoomCommitTimer = null;
    }
    showTransition(
      "NEW INCIDENT",
      "Initializing first response",
      "Loading the real monitoring network, probabilistic belief state, and a blinded synthetic hidden extent."
    );
    const data = await api("/api/reset", "POST", { case_id: caseId || $("case-select").value });
    render(data, null);
    setTimeout(hideTransition, 300);
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
      `Effort ${state.selectedEffort}. Hidden truth remains locked.`
    );

    const data = await api("/api/deploy", "POST", {
      site_id: state.selectedSite,
      effort: state.selectedEffort,
    });

    const obs = data.last_round?.observations?.[0];
    if (obs) {
      $("transition-kicker").textContent = "FIELD RETURN";
      $("transition-title").textContent = obs.detection
        ? `Detection at Site ${obs.site_id}`
        : `No detection at Site ${obs.site_id}`;
      $("transition-detail").textContent = `Effort ${obs.effort} · local belief ${fmtPct(obs.belief_before)} → ${fmtPct(obs.belief_after)}. Non-detection is interpreted through effort and q; the full spatial posterior is then recomputed.`;
    }

    await new Promise((resolve) => setTimeout(resolve, 450));
    state.selectedSite = data.global_recommendations?.[0]?.site_id || state.selectedSite;
    if (state.selectedSite) {
      state.selectedEffort = data.global_recommendations?.[0]?.recommended_effort || state.selectedEffort;
    }
    render(data, before);
    setTimeout(hideTransition, 650);
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
      "Scoring Adaptive Response, Static Response, and Your Path on the same blinded incident while keeping resource use visible."
    );

    const before = state.data;
    const data = await api("/api/reveal", "POST");
    await new Promise((resolve) => setTimeout(resolve, 380));
    render(data, before);

    $("transition-kicker").textContent = "TRUE EXTENT REVEALED";
    $("transition-title").textContent = "Same incident. Different allocations.";
    $("transition-detail").textContent = "The evaluator can now separate good ex-ante decisions from lucky or unlucky field realizations.";
    setTimeout(hideTransition, 1100);
  } catch (error) {
    hideTransition();
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

function scheduleMapRender() {
  if (state.mapFrame !== null) return;
  state.mapFrame = requestAnimationFrame(() => {
    state.mapFrame = null;
    renderMap(state.data);
  });
}

function mapRootElement() {
  return document.getElementById("map-root");
}

function clearMapPreviewTransform() {
  const root = mapRootElement();
  if (root) root.removeAttribute("transform");
}

function setMapPreviewTransform(transform) {
  const root = mapRootElement();
  if (root) root.setAttribute("transform", transform);
}

function zoomBy(delta, anchorX = 550, anchorY = 320) {
  const previousZoom = state.mapZoom;
  const nextZoom = Math.max(-1, Math.min(6, previousZoom + delta));
  if (nextZoom === previousZoom) return;

  const scale = 2 ** (nextZoom - previousZoom);
  const offsetX = Number(anchorX) - 550;
  const offsetY = Number(anchorY) - 320;
  state.mapPanX = scale * state.mapPanX + (1 - scale) * offsetX;
  state.mapPanY = scale * state.mapPanY + (1 - scale) * offsetY;
  state.mapZoom = nextZoom;
  renderMap(state.data);
}

function commitZoomPreview() {
  if (state.zoomCommitTimer !== null) {
    clearTimeout(state.zoomCommitTimer);
    state.zoomCommitTimer = null;
  }

  const scale = Number(state.zoomPreviewScale || 1);
  const anchorX = state.zoomPreviewAnchorX;
  const anchorY = state.zoomPreviewAnchorY;
  clearMapPreviewTransform();

  let step = 0;
  if (scale > 1.10) {
    step = Math.min(2, Math.max(1, Math.ceil(Math.log2(scale))));
  } else if (scale < 0.91) {
    step = Math.max(-2, Math.min(-1, Math.floor(Math.log2(scale))));
  }

  state.zoomPreviewScale = 1;
  if (step !== 0) zoomBy(step, anchorX, anchorY);
}

$("clear-world-btn").addEventListener("click", () => {
  state.selectedWorld = null;
  render(state.data);
});

$("follow-marine-btn").addEventListener("click", () => {
  const top = state.data.global_recommendations?.[0];
  if (!top) return;
  state.selectedSite = top.site_id;
  state.selectedEffort = Number(top.recommended_effort || state.selectedEffort);
  render(state.data);
});

$("site-select").addEventListener("change", () => {
  state.selectedSite = $("site-select").value;
  applyRecommendedEffort(state.selectedSite);
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

$("layer-select").addEventListener("change", () => {
  const id = $("layer-select").value;
  const metadata = state.data.environment_layers?.[id];
  if (metadata?.available === false) return;
  state.layer = id;
  state.selectedWorld = null;
  render(state.data);
});

$("zoom-in").addEventListener("click", () => {
  commitZoomPreview();
  zoomBy(1);
});
$("zoom-out").addEventListener("click", () => {
  commitZoomPreview();
  zoomBy(-1);
});
$("zoom-fit").addEventListener("click", () => {
  commitZoomPreview();
  state.mapZoom = 0;
  state.mapPanX = 0;
  state.mapPanY = 0;
  renderMap(state.data);
});

$("graph").addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!state.data || state.drag) return;

  const rect = $("graph").getBoundingClientRect();
  const anchorX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 1100;
  const anchorY = ((event.clientY - rect.top) / Math.max(1, rect.height)) * 640;
  const factor = Math.exp(-Number(event.deltaY) * 0.0024);

  state.zoomPreviewAnchorX = anchorX;
  state.zoomPreviewAnchorY = anchorY;
  state.zoomPreviewScale = Math.max(0.48, Math.min(2.15, state.zoomPreviewScale * factor));

  const s = state.zoomPreviewScale;
  setMapPreviewTransform(
    `translate(${anchorX} ${anchorY}) scale(${s}) translate(${-anchorX} ${-anchorY})`
  );

  if (state.zoomCommitTimer !== null) clearTimeout(state.zoomCommitTimer);
  state.zoomCommitTimer = setTimeout(commitZoomPreview, 95);
}, { passive: false });

$("graph").addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  commitZoomPreview();
  state.drag = {
    x: event.clientX,
    y: event.clientY,
    dx: 0,
    dy: 0,
  };
  $("graph").setPointerCapture(event.pointerId);
  $("graph").classList.add("dragging");
});

$("graph").addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const rect = $("graph").getBoundingClientRect();
  const scaleX = 1100 / Math.max(1, rect.width);
  const scaleY = 640 / Math.max(1, rect.height);
  const dx = (event.clientX - state.drag.x) * scaleX;
  const dy = (event.clientY - state.drag.y) * scaleY;
  state.drag.dx = dx;
  state.drag.dy = dy;
  setMapPreviewTransform(`translate(${dx} ${dy})`);
});

function endDrag(event) {
  if (!state.drag) return;
  const dx = Number(state.drag.dx || 0);
  const dy = Number(state.drag.dy || 0);
  state.mapPanX += dx;
  state.mapPanY += dy;
  state.drag = null;
  clearMapPreviewTransform();
  try { $("graph").releasePointerCapture(event.pointerId); } catch (_) {}
  $("graph").classList.remove("dragging");
  renderMap(state.data);
}
$("graph").addEventListener("pointerup", endDrag);
$("graph").addEventListener("pointercancel", endDrag);

$("minutes-per-unit").addEventListener("input", renderConversion);
$("cost-per-unit").addEventListener("input", renderConversion);

$("deploy-btn").addEventListener("click", deploy);
$("reveal-btn").addEventListener("click", reveal);
$("reset-btn").addEventListener("click", () => resetCase(state.data?.case?.case_id));
$("case-select").addEventListener("change", () => resetCase($("case-select").value));
$("next-case-btn").addEventListener("click", () => {
  const rows = state.cases?.cases || [];
  if (!rows.length) return;
  const current = rows.findIndex((row) => row.case_id === state.data?.case?.case_id);
  resetCase(rows[(current + 1 + rows.length) % rows.length].case_id);
});

bootstrap();
