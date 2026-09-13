from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class BenchmarkCaseSpec:
    """Planner-independent synthetic benchmark case descriptor.

    This object records only inputs that every planner must share: incident-subgraph
    seed, world-model family/seed, latent simulator q scenario, and belief-side q
    support. It deliberately does NOT contain hidden occupancy truth, planner outputs,
    or action/reward internals.
    """

    case_id: str
    split: str
    family_id: str
    world_seed: int
    incident_seed_site_id: str
    incident_site_count: int
    simulator_q_true: float
    belief_q_support: tuple[float, ...]


def _site_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def eligible_incident_seeds(incident_audit: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    """Return only topology-only incident seeds inside the preferred graph-size gate."""

    preferred_min = int(incident_audit["preferred_min_sites"])
    max_sites = int(incident_audit["max_sites"])
    rows = []
    for row in incident_audit["seed_rows"]:
        count = int(row["selected_sites"])
        if preferred_min <= count <= max_sites:
            rows.append((str(row["seed_site_id"]), count))
    rows.sort(key=lambda item: _site_sort_key(item[0]))
    if not rows:
        raise ValueError("No topology-only incident seeds satisfy the preferred size gate.")
    return tuple(rows)


def _q_train_support(q_protocol: Mapping[str, Any]) -> tuple[float, ...]:
    values = tuple(float(v) for v in q_protocol["benchmark_design_support"]["simulator_q_true_train_values"])
    belief = tuple(float(v) for v in q_protocol["benchmark_design_support"]["belief_q_values"])
    if values != belief:
        raise ValueError("R8 manifest assumes the frozen R7 train q_true scenarios match belief support.")
    if not values:
        raise ValueError("q train support cannot be empty.")
    return values


def _append_family_block(
    cases: list[BenchmarkCaseSpec],
    *,
    split: str,
    family_id: str,
    count: int,
    seed_start: int,
    seed_offset: int,
    incident_seeds: Sequence[tuple[str, int]],
    q_train: Sequence[float],
    belief_q_support: tuple[float, ...],
    fixed_q_true: float | None = None,
) -> None:
    for index in range(count):
        world_seed = int(seed_start + seed_offset + index)
        incident_site_id, incident_size = incident_seeds[world_seed % len(incident_seeds)]
        q_true = float(fixed_q_true if fixed_q_true is not None else q_train[world_seed % len(q_train)])
        cases.append(
            BenchmarkCaseSpec(
                case_id=f"{split}__{family_id}__{index:03d}",
                split=split,
                family_id=family_id,
                world_seed=world_seed,
                incident_seed_site_id=incident_site_id,
                incident_site_count=incident_size,
                simulator_q_true=q_true,
                belief_q_support=belief_q_support,
            )
        )


def build_benchmark_case_manifest(
    incident_audit: Mapping[str, Any],
    benchmark_r5: Mapping[str, Any],
    benchmark_r7: Mapping[str, Any],
    q_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen planner-independent R8 formal case manifest.

    Case identity is frozen before the final planner comparison. The manifest contains
    no planner result, reward, action, or latent occupancy vector. Action/reward and
    baseline-fairness rules live in the versioned benchmark protocol, not in case id.
    """

    train_families = tuple(str(v) for v in benchmark_r7["frozen"]["family_split"]["train"])
    ood_families = tuple(str(v) for v in benchmark_r7["frozen"]["family_split"]["ood_model_holdout"])
    if len(train_families) < 3 or len(ood_families) < 1:
        raise ValueError("Expected >=3 train world families and >=1 OOD model holdout.")

    counts = benchmark_r7["formal_case_count_recommendation"]
    validation_per_family = int(counts["validation_per_train_family"])
    id_per_family = int(counts["id_test_per_train_family"])
    ood_model_count = int(counts["ood_model_holdout_cases"])
    ood_q_count = int(counts["ood_q_shift_cases_per_shift"])

    namespaces = benchmark_r5["seed_namespaces"]
    incident_seeds = eligible_incident_seeds(incident_audit)
    q_train = _q_train_support(q_protocol)
    belief_q_support = tuple(float(v) for v in q_protocol["benchmark_design_support"]["belief_q_values"])

    cases: list[BenchmarkCaseSpec] = []

    for family_index, family_id in enumerate(train_families):
        _append_family_block(
            cases,
            split="validation",
            family_id=family_id,
            count=validation_per_family,
            seed_start=int(namespaces["validation"]["start"]),
            seed_offset=family_index * 1000,
            incident_seeds=incident_seeds,
            q_train=q_train,
            belief_q_support=belief_q_support,
        )
        _append_family_block(
            cases,
            split="id_test",
            family_id=family_id,
            count=id_per_family,
            seed_start=int(namespaces["id_test"]["start"]),
            seed_offset=family_index * 1000,
            incident_seeds=incident_seeds,
            q_train=q_train,
            belief_q_support=belief_q_support,
        )

    for family_index, family_id in enumerate(ood_families):
        _append_family_block(
            cases,
            split="ood_model_test",
            family_id=family_id,
            count=ood_model_count,
            seed_start=int(namespaces["ood_model_test"]["start"]),
            seed_offset=family_index * 1000,
            incident_seeds=incident_seeds,
            q_train=q_train,
            belief_q_support=belief_q_support,
        )

    q_shifts = benchmark_r7["ood_q_tests"]
    for shift_index, shift in enumerate(q_shifts):
        q_true = float(shift["q_true"])
        split = "ood_q_low" if q_true < min(q_train) else "ood_q_high"
        per_family = ood_q_count // len(train_families)
        remainder = ood_q_count % len(train_families)
        emitted = 0
        for family_index, family_id in enumerate(train_families):
            count = per_family + (1 if family_index < remainder else 0)
            _append_family_block(
                cases,
                split=split,
                family_id=family_id,
                count=count,
                seed_start=int(namespaces["ood_parameter_test"]["start"]),
                seed_offset=shift_index * 5000 + family_index * 1000,
                incident_seeds=incident_seeds,
                q_train=q_train,
                belief_q_support=belief_q_support,
                fixed_q_true=q_true,
            )
            emitted += count
        if emitted != ood_q_count:
            raise RuntimeError("OOD q case allocation lost cases.")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Benchmark case ids must be unique.")
    world_keys = [(case.split, case.world_seed) for case in cases]
    if len(world_keys) != len(set(world_keys)):
        raise RuntimeError("World seeds must be unique within each formal split.")

    split_counts: dict[str, int] = {}
    topology_counts: dict[str, int] = {}
    for case in cases:
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
        topology_counts[str(case.incident_site_count)] = topology_counts.get(str(case.incident_site_count), 0) + 1

    return {
        "version": "r8-v1",
        "status": "FROZEN_PLANNER_INDEPENDENT_CASE_MANIFEST",
        "case_count_status": "FROZEN_AFTER_RUNTIME_GATE",
        "case_count": len(cases),
        "split_counts": split_counts,
        "incident_topology_size_counts": dict(sorted(topology_counts.items(), key=lambda item: int(item[0]))),
        "eligible_topology_seed_count": len(incident_seeds),
        "belief_q_support": list(belief_q_support),
        "contains_hidden_occupancy_truth": False,
        "contains_planner_results": False,
        "contains_action_or_reward_contract": False,
        "cases": [asdict(case) for case in cases],
        "guardrails": [
            "Every planner receives exactly the same realized case for a given case_id.",
            "incident_seed_site_id comes only from the frozen topology audit; no biological future outcome is used to select it.",
            "simulator_q_true is latent simulator/evaluator state and must not enter policy-facing data.",
            "No simulator family/range may be retuned after formal planner results without a new versioned protocol.",
            "OOD topology remains a separate OPEN lane and is not silently claimed by this manifest.",
        ],
    }
