from __future__ import annotations

from adaptive_response import (
    Edge,
    QHypothesis,
    Site,
    SpatialBeliefEngine,
    WorldModelContext,
    default_world_model_split,
    sample_ecological_hypotheses,
)


def make_context() -> WorldModelContext:
    sites = tuple(
        Site(
            id=f"site_{index:02d}",
            x=float(index % 4),
            y=float(index // 4),
            habitat_score=((index * 7) % 10) / 9.0,
            q_model=0.2,
        )
        for index in range(12)
    )
    edges: list[Edge] = []
    for index in range(12):
        row, col = divmod(index, 4)
        if col < 3:
            edges.append(Edge(f"site_{index:02d}", f"site_{index + 1:02d}", distance=1.0))
        if row < 2:
            edges.append(Edge(f"site_{index:02d}", f"site_{index + 4:02d}", distance=1.0))
    return WorldModelContext(
        sites=sites,
        edges=tuple(edges),
        initial_detection="site_00",
    )


def main() -> None:
    context = make_context()
    split = default_world_model_split()

    print("=== R4 WORLD-MODEL FAMILY DIAGNOSTIC ===")
    print("STATUS: MODEL ASSUMPTIONS for synthetic latent incidents, not ecological truth.")
    print("Rule: >=3 train families + >=1 structurally held-out family.")
    print()

    print("TRAIN FAMILIES")
    for model in split["train"]:
        occupied_counts = [
            sum(model.sample(context, seed=seed).occupied_by_site.values())
            for seed in range(20)
        ]
        print(
            f"  {model.family_id}: draws=20 | occupied min/median-ish/max="
            f"{min(occupied_counts)}/{sorted(occupied_counts)[len(occupied_counts)//2]}/{max(occupied_counts)}"
        )

    print()
    print("OOD MODEL HOLDOUT")
    for model in split["ood_model_holdout"]:
        occupied_counts = [
            sum(model.sample(context, seed=1000 + seed).occupied_by_site.values())
            for seed in range(20)
        ]
        print(
            f"  {model.family_id}: draws=20 | occupied min/median-ish/max="
            f"{min(occupied_counts)}/{sorted(occupied_counts)[len(occupied_counts)//2]}/{max(occupied_counts)}"
        )

    hypotheses = sample_ecological_hypotheses(
        split["train"],
        context,
        draws_per_model=20,
        seed=500,
    )
    labels = sorted({label for hypothesis in hypotheses for label in (hypothesis.label or "").split("|") if label})
    holdout_ids = {model.family_id for model in split["ood_model_holdout"]}

    print()
    print("BELIEF-ENSEMBLE BRIDGE")
    print(f"  sampled train draws={20 * len(split['train'])}")
    print(f"  unique ecological hypotheses={len(hypotheses)}")
    print(f"  labels present={labels}")
    print(f"  holdout absent from train ensemble={holdout_ids.isdisjoint(labels)}")

    belief = SpatialBeliefEngine.initialize(
        hypotheses,
        [QHypothesis(0.05), QHypothesis(0.15), QHypothesis(0.30)],
        confirmed_sites={context.initial_detection},
    )
    p = belief.p_by_site()
    print(f"  posterior hypotheses x q={len(belief.hypotheses)}")
    print(f"  initial detection belief={p[context.initial_detection]:.3f}")
    print(f"  q support remains explicit={list(belief.q_posterior())}")

    print()
    print("INTERPRETATION GATE")
    print("  A/B/C are CURRENT DEFAULT training families, not calibrated ecological laws.")
    print("  E is kept completely outside the train/inference family set in this diagnostic.")
    print("  Numeric parameter ranges are design defaults until simulator criticism constrains them.")
    print("  Do not tune families to make any planner look good.")
    print("  Next: simulator-criticism statistics + frozen benchmark/split contract.")


if __name__ == "__main__":
    main()
