import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r9_mvp_scope_freezes_static_extent_within_episode() -> None:
    config = json.loads((ROOT / "configs/mvp_scope_r9.json").read_text(encoding="utf-8"))
    episode = config["rapid_response_episode"]
    assert config["status"] == "FROZEN_MVP_SCOPE"
    assert episode["latent_extent_dynamics"] == "static_within_episode"
    assert "latent world does not evolve" in episode["semantics"]


def test_r9_graph_v0_is_explicitly_unweighted_and_undirected() -> None:
    config = json.loads((ROOT / "configs/mvp_scope_r9.json").read_text(encoding="utf-8"))
    graph = config["graph_v0"]
    assert graph["directionality"] == "undirected"
    assert graph["ecological_edge_weighting"] == "unweighted"
    assert "not a source-grounded dispersal probability" in graph["graph_state_implementation_note"]


def test_r9_scope_does_not_overclaim_ood_or_dynamic_spread() -> None:
    config = json.loads((ROOT / "configs/mvp_scope_r9.json").read_text(encoding="utf-8"))
    forbidden = config["benchmark_claims"]["not_allowed"]
    assert "dynamic spread forecasting during the mission" in forbidden
    assert "generic OOD robustness beyond the explicitly tested lanes" in forbidden
