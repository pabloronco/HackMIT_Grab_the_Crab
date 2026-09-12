from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .backbone import GNNActorCritic
from .round_policy import RoundPolicy
from .site_effort_policy import DEFAULT_EFFORT_LEVELS, SiteEffortRoundPolicy

# Engineering hardening (see docs/CHECKPOINT_ENGINEERING_REPORT.md follow-up):
# a checkpoint must be self-describing. Saving only a state_dict makes correct
# loading depend on the caller separately remembering/reconstructing the exact
# architecture (hidden_dim, num_layers, effort_per_pick) it was trained with -
# a silent source of load-time mismatches. Every checkpoint written here embeds
# its own architecture config, so load_policy_checkpoint() alone is enough to
# reconstruct an identical, ready-to-use policy.

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyArchitectureConfig:
    hidden_dim: int
    num_layers: int
    effort_per_pick: int
    max_picks_per_round: int | None = None

    def build(self) -> RoundPolicy:
        backbone = GNNActorCritic(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        return RoundPolicy(
            backbone,
            hidden_dim=self.hidden_dim,
            effort_per_pick=self.effort_per_pick,
            max_picks_per_round=self.max_picks_per_round,
        )


@dataclass(frozen=True)
class SiteEffortPolicyArchitectureConfig:
    """Architecture config for the R7 (site, effort) action-contract policy.

    Sibling of `PolicyArchitectureConfig` (the older autoregressive multi-site
    policy), not a replacement: the older policy/toy AdaptiveMissionLoop path
    stays supported, so both architectures must round-trip through the same
    checkpoint format. See `_ARCHITECTURE_REGISTRY` below.
    """

    hidden_dim: int
    num_layers: int
    effort_levels: tuple[int, ...] = DEFAULT_EFFORT_LEVELS

    def build(self) -> SiteEffortRoundPolicy:
        backbone = GNNActorCritic(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        return SiteEffortRoundPolicy(
            backbone,
            hidden_dim=self.hidden_dim,
            effort_levels=self.effort_levels,
        )


_ARCHITECTURE_REGISTRY: dict[str, type] = {
    "round_policy": PolicyArchitectureConfig,
    "site_effort_policy": SiteEffortPolicyArchitectureConfig,
}


def _architecture_kind(architecture: PolicyArchitectureConfig | SiteEffortPolicyArchitectureConfig) -> str:
    if isinstance(architecture, SiteEffortPolicyArchitectureConfig):
        return "site_effort_policy"
    if isinstance(architecture, PolicyArchitectureConfig):
        return "round_policy"
    raise TypeError(f"Unknown architecture config type: {type(architecture)!r}")


@dataclass(frozen=True)
class LoadedCheckpoint:
    policy: RoundPolicy | SiteEffortRoundPolicy
    architecture: PolicyArchitectureConfig | SiteEffortPolicyArchitectureConfig
    update_idx: int
    extra: dict[str, Any]


def save_policy_checkpoint(
    path: str | Path,
    policy: RoundPolicy | SiteEffortRoundPolicy,
    *,
    architecture: PolicyArchitectureConfig | SiteEffortPolicyArchitectureConfig,
    update_idx: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a single self-describing checkpoint file.

    Deliberately does not require an optimizer: inference-only deployment
    (e.g. demo mode) should not need to carry optimizer state around.
    """

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_kind": _architecture_kind(architecture),
        "architecture": asdict(architecture),
        "update_idx": update_idx,
        "policy_state_dict": policy.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> LoadedCheckpoint:
    """Reconstruct a ready-to-use policy from a self-describing checkpoint.

    Dispatches on `architecture_kind` (written by save_policy_checkpoint) to
    reconstruct either the older autoregressive `RoundPolicy` or the R7
    `SiteEffortRoundPolicy`. A checkpoint written before this field existed
    has no `architecture_kind` key and defaults to `"round_policy"` - the
    only kind that could have been saved at that time, so this is a correct
    default, not a guess. Raises `ValueError` on a checkpoint written by an
    incompatible/unknown format rather than silently misloading it.
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(
            f"{path} does not look like a checkpoint written by "
            "save_policy_checkpoint() (missing format_version)."
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version {payload['format_version']} is not "
            f"supported (expected {CHECKPOINT_FORMAT_VERSION})."
        )

    kind = payload.get("architecture_kind", "round_policy")
    config_cls = _ARCHITECTURE_REGISTRY.get(kind)
    if config_cls is None:
        raise ValueError(f"Unknown checkpoint architecture_kind {kind!r}.")

    architecture = config_cls(**payload["architecture"])
    policy = architecture.build()
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()

    return LoadedCheckpoint(
        policy=policy,
        architecture=architecture,
        update_idx=payload["update_idx"],
        extra=payload.get("extra", {}),
    )


def load_optimizer_state(
    path: str | Path, optimizer: torch.optim.Optimizer, *, map_location: str | torch.device = "cpu"
) -> None:
    """Restore optimizer state from a checkpoint written with one, for resuming
    training (as opposed to inference-only loading via `load_policy_checkpoint`).
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "optimizer_state_dict" not in payload:
        raise ValueError(f"{path} was saved without optimizer state.")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
