from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Engineering block (2026-09-12): structured per-round logging for debugging
# and later benchmark comparison, requested alongside the aggregate CSVs
# already produced by train_gnn_policy.py. This is opt-in (see run_episode's
# `decision_logger` parameter): normal training is entirely unaffected unless
# a caller explicitly attaches one, so this cannot change any existing run's
# behavior, timing, or output.


@dataclass(frozen=True)
class RoundLogRecord:
    episode_index: int
    round_index: int
    budget_before: int
    budget_after: int
    site_ids_picked: tuple[str, ...]
    num_picks: int
    node_ids: tuple[str, ...]
    node_logits: tuple[float, ...]
    eligible_mask: tuple[bool, ...]
    value_estimate: float
    entropy: float
    log_prob: float
    reward_components: dict[str, float]
    reward_total: float
    detections_this_round: int
    done: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialRoundLogRecord:
    """One round under the R7 (site, effort) action contract.

    Sibling of `RoundLogRecord` (the older autoregressive multi-site record),
    not a replacement - the two action contracts produce genuinely different
    per-round shapes (one chosen site+effort here vs. a variable-length list
    of picks there), so reusing one schema for both would force one of them
    into an awkward, ambiguous encoding. `node_logits`/`eligible_mask` are
    [N][K] (K = number of effort levels), index-aligned with `node_ids` and
    `effort_levels`.
    """

    episode_index: int
    round_index: int
    budget_before: int
    budget_after: int
    site_id: str
    effort_units: int
    node_ids: tuple[str, ...]
    effort_levels: tuple[int, ...]
    node_logits: tuple[tuple[float, ...], ...]
    eligible_mask: tuple[tuple[bool, ...], ...]
    value_estimate: float
    entropy: float
    log_prob: float
    reward_components: dict[str, float]
    reward_total: float
    detection: bool
    done: bool
    extra: dict[str, Any] = field(default_factory=dict)


class JsonlDecisionLogger:
    """Append-only JSONL writer: one line per `RoundLogRecord`.

    Usable as a context manager or with an explicit `close()`. Safe to pass
    as `None` anywhere a `DecisionLogger | None` is expected (see
    `training_env.run_episode`) - callers that don't want detailed logging
    simply don't construct one.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def log_round(self, record: RoundLogRecord | SpatialRoundLogRecord) -> None:
        self._file.write(json.dumps(asdict(record)) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlDecisionLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read back a log written by `JsonlDecisionLogger`, for debugging/tests."""

    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
