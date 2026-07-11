from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserTsumogiriOverride:
    value: bool
    item_id: str
    field_path: str


@dataclass
class ReconstructionOverrides:
    user_visible: dict[tuple[int, int], UserTsumogiriOverride] = field(default_factory=dict)
    user_ghosts: dict[tuple[int, int], UserTsumogiriOverride] = field(default_factory=dict)
    river_ids: dict[tuple[int, int], str] = field(default_factory=dict)
    ghost_ids: dict[tuple[int, int], str] = field(default_factory=dict)
    ghost_order: list[tuple[int, int]] = field(default_factory=list)
