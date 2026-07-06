"""HUD-element detector taxonomy — the 21 classes appended after the frozen 38
tile classes (ids 38-58; spec docs/superpowers/specs/2026-07-04-hud-detection-design.md §3,
reach sticks added in §10).

Pure data (no cv2/numpy) so every component can import it. Button classes are
SEMANTIC — CN/JP/TW glyphs are all training samples of the same class.
"""
from __future__ import annotations

from majsoul_eye.tiles import TILE_NAMES

# Reach-stick slots (ids 55-58; spec §10, 2026-07-06): four center-panel-edge
# slots, one per seat, lit up while that seat is in riichi. LABEL-ONLY like the
# buttons below — no text to read, so these are deliberately absent from
# FIELD_ROT / NUMERIC_FIELDS (there is nothing to rotate-and-CTC-read; the
# detected class IS the label). Distinct from `riichi_stick_count` (the
# top-left kyotaku/供托 pot counter, a NUMERIC_FIELD): that one counts how many
# 1000-point sticks are in the pot; these four say WHICH seat(s) contributed
# one, i.e. "this player is currently in riichi".
REACH_STICK_NAMES: list[str] = [
    "reach_stick_self", "reach_stick_right", "reach_stick_across", "reach_stick_left",
]

HUD_NAMES: list[str] = [
    # center info panel (center-anchored; identical on PC/mobile)
    "score_self", "score_right", "score_across", "score_left",
    "round_label", "wall_count", "seat_wind_self",
    # top-left panel
    "riichi_stick_count", "honba_count",
    # action buttons (semantic; glyph varies per server language)
    "btn_chi", "btn_pon", "btn_kan", "btn_riichi",
    "btn_tsumo", "btn_ron", "btn_kyushu", "btn_skip",
] + REACH_STICK_NAMES                                  # ids 55-58 (spec §10)
DET_NAMES: list[str] = TILE_NAMES + HUD_NAMES          # 59-class detector head
HUD_NAME_TO_ID: dict[str, int] = {n: len(TILE_NAMES) + i for i, n in enumerate(HUD_NAMES)}
NUM_DET_CLASSES: int = len(DET_NAMES)
assert NUM_DET_CLASSES == 59, NUM_DET_CLASSES

# liqi operation type -> button class. Wire shape verified on run_13/game1:
# raw_liqi.data.data.operation = {seat, operationList:[{type, combination,...}], ...}.
# type 1 = dapai (no button), 11 = babei (3p, out of scope). An/dai/ka kan share
# ONE button. Codes follow Akagi/MahjongCopilot convention — re-check against
# Akagi's liqi parser if a mismatch shows up in calibration (spec §3.3).
OP_TO_BTN: dict[int, str] = {
    2: "btn_chi", 3: "btn_pon",
    4: "btn_kan", 5: "btn_kan", 6: "btn_kan",
    7: "btn_riichi", 8: "btn_tsumo", 9: "btn_ron", 10: "btn_kyushu",
}


def buttons_for_ops(op_types: list[int]) -> list[str]:
    """Pending liqi op types -> button classes expected on screen (dapai-only -> []).
    Order = HUD_NAMES order (stable); on-screen ordering is assigned by x-sort at
    annotation time, not here. btn_skip accompanies any other button —
    VERIFIED (Task 7 Step 5) on a real own-turn riichi offer (seq 302,
    captures/raw/ai_session3/run_1/game1, ops=[1(dapai), 7(riichi)]): the frame
    shows BOTH a 立直 banner AND a スキップ banner, so own-turn-only offers do
    NOT drop the skip button — no change needed here."""
    btns = [b for b in HUD_NAMES if b in {OP_TO_BTN.get(t) for t in op_types}]
    if btns:
        btns.append("btn_skip")
    return btns


# --- micro-reader contracts -------------------------------------------------
CTC_CHARSET: str = "0123456789-x余"   # model emits index+1; 0 = CTC blank
NUMERIC_FIELDS: tuple[str, ...] = (
    "score_self", "score_right", "score_across", "score_left",
    "wall_count", "riichi_stick_count", "honba_count",
)
ROUND_CLASSES: list[str] = [f"{w}{k}" for w in "ESWN" for k in (1, 2, 3, 4)]  # 16
WIND_CLASSES: list[str] = ["E", "S", "W", "N"]

# Per-class rotation (degrees CW) that uprights the crop before reading.
# CALIBRATED (Task 6, calibrated 2026-07-05): the ±90° rotation directions for
# score_left, score_right, and score_across were verified by rotating real-frame
# crops (from multiple sessions) until digits read upright. Overlay QA in
# scripts/inspect/overlay_hud.py. Details: score_left requires 270 CW (equiv. 90
# CCW) to display correct digit order; score_right requires 90 CW; score_across
# is upside down (180).
FIELD_ROT: dict[str, int] = {
    "score_self": 0, "score_across": 180, "score_left": 270, "score_right": 90,
    "round_label": 0, "wall_count": 0, "seat_wind_self": 0,
    "riichi_stick_count": 0, "honba_count": 0,
}
