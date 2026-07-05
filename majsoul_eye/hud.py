"""HUD-element detector taxonomy — the 17 classes appended after the frozen 38
tile classes (ids 38-54; spec docs/superpowers/specs/2026-07-04-hud-detection-design.md §3).

Pure data (no cv2/numpy) so every component can import it. Button classes are
SEMANTIC — CN/JP/TW glyphs are all training samples of the same class.
"""
from __future__ import annotations

from majsoul_eye.tiles import TILE_NAMES

HUD_NAMES: list[str] = [
    # center info panel (center-anchored; identical on PC/mobile)
    "score_self", "score_right", "score_across", "score_left",
    "round_label", "wall_count", "seat_wind_self",
    # top-left panel
    "riichi_stick_count", "honba_count",
    # action buttons (semantic; glyph varies per server language)
    "btn_chi", "btn_pon", "btn_kan", "btn_riichi",
    "btn_tsumo", "btn_ron", "btn_kyushu", "btn_skip",
]
DET_NAMES: list[str] = TILE_NAMES + HUD_NAMES          # 55-class detector head
HUD_NAME_TO_ID: dict[str, int] = {n: len(TILE_NAMES) + i for i, n in enumerate(HUD_NAMES)}
NUM_DET_CLASSES: int = len(DET_NAMES)
assert NUM_DET_CLASSES == 55, NUM_DET_CLASSES

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
    annotation time, not here. btn_skip accompanies any other button — verify
    empirically at button calibration (Task 7) and adjust if own-turn-only
    options (riichi/ankan/tsumo) turn out to render without a skip button."""
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
# score_across is upside down; left/right signs are CALIBRATED in Task 6 —
# the values below are the seed guess from captures/raw/ai_session frames.
FIELD_ROT: dict[str, int] = {
    "score_self": 0, "score_across": 180, "score_left": 270, "score_right": 90,
    "round_label": 0, "wall_count": 0, "seat_wind_self": 0,
    "riichi_stick_count": 0, "honba_count": 0,
}
