"""HUD taxonomy: ids append after the frozen 38 tiles; op->button mapping dedupes."""
from majsoul_eye.tiles import TILE_NAMES
from majsoul_eye.hud import (HUD_NAMES, DET_NAMES, HUD_NAME_TO_ID, OP_TO_BTN,
                             REACH_STICK_SLOTS, buttons_for_ops, CTC_CHARSET, NUMERIC_FIELDS)

assert len(HUD_NAMES) == 19
assert DET_NAMES[:38] == TILE_NAMES and len(DET_NAMES) == 57
assert HUD_NAME_TO_ID["score_self"] == 38 and HUD_NAME_TO_ID["btn_skip"] == 54  # skip unchanged
assert len(set(DET_NAMES)) == 57
# reach stick: single class appended after btn_skip (spec §10, revised to 1 class)
assert HUD_NAME_TO_ID["reach_stick"] == 55
assert REACH_STICK_SLOTS == ("self", "right", "across", "left")
# btn_babei (sanma) APPENDED last so ids 0-55 stay frozen (56-class weights = prefix)
assert HUD_NAME_TO_ID["btn_babei"] == 56 and DET_NAMES[-1] == "btn_babei"
# an/dai/ka kan share one button; dapai(1) has none; babei(11) -> btn_babei (STATUS §1.61)
assert OP_TO_BTN[4] == OP_TO_BTN[5] == OP_TO_BTN[6] == "btn_kan"
assert 1 not in OP_TO_BTN and OP_TO_BTN[11] == "btn_babei"
assert buttons_for_ops([1]) == []
# order = HUD_NAMES order (kan before riichi); on-screen order calibrated in Task 7
assert buttons_for_ops([1, 7, 4]) == ["btn_kan", "btn_riichi", "btn_skip"]
assert buttons_for_ops([2, 9]) == ["btn_chi", "btn_ron", "btn_skip"]
# sanma babei offer: 北抜き banner + skip (verified on run_1 seq 225/286 frames)
assert buttons_for_ops([1, 11]) == ["btn_babei", "btn_skip"]
assert CTC_CHARSET == "0123456789-x余"
assert set(NUMERIC_FIELDS) == {"score_self", "score_right", "score_across", "score_left",
                               "wall_count", "riichi_stick_count", "honba_count"}
print("test_hud_taxonomy OK")
