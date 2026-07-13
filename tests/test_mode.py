"""Sanma/yonma mode detection. The one failure this feature cannot tolerate is a
confident, coherent, WRONG board — so every ambiguity here must BLOCK, never guess.
"""
from majsoul_eye.annotate import pipeline as P
from majsoul_eye.hud import HUD_NAME_TO_ID
from majsoul_eye.normalize import BoardRegion
from majsoul_eye.recognize.detector import Detection
from majsoul_eye.recognize.mode import detect_mode
from majsoul_eye.tiles import NAME_TO_ID

REGION = BoardRegion(0, 0, 1920, 1080)
H = P.build_homographies(1920, 1080)
_PLATES = ("score_self", "score_right", "score_across", "score_left")


def _hud(name, cx=960, cy=540):
    return Detection(xyxy=(cx - 20, cy - 10, cx + 20, cy + 10), name=name, tile=None,
                     cls=HUD_NAME_TO_ID[name], score=0.9, poly=None)


def _plates(seats):
    return [_hud(_PLATES[s], cx=900 + 30 * s) for s in seats]


def _tile_at(tile, poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Detection(xyxy=(min(xs), min(ys), max(xs), max(ys)), name=tile, tile=tile,
                     cls=NAME_TO_ID[tile], score=0.9,
                     poly=tuple((float(x), float(y)) for x, y in poly))


def _river_tile(seat, tile, geom=P.GEOMETRY_3P):
    slots = P.generate_discard_slots(
        seat, [{"pai": tile, "riichi": False, "tsumogiri": False}], H["H_full_inv"])
    return _tile_at(tile, slots[0]["poly_original"])


def _nuki_tile(seat):
    boxes = P.generate_nukidora_boxes(seat, 1, H["H_full_inv"])
    return _tile_at("N", boxes[0]["poly_original"])


# --- the primary signal -----------------------------------------------------

def test_four_plates_is_four_player():
    d = detect_mode(_plates([0, 1, 2, 3]), REGION)
    assert d.ok and d.sanma is False and d.phantom_rel is None


def test_a_missing_plate_names_the_empty_chair():
    for phantom in (1, 2, 3):
        seats = [s for s in range(4) if s != phantom]
        d = detect_mode(_plates(seats), REGION)
        assert d.ok, d.issues
        assert d.sanma is True
        assert d.phantom_rel == phantom, (phantom, d.phantom_rel)


def test_the_hero_is_never_the_phantom():
    # Losing the hero's own plate means the shot is unusable, not that the hero
    # is the empty chair.
    d = detect_mode(_plates([1, 2, 3]), REGION)
    assert not d.ok
    assert d.issues[0]["code"] == "SCORE_PLATE_SELF_MISSING"


def test_too_few_plates_blocks_rather_than_guesses():
    d = detect_mode(_plates([0, 1]), REGION)
    assert not d.ok
    assert d.issues[0]["code"] == "SCORE_PLATES_UNREADABLE"


# --- the vetoes (each REFUTES; none of them votes) --------------------------

def test_a_manzu_tile_refutes_sanma():
    # Three plates say 3P, but 5m does not exist in 3P. Refuse; do not "win the vote".
    dets = _plates([0, 1, 2]) + [_river_tile(0, "5m")]
    d = detect_mode(dets, REGION)
    assert not d.ok
    assert d.issues[0]["code"] == "SANMA_MODE_CONTRADICTED"
    assert "5m" in d.issues[0]["params"]["tiles"]


def test_the_babei_button_refutes_yonma():
    # Four plates say 4P, but the north-pull button only exists in 3P. One of the
    # two readings is wrong and we do not know which -> block.
    dets = _plates([0, 1, 2, 3]) + [_hud("btn_babei")]
    d = detect_mode(dets, REGION)
    assert not d.ok
    assert d.issues[0]["code"] == "SANMA_MODE_CONTRADICTED"


def test_tiles_in_the_empty_chair_refute_sanma():
    # A missing plate is only ONE piece of evidence; a river in that chair is
    # another, and it points the other way. A false-positive/missed plate must not
    # be able to silently turn a 4P board into a 3P one.
    dets = _plates([0, 1, 2]) + [_river_tile(3, "1p")]
    d = detect_mode(dets, REGION)
    assert not d.ok
    assert d.issues[0]["code"] == "PHANTOM_SEAT_NOT_EMPTY"


def test_a_north_pile_in_the_empty_chair_refutes_sanma():
    dets = _plates([0, 1, 2]) + [_nuki_tile(3)]
    d = detect_mode(dets, REGION)
    assert not d.ok
    assert d.issues[0]["code"] == "PHANTOM_SEAT_NOT_EMPTY"


def test_a_north_pile_in_a_LIVE_chair_is_fine():
    dets = _plates([0, 1, 2]) + [_nuki_tile(0)]
    d = detect_mode(dets, REGION)
    assert d.ok and d.sanma and d.phantom_rel == 3


# --- the override buys geometry, never truth --------------------------------

def test_forcing_four_player_is_honoured():
    d = detect_mode(_plates([0, 1, 2]), REGION, override="4p")
    assert d.ok and d.sanma is False and d.source == "override"


def test_forcing_three_player_still_needs_the_empty_chair_located():
    d = detect_mode(_plates([0, 1, 2, 3]), REGION, override="3p")
    assert not d.ok
    assert d.issues[0]["code"] == "PHANTOM_SEAT_UNKNOWN"


def test_a_forced_mode_does_not_silence_the_vetoes():
    # This is the whole safety argument for the override: it selects geometry, it
    # does not assert truth. A forced 3p on a board showing 5m still blocks.
    dets = _plates([0, 1, 2]) + [_river_tile(0, "5m")]
    d = detect_mode(dets, REGION, override="3p")
    assert not d.ok
    assert d.issues[0]["code"] == "SANMA_MODE_CONTRADICTED"


def test_a_garbage_override_is_rejected():
    d = detect_mode(_plates([0, 1, 2, 3]), REGION, override="sanma")
    assert not d.ok
    assert d.issues[0]["code"] == "MODE_OVERRIDE_INVALID"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_mode OK")
