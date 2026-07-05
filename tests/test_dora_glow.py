"""Unit tests for the dora-glow logic (tiles.next_of / tiles.dora_names) that
drives scripts/inspect/count_dora_glow.py. Pure — no frames, no GPU."""

from majsoul_eye.tiles import (
    next_of, dora_names, from_mjai, is_red_five, red_to_normal,
)


def test_next_of_suits_wrap():
    assert next_of("1m") == "2m"
    assert next_of("8p") == "9p"
    assert next_of("9m") == "1m"       # wrap within suit
    assert next_of("9p") == "1p"
    assert next_of("9s") == "1s"
    assert next_of("5s") == "6s"


def test_next_of_winds_cycle():
    assert next_of("E") == "S"
    assert next_of("S") == "W"
    assert next_of("W") == "N"
    assert next_of("N") == "E"         # wrap


def test_next_of_dragons_cycle():
    assert next_of("P") == "F"         # 白 -> 發
    assert next_of("F") == "C"         # 發 -> 中
    assert next_of("C") == "P"         # 中 -> 白 (wrap)


def test_next_of_red_five_indicator_counts_as_plain():
    assert next_of("5mr") == "6m"      # canonical red
    assert next_of("0m") == "6m"       # MJAI red
    assert next_of("0p") == "6p"
    assert next_of("0s") == "6s"


def test_next_of_accepts_mjai_honors():
    assert next_of("1z") == "S"        # MJAI East -> South
    assert next_of("5z") == "F"        # MJAI 白 -> 發
    assert next_of("7z") == "P"        # MJAI 中 -> 白


def test_dora_names_mixed_indicators():
    # 4m -> 5m; E -> S; 0p(red5p) -> 6p
    assert dora_names(["4m", "E", "0p"]) == {"5m", "S", "6p"}


def test_dora_names_empty():
    assert dora_names([]) == set()


def _glows(raw_tile, dset):
    """The exact per-tile glow expression used by count_dora_glow.py."""
    canon = from_mjai(raw_tile)
    return is_red_five(canon) or red_to_normal(canon) in dset


def test_glow_rule_red_five_always():
    assert _glows("0m", set()) is True                   # red five glows w/ no dora
    assert _glows("5mr", dora_names(["1m"])) is True


def test_glow_rule_dora_match():
    dset = dora_names(["4m"])                             # dora = 5m
    assert _glows("5m", dset) is True                    # plain 5m glows
    assert _glows("0m", dset) is True                    # red 5m also glows
    assert _glows("6m", dset) is False                   # non-dora


def test_glow_rule_honor_dora():
    dset = dora_names(["S"])                              # indicator S -> dora W
    assert _glows("W", dset) is True
    assert _glows("N", dset) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_dora_glow OK")
