from majsoul_eye import tiles as t


def test_class_count_and_anchors():
    assert t.NUM_CLASSES == 38
    assert t.TILE_NAMES[0] == "1m"
    assert t.TILE_NAMES[34] == "5mr"
    assert t.TILE_NAMES[37] == "back"


def test_mjai_roundtrip():
    for name in t.TILE_NAMES:
        if name == "back":
            assert t.to_mjai(name) is None
            continue
        assert t.from_mjai(t.to_mjai(name)) == name


def test_mjai_specifics():
    assert t.to_mjai("5mr") == "0m" and t.from_mjai("0m") == "5mr"
    assert t.to_mjai("E") == "1z" and t.to_mjai("C") == "7z"


def test_helpers():
    assert t.red_to_normal("5pr") == "5p"
    assert t.red_to_normal("7m") == "7m"
    assert t.is_red_five("5sr") and not t.is_red_five("5s")
    assert len(t.TILE34_NAMES) == 34


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_tiles OK")
