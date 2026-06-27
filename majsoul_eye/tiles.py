"""Unified tile taxonomy for majsoul_eye.

Single source of truth for tile classes, shared by every component
(classifier, detector, dataset labels, structured-state output).

Ported from the proven taxonomy in ``auto/mycv`` (``classifier2.py`` and the
active ``dictjianxie`` in ``main2.py:67-70``). NOTE: an older, *commented-out*
table in ``main2.py:63-66`` ordered the suits as s/p/m and reversed the red
fives (``34=5sr,35=5pr,36=5mr``). That ordering is DEAD. The canonical order
below — m, p, s, honors, red5(m,p,s), back — is what ``tile.model`` was trained
on. Do not reorder without retraining.

Class id layout (38 classes, matching ``TileNet`` output):
    0-8   : 1m-9m  (manzu / 萬)
    9-17  : 1p-9p  (pinzu / 筒)
    18-26 : 1s-9s  (souzu / 索)
    27-33 : E S W N P F C  (winds + dragons; P=白 F=發 C=中)
    34-36 : 5mr 5pr 5sr    (red fives / 赤五)
    37    : back           (tile back / 牌背 — opponents' concealed tiles)
"""

from __future__ import annotations

# --- canonical class list (index == class id) -------------------------------

TILE_NAMES: list[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "back",
]

NUM_CLASSES: int = len(TILE_NAMES)  # 38
assert NUM_CLASSES == 38, NUM_CLASSES

NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(TILE_NAMES)}

BACK_ID: int = NAME_TO_ID["back"]
RED_FIVE_NAMES: frozenset[str] = frozenset({"5mr", "5pr", "5sr"})
HONOR_NAMES: frozenset[str] = frozenset({"E", "S", "W", "N", "P", "F", "C"})

# --- MJAI interop -----------------------------------------------------------
# MJAI notation: suited tiles "1m".."9m"/"1p"../"1s"..; red five "0m"/"0p"/"0s";
# honors "1z"=E "2z"=S "3z"=W "4z"=N "5z"=P(白) "6z"=F(發) "7z"=C(中).
# 'back' has no MJAI equivalent (it is hidden information).

_MYCV_TO_MJAI: dict[str, str] = {
    "5mr": "0m", "5pr": "0p", "5sr": "0s",
    "E": "1z", "S": "2z", "W": "3z", "N": "4z",
    "P": "5z", "F": "6z", "C": "7z",
}
_MJAI_TO_MYCV: dict[str, str] = {v: k for k, v in _MYCV_TO_MJAI.items()}


def to_mjai(name: str) -> str | None:
    """Convert a canonical tile name to MJAI notation. 'back' -> None."""
    if name == "back":
        return None
    return _MYCV_TO_MJAI.get(name, name)


def from_mjai(mjai: str) -> str:
    """Convert MJAI notation to a canonical tile name."""
    return _MJAI_TO_MYCV.get(mjai, mjai)


# --- helpers ----------------------------------------------------------------

# 34-tile standard order (no red fives, no back) for count arrays such as the
# discard/meld/dora histograms (mycv's ``paihe[4, 37]`` style buffers).
TILE34_NAMES: list[str] = TILE_NAMES[:34]
TILE34_TO_ID: dict[str, int] = {name: i for i, name in enumerate(TILE34_NAMES)}


def is_red_five(name: str) -> bool:
    return name in RED_FIVE_NAMES


def red_to_normal(name: str) -> str:
    """Collapse a red five to its plain five ('5mr' -> '5m'); else unchanged."""
    if name in RED_FIVE_NAMES:
        return name[:2]  # '5mr' -> '5m'
    return name


def name_of(class_id: int) -> str:
    return TILE_NAMES[class_id]


def id_of(name: str) -> int:
    return NAME_TO_ID[name]
