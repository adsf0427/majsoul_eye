"""Build MajsoulMax randomization config from the game catalog (runs in the `majsoulmax` env).

Parses ``<majsoul_dir>/proto/lqc.lqbin`` (needs protobuf==3.20.1 + MajsoulMax's proto modules,
hence the dedicated env) and writes per-game randomization into
``<majsoul_dir>/config/settings.mod.yaml`` so that, on every ``.lq.FastTest.authGame``, MajsoulMax's
``mod`` plugin picks:

  * a random character+skin for the hero (``random_character``), and — if ``--seats all`` — for
    every other seat too (requires the ``random_all_seats`` patch in ``plugin/mod.py``), and
  * a random decoration per requested view slot (``config.views`` type==1 random slots): 牌面 /
    牌背 / 桌布 / 场景 by default (the pixels a tile recognizer must generalize over).

This is the "calibration" the plan called for: decoration ``ItemDefinitionItem.type`` == the
Majsoul view ``slot`` (verified: type 5 == 头像框 == the ``avatar_frame`` slot mod.py hardcodes).

Invoked automatically by ``majsoul_eye.capture.skins.SkinProxy(randomize=...)``; also runnable by
hand:
    conda run -n majsoulmax python scripts/capture/build_skin_config.py _external/MajsoulMax \
        --slots 13,7,6,8 --seats all
"""
from __future__ import annotations

import argparse
import os
import sys

# ItemDefinitionItem.type  ->  Majsoul view slot / decoration kind (category-5 装扮).
DECOR_SLOTS = {
    0: "立直棒", 1: "和牌特效", 2: "立直特效", 3: "手", 4: "入场特效", 5: "头像框",
    6: "桌布", 7: "牌背", 8: "场景", 9: "主题BGM", 10: "鸣牌指示", 13: "牌面",
}
# Default randomized slots: background pixels the recognizer must generalize over — 牌背 / 桌布 /
# 场景. Slot 13 (牌面, the tile FACE) is EXCLUDED by default: it changes the tile symbols themselves,
# i.e. the recognizer's target, so randomizing it shifts data away from the standard face; opt in
# with --skins-slots if you want face-style augmentation. Transient FX (1/2/4), audio (9) and the
# 2D HUD frame (5) are also excluded; slot 5 must never be random anyway (mod.py's authGame reads
# view['item_id'] for slot 5 -> KeyError).
DEFAULT_SLOTS = [7, 6, 8]
FORBIDDEN_SLOTS = {5}


def _iter_sheet(config_table, class_name, pb_factory):
    for data in config_table.datas:
        cn = "".join(w.capitalize() for w in f"{data.table}_{data.sheet}".split("_"))
        if cn == class_name:
            for raw in data.data:
                pb = pb_factory()
                pb.ParseFromString(raw)
                yield pb


def build_random_config(lqbin_bytes: bytes, slots: list[int], all_seats: bool) -> dict:
    """Pure builder: catalog bytes -> the ``config`` fragment we merge into settings.mod.yaml.

    Returns a dict with ``random_character`` (pool over every character+skin), ``views``
    (index 0 = one random slot per requested decoration type; 1..9 empty), ``views_index`` and
    ``random_all_seats``. Requires MajsoulMax's ``proto`` importable (majsoulmax env)."""
    from proto import config_pb2, sheets_pb2  # noqa: E402  (env-specific)

    ct = config_pb2.ConfigTables()
    ct.ParseFromString(lqbin_bytes)

    # character+skin pool: every skin paired with its owning character.
    pool = []
    for skin in _iter_sheet(ct, "ItemDefinitionSkin", sheets_pb2.ItemDefinitionSkin):
        if skin.character_id:
            pool.append({"character_id": int(skin.character_id), "skin_id": int(skin.id)})

    # decoration items grouped by type (== view slot).
    by_slot: dict[int, list[int]] = {}
    for it in _iter_sheet(ct, "ItemDefinitionItem", sheets_pb2.ItemDefinitionItem):
        if it.category == 5:
            by_slot.setdefault(int(it.type), []).append(int(it.id))

    views0 = []
    for s in slots:
        if s in FORBIDDEN_SLOTS:
            print(f"[build_skin_config] skipping forbidden slot {s} ({DECOR_SLOTS.get(s,'?')})", flush=True)
            continue
        ids = by_slot.get(s)
        if not ids:
            print(f"[build_skin_config] slot {s} ({DECOR_SLOTS.get(s,'?')}) has no items; skipped", flush=True)
            continue
        views0.append({"slot": s, "type": 1, "item_id_list": ids})   # type 1 == random per game
        print(f"[build_skin_config] slot {s} {DECOR_SLOTS.get(s,'?')}: {len(ids)} options", flush=True)

    views = {i: [] for i in range(10)}
    views[0] = views0
    print(f"[build_skin_config] character+skin pool: {len(pool)} combos; all_seats={all_seats}", flush=True)
    return {
        "random_character": {"enabled": True, "pool": pool},
        "views": views,
        "views_index": 0,
        "random_all_seats": bool(all_seats),
    }


def merge_into_settings(majsoul_dir: str, frag: dict) -> str:
    """Load settings.mod.yaml (creating a minimal one if absent), set the randomization keys +
    pin resource.auto_update off, and dump. mod.py owns/re-dumps this file; we only set keys it
    already understands. Returns the settings path."""
    from ruamel.yaml import YAML

    yaml = YAML()
    path = os.path.join(majsoul_dir, "config", "settings.mod.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.load(f) or {}
    else:
        doc = {}
    doc.setdefault("config", {})
    doc.setdefault("resource", {})
    doc["config"]["random_character"] = frag["random_character"]
    doc["config"]["views"] = frag["views"]
    doc["config"]["views_index"] = frag["views_index"]
    doc["config"]["random_all_seats"] = frag["random_all_seats"]
    doc["resource"]["auto_update"] = False
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("majsoul_dir", help="Path to the MajsoulMax repo.")
    ap.add_argument("--slots", default=",".join(map(str, DEFAULT_SLOTS)),
                    help="Comma-separated decoration slots to randomize (default: 牌面,牌背,桌布,场景).")
    ap.add_argument("--seats", choices=["hero", "all"], default="hero",
                    help="'all' randomizes every seat (needs the random_all_seats patch in mod.py).")
    args = ap.parse_args(argv)

    majsoul_dir = os.path.abspath(args.majsoul_dir)
    sys.path.insert(0, majsoul_dir)                 # import MajsoulMax's `proto` regardless of cwd
    if args.seats == "all":
        # random_all_seats needs the mod.py patch; ensure it (idempotent) so config never lies.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from patch_majsoulmax import ensure_patched
        applied = ensure_patched(majsoul_dir)
        print(f"[build_skin_config] mod.py patch: {'applied' if applied else 'already present'}", flush=True)
    slots = [int(s) for s in args.slots.split(",") if s.strip()]
    lqbin = os.path.join(majsoul_dir, "proto", "lqc.lqbin")
    with open(lqbin, "rb") as f:
        frag = build_random_config(f.read(), slots, all_seats=(args.seats == "all"))
    path = merge_into_settings(majsoul_dir, frag)
    print(f"[build_skin_config] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
