"""Idempotently apply majsoul_eye's ``random_all_seats`` patch to MajsoulMax's ``plugin/mod.py``.

``_external/`` is gitignored, so the in-place edit to the vendored MajsoulMax is NOT tracked and
would be lost on a re-clone/update. This tracked script is the recoverable source of truth for
that edit: it re-applies the three hunks that let MajsoulMax randomize EVERY seat's 立绘 per game
(needed by ``--skins-all-seats`` / ``build_skin_config --seats all``). Manual mode and hero-only
randomization don't need it — they use MajsoulMax's native ``random_character`` + ``views``.

Idempotent: if the marker (``_rand_seat_skin``) is already present it's a no-op. If mod.py has
drifted from the expected upstream text (a different MajsoulMax version), it raises with the hunk
that failed to match rather than silently skipping — so ``--skins-all-seats`` never lies.

  conda-agnostic (pure stdlib):  python scripts/capture/patch_majsoulmax.py _external/MajsoulMax
"""
from __future__ import annotations

import argparse
import os

MARKER = "_rand_seat_skin"

# (label, upstream_text -> patched_text). Order matters only for readability; each is applied once.
HUNKS = [
    ("default config: register random_all_seats key so the merge doesn't drop it",
     "  random_character: # 对局随机角色皮肤\n"
     "    enabled: false\n"
     "    pool: []\n"
     "  safe_mode: false  # 地铁模式，将除自己外所有人变成一姬初始形象，防止被误认为玩黄油。电脑形象请打开“游戏设置-偏好-电脑形象-一姬的初始形象”选项。",
     "  random_character: # 对局随机角色皮肤\n"
     "    enabled: false\n"
     "    pool: []\n"
     "  random_all_seats: false  # majsoul_eye: 训练数据多样性——每局给每个座位(含AI对手)随机角色皮肤(从random_character.pool抽)\n"
     "  safe_mode: false  # 地铁模式，将除自己外所有人变成一姬初始形象，防止被误认为玩黄油。电脑形象请打开“游戏设置-偏好-电脑形象-一姬的初始形象”选项。"),

    ("helper: _rand_seat_skin() pool picker",
     "    def SaveSettings(self):\n"
     "        with open('./config/settings.mod.yaml', 'w', encoding='utf-8') as f:\n"
     "            self.yaml.dump(self.settings, f)",
     "    def SaveSettings(self):\n"
     "        with open('./config/settings.mod.yaml', 'w', encoding='utf-8') as f:\n"
     "            self.yaml.dump(self.settings, f)\n"
     "\n"
     "    def _rand_seat_skin(self):\n"
     "        \"\"\"majsoul_eye: pick a random (charid, skin_id) from random_character.pool, or None.\n"
     "        Used to diversify every seat's 立绘 per game for capture data (see random_all_seats).\"\"\"\n"
     "        pool = self.settings['config'].get('random_character', {}).get('pool') or []\n"
     "        if not pool:\n"
     "            return None\n"
     "        item = random.choice(pool)\n"
     "        return int(item['character_id']), int(item['skin_id'])"),

    ("authGame: randomize non-hero players + all robots when random_all_seats is on",
     "                                p.verified = self.settings['config']['verified']\n"
     "\n"
     "                            if self.settings['config']['show_server']:\n"
     "                                p.nickname =self.get_zone_id(p.account_id)+p.nickname\n"
     "                            if self.settings['config']['safe_mode']:\n"
     "                                p.character.charid=200001\n"
     "                                p.character.skin=400101\n"
     "                                p.avatar_id= 400101\n"
     "                        for p in data.robots:\n"
     "                            p.character.level = 5\n"
     "                            p.character.is_upgraded = True\n"
     "                            p.character.rewarded_level.extend([1, 2, 3, 4, 5])\n"
     "                            p.character.exp = 0\n"
     "                            if self.settings['config']['safe_mode']:\n"
     "                                p.character.charid=200001\n"
     "                                p.character.skin=400101\n"
     "                                p.avatar_id= 400101",
     "                                p.verified = self.settings['config']['verified']\n"
     "\n"
     "                            elif self.settings['config'].get('random_all_seats'):   # majsoul_eye: diversify non-hero seats\n"
     "                                pick = self._rand_seat_skin()\n"
     "                                if pick:\n"
     "                                    p.character.charid, p.character.skin = pick\n"
     "                                    p.avatar_id = pick[1]\n"
     "                            if self.settings['config']['show_server']:\n"
     "                                p.nickname =self.get_zone_id(p.account_id)+p.nickname\n"
     "                            if self.settings['config']['safe_mode']:\n"
     "                                p.character.charid=200001\n"
     "                                p.character.skin=400101\n"
     "                                p.avatar_id= 400101\n"
     "                        for p in data.robots:\n"
     "                            p.character.level = 5\n"
     "                            p.character.is_upgraded = True\n"
     "                            p.character.rewarded_level.extend([1, 2, 3, 4, 5])\n"
     "                            p.character.exp = 0\n"
     "                            if self.settings['config'].get('random_all_seats'):   # majsoul_eye: diversify AI-opponent seats\n"
     "                                pick = self._rand_seat_skin()\n"
     "                                if pick:\n"
     "                                    p.character.charid, p.character.skin = pick\n"
     "                                    p.avatar_id = pick[1]\n"
     "                            if self.settings['config']['safe_mode']:\n"
     "                                p.character.charid=200001\n"
     "                                p.character.skin=400101\n"
     "                                p.avatar_id= 400101"),
]


# protobuf-7 compat: MajsoulMax was written for protobuf 3.20 and calls MessageToDict with the
# `including_default_value_fields` kwarg, which protobuf 5+ RENAMED. Under a modern protobuf (auto
# has 7.35.1) this throws on EVERY message parse -> res_type never populates -> every unlock
# response fails the assert -> unlock silently no-ops (the real bug behind "换肤没生效"). Rename it
# to the protobuf-7 spelling across the files that use it. (This is why MajsoulMax pins
# protobuf==3.20.1; we instead run under auto's protobuf 7 and patch the call sites.)
PROTOBUF7_FILES = ("liqi_new.py", os.path.join("plugin", "mod.py"), os.path.join("plugin", "helper.py"))
OLD_KW = "including_default_value_fields"
NEW_KW = "always_print_fields_with_no_presence"


def ensure_protobuf7(majsoul_dir: str) -> list:
    """Rename the removed protobuf kwarg to its protobuf-7 spelling. Idempotent. Returns the list
    of files changed (empty if already current)."""
    changed = []
    for rel in PROTOBUF7_FILES:
        p = os.path.join(majsoul_dir, rel)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            s = f.read()
        if OLD_KW in s:
            with open(p, "w", encoding="utf-8") as f:
                f.write(s.replace(OLD_KW, NEW_KW))
            changed.append(rel)
    return changed


# MajsoulData catalog override: MajsoulMax's load_lqc_lqbin parses lqc.lqbin (488 skins on the
# current data), but Majsoul→Unity WebGL means the lqbin sheet is INCOMPLETE — MajsoulData extracts
# the full catalog from the WebGL client into max_data.yaml (494 skins). Load that (if SkinProxy
# copied it to proto/max_data.yaml) and override the parsed mod: so ALL skins unlock.
MAX_DATA_MARKER = "max_data.yaml"
MAX_DATA_OLD = (
    "        with open('./proto/lqc.lqbin', 'rb') as f:\n"
    "            self.load_lqc_lqbin(f.read())\n"
    "        self.SaveSettings()")
MAX_DATA_NEW = (
    "        with open('./proto/lqc.lqbin', 'rb') as f:\n"
    "            self.load_lqc_lqbin(f.read())\n"
    "        # majsoul_eye: override with MajsoulData's complete WebGL catalog (max_data.yaml) if present\n"
    "        try:\n"
    "            with open('./proto/max_data.yaml', 'r', encoding='utf-8') as _mdf:\n"
    "                _md = self.yaml.load(_mdf)\n"
    "            if _md and 'mod' in _md:\n"
    "                self.settings['mod'] = _md['mod']\n"
    "                logger.success('majsoul_eye: 已用 max_data.yaml 覆盖解锁目录')\n"
    "        except FileNotFoundError:\n"
    "            pass\n"
    "        except Exception as _e:\n"
    "            logger.warning(f'max_data.yaml 读取失败，回退 lqc.lqbin 目录: {_e}')\n"
    "        self.SaveSettings()")


def ensure_max_data(majsoul_dir: str) -> bool:
    """Make mod.py prefer proto/max_data.yaml's catalog. Idempotent; raises if the anchor drifted."""
    p = mod_path(majsoul_dir)
    with open(p, "r", encoding="utf-8") as f:
        s = f.read()
    if MAX_DATA_MARKER in s:
        return False
    if MAX_DATA_OLD not in s:
        raise RuntimeError(f"cannot apply max_data override to {p}: anchor not found (MajsoulMax drift?)")
    with open(p, "w", encoding="utf-8") as f:
        f.write(s.replace(MAX_DATA_OLD, MAX_DATA_NEW, 1))
    return True


# RES framing: Majsoul's native frames ALWAYS carry BaseMessage field 1 (method_name) — for a RES
# it is an EMPTY string (`0a 00` on the wire; verified against captured frames). BaseMessage is
# proto3 (no presence on scalars), so mod.py's write-back for a modified non-Notify message
# (`msg = buf[:3] + msg_block.SerializeToString()`) silently DROPS that empty field. The browser
# client tolerates it, but POSITIONAL liqi parsers downstream (MahjongCopilot liqi.py:157 — our
# autoplay tap) assert block[0] is the empty method name and throw AssertionError on every
# REWRITTEN RES. mod.py rewrites the authGame RES unconditionally, so with --skins the tap dropped
# authGame -> GameState never learned the hero seat (self.seat stuck at 0) -> Mortal never acted.
RES_FRAMING_MARKER = "majsoul_eye: preserve the method_name block"
RES_FRAMING_OLD = (
    "        if modify:\n"
    "            msg_block.data = data.SerializeToString()\n"
    "            if msg_type == liqi_new.MsgType.Notify:\n"
    "                msg = b'\\x01' + msg_block.SerializeToString()\n"
    "            else:\n"
    "                msg = buf[:3] + msg_block.SerializeToString()\n")
RES_FRAMING_NEW = (
    "        if modify:\n"
    "            msg_block.data = data.SerializeToString()\n"
    "            if msg_type == liqi_new.MsgType.Notify:\n"
    "                msg = b'\\x01' + msg_block.SerializeToString()\n"
    "            else:\n"
    "                # majsoul_eye: preserve the method_name block even when EMPTY (RES). proto3\n"
    "                # drops empty scalars, but Majsoul's native frames always carry it and\n"
    "                # positional liqi parsers downstream require it at block[0].\n"
    "                msg = buf[:3] + liqi_new.toProtobuf([\n"
    "                    {'id': 1, 'type': 'string', 'data': msg_block.method_name.encode()},\n"
    "                    {'id': 2, 'type': 'string', 'data': msg_block.data},\n"
    "                ])\n")


def ensure_res_framing(majsoul_dir: str) -> bool:
    """Make mod.py's modified-message write-back byte-identical to Majsoul's native framing.
    Idempotent; raises if the anchor drifted."""
    p = mod_path(majsoul_dir)
    with open(p, "r", encoding="utf-8") as f:
        s = f.read()
    if RES_FRAMING_MARKER in s:
        return False
    if RES_FRAMING_OLD not in s:
        raise RuntimeError(f"cannot apply RES-framing fix to {p}: anchor not found (MajsoulMax drift?)")
    with open(p, "w", encoding="utf-8") as f:
        f.write(s.replace(RES_FRAMING_OLD, RES_FRAMING_NEW, 1))
    return True


def ensure_all(majsoul_dir: str) -> None:
    """Apply every majsoul_eye patch MajsoulMax needs: protobuf-7 kwarg (always — required for the
    addon to parse anything) + max_data catalog override + RES framing fix (required for the
    autoplay tap to survive rewritten RES frames) + random_all_seats hunks. All idempotent."""
    ensure_protobuf7(majsoul_dir)
    ensure_max_data(majsoul_dir)
    ensure_res_framing(majsoul_dir)
    ensure_patched(majsoul_dir)


def mod_path(majsoul_dir: str) -> str:
    return os.path.join(majsoul_dir, "plugin", "mod.py")


def is_patched(majsoul_dir: str) -> bool:
    with open(mod_path(majsoul_dir), "r", encoding="utf-8") as f:
        return MARKER in f.read()


def ensure_patched(majsoul_dir: str) -> bool:
    """Apply the patch if absent. Returns True if it applied it, False if already present.
    Raises RuntimeError if a hunk's upstream text can't be matched (MajsoulMax version drift)."""
    path = mod_path(majsoul_dir)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        return False
    for label, old, new in HUNKS:
        if old not in src:
            raise RuntimeError(
                f"cannot apply majsoul_eye patch hunk [{label}] to {path}: upstream text not found "
                f"(MajsoulMax version drift?). Re-apply manually — see scripts/capture/patch_majsoulmax.py.")
        src = src.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("majsoul_dir", nargs="?", default=os.path.join("_external", "MajsoulMax"),
                    help="Path to the MajsoulMax repo.")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.majsoul_dir)
    pb7 = ensure_protobuf7(d)
    print(f"protobuf-7 kwarg: {'renamed in ' + ', '.join(pb7) if pb7 else 'already current (no-op)'}", flush=True)
    print(f"max_data override: {'applied' if ensure_max_data(d) else 'already present (no-op)'}", flush=True)
    print(f"RES framing fix: {'applied' if ensure_res_framing(d) else 'already present (no-op)'}", flush=True)
    applied = ensure_patched(d)
    print("patched mod.py (random_all_seats)" if applied else "mod.py already patched (no-op)", flush=True)


if __name__ == "__main__":
    main()
