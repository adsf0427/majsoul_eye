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
    applied = ensure_patched(os.path.abspath(args.majsoul_dir))
    print("patched mod.py" if applied else "mod.py already patched (no-op)", flush=True)


if __name__ == "__main__":
    main()
