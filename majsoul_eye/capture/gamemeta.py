"""Per-game metadata for AI-autoplay captures — currently just the display language.

The display language (简体 / 繁體 / 日本語 / English) is NOT carried in the liqi protocol
(game state is language-neutral: tiles are ``1m``/``1z``…, actions are enums) — it is a
client-side render setting. We resolve it from three sources, highest priority first:

  1. an explicit ``--lang`` override (the user knows their own client — 100% reliable);
  2. a page probe (``page.evaluate`` reading the MajSoul client) — mainly to split cn 简/繁;
  3. a coarse map from ``--server`` (jp→ja, en→en, cn→zh-Hans default).

and write ``<game_dir>/metadata.json`` = ``{"language": "<code>"}`` (BCP-47-ish codes).

Browser-agnostic: the probe is exposed as a JS *string* (``probe_language_js``) that the
caller runs on its own page; ``parse_probe_dump`` interprets the returned dump. Nothing here
imports a browser, Akagi, or ultralytics — the module is pure + unit-testable.
"""
from __future__ import annotations

import json
import os
import re

# --server -> coarse display language. jp/en are unambiguous; cn defaults to Simplified
# (Traditional on the cn/global client is a display setting → needs the probe or --lang).
SERVER_LANG = {"jp": "ja", "en": "en", "cn": "zh-Hans"}

# Aliases MajSoul / browsers use → canonical BCP-47-ish code. Keys are lower-cased with
# '_'→'-' (so MajSoul's "chs_t" is matched as "chs-t"). MajSoul's own localStorage
# `prefer_language`/`language` codes: chs=简体, chs_t=繁體, jp=日本語, en=English, kr=한국어.
_LANG_ALIASES = {
    "chs": "zh-Hans", "zh-hans": "zh-Hans", "zh-cn": "zh-Hans", "zhs": "zh-Hans",
    "zh": "zh-Hans", "zh-chs": "zh-Hans", "hans": "zh-Hans",
    "chs-t": "zh-Hant", "cht": "zh-Hant", "zh-hant": "zh-Hant", "zh-tw": "zh-Hant",
    "zh-hk": "zh-Hant", "zht": "zh-Hant", "zh-cht": "zh-Hant", "hant": "zh-Hant",
    "jp": "ja", "ja": "ja", "ja-jp": "ja", "jpn": "ja",
    "en": "en", "en-us": "en", "en-gb": "en", "eng": "en",
    "kr": "ko", "ko": "ko", "ko-kr": "ko", "kor": "ko",
}

_KNOWN = ("zh-Hans", "zh-Hant", "ja", "en", "ko")


def normalize_lang(raw):
    """Map a raw language string to a canonical code (e.g. ``chs``→``zh-Hans``).
    Returns ``None`` for empty/None; passes an unknown non-empty value through trimmed."""
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("_", "-")
    if not key:
        return None
    return _LANG_ALIASES.get(key, str(raw).strip())


def resolve_language(server, probe=None, override=None):
    """Resolve the captured display language. Priority: override > probe > server-coarse.
    Unknown server with no override/probe → ``"unknown"``."""
    for cand in (override, probe):
        norm = normalize_lang(cand)
        if norm:
            return norm
    return SERVER_LANG.get(server, "unknown")


def probe_language_js():
    """JS (for ``page.evaluate``) returning a JSON string of candidate language sources:
    lang/locale-ish localStorage entries, ``navigator.language``, and cookies. On first run
    the caller should LOG this dump to discover where MajSoul stores the display language."""
    return (
        "(() => {"
        "  const hits = {};"
        "  try {"
        "    for (let i = 0; i < localStorage.length; i++) {"
        "      const k = localStorage.key(i);"
        "      const v = localStorage.getItem(k);"
        "      if (/lang|locale|chs|cht|hans|hant/i.test(k) ||"
        "          (v && v.length < 300 && /\\b(chs|cht|zh-?han[st]|ja|jp|en)\\b/i.test(v))) {"
        "        hits[k] = v;"
        "      }"
        "    }"
        "  } catch (e) {}"
        "  let nav = null; try { nav = navigator.language; } catch (e) {}"
        "  let cookie = null; try { cookie = document.cookie; } catch (e) {}"
        "  return JSON.stringify({localStorage: hits, navigatorLanguage: nav, cookie: cookie});"
        "})()"
    )


def parse_probe_dump(dump):
    """Best-effort extract a language code from ``probe_language_js``'s dump. Conservative:
    only trusts a localStorage entry whose KEY names a language/locale setting AND whose value
    is a recognizable code — avoids false positives from unrelated code-looking blobs.
    Returns a canonical code (one of _KNOWN) or ``None`` if nothing trustworthy was found."""
    if isinstance(dump, str):
        try:
            dump = json.loads(dump)
        except Exception:
            return None
    if not isinstance(dump, dict):
        return None
    ls = dump.get("localStorage") or {}
    for k, v in ls.items():
        if re.search(r"lang|locale", str(k), re.I):
            norm = normalize_lang(v)
            if norm in _KNOWN:
                return norm
    return None


def extract_authgame_skins(data):
    """Pull the ACTUAL per-game skins from a parsed ``.lq.FastTest.authGame`` RES dict (as produced
    by MahjongCopilot's liqi parser — camelCase keys). Returns
    ``{"table": {slot: item_id, ...}, "characters": [{account_id, charid, skin, robot}, ...]}``.

    ``table`` = the hero's (``players[0]``) ``views`` decorations — mod.py randomizes only the
    hero's views, and those render the whole table we capture (牌背=slot 7 / 桌布=6 / 场景=8).
    ``characters`` = every seat's 立绘 (charid+skin); ``players`` are humans, ``robots`` the AI.
    Reads the frame AFTER mod.py rewrote it (our WS tap sees what the browser receives), so these
    are the swapped values actually rendered. Missing/zero fields default to 0."""
    data = data or {}

    def _seat(p, robot):
        ch = p.get("character") or {}
        return {"account_id": p.get("accountId", 0), "charid": ch.get("charid", 0),
                "skin": ch.get("skin", 0), "robot": robot}

    players_raw = data.get("players") or []
    characters = [_seat(p, False) for p in players_raw] + \
                 [_seat(p, True) for p in (data.get("robots") or [])]
    table = {}
    if players_raw:
        for v in (players_raw[0].get("views") or []):
            if v.get("slot") is not None:
                table[str(v.get("slot"))] = v.get("itemId", 0)
    return {"table": table, "characters": characters}


def write_metadata(game_dir, language, extra=None):
    """Write ``<game_dir>/metadata.json`` = ``{"language": language, **extra}``. Returns the path.
    ``extra`` (e.g. ``{"skins": {...}}``) is merged in for provenance; falsy → language only."""
    path = os.path.join(game_dir, "metadata.json")
    data = {"language": language}
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path
