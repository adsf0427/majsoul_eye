# Skin-swap for capture diversity (`--skins`)

Route the AI-autoplay capture browser through **MajsoulMax** (a mitmproxy addon in
`_external/MajsoulMax`) so the client renders swapped **牌面 / 牌背 / 桌布 / 场景 / 角色立绘**.
Purpose: diversify the training data so the tile detector/classifier generalize across skin
themes. **Data collection only — bot games (vs 电脑), never against real humans.**

## Why a MITM (and not Playwright routing)

MajsoulMax swaps skins by **rewriting the game's WebSocket protobuf** (`.lq.FastTest.authGame`,
`.lq.Lobby.fetchInfo`, …) — it fakes local ownership and the equipped skin. Playwright's
`context.route` only intercepts HTTP, not WS frames, so the swap needs a real proxy between the
browser and the server. `GameBrowser.start(url, proxy, …)` already plumbs a proxy into Chromium,
so enabling this is a launch-time proxy + a trusted CA.

**Labels are unaffected.** MajsoulMax's `mod` plugin only rewrites lobby/account/`authGame`
messages; it never touches `.lq.ActionPrototype` game actions, which is what the GTRecord is built
from. Skins change *pixels*, not *which tile / who discarded*. (Verify — see Runbook §4.)

## One-time setup

1. **Add mitmproxy to `auto`** (one-time) — the mitmdump subprocess runs in the project's `auto`
   env:
   ```
   conda run -n auto pip install --no-cache-dir mitmproxy loguru requests ruamel.yaml
   ```
   This installs mitmproxy 12 and, because every mitmproxy release caps `cryptography` below
   `auto`'s former 49, downgrades `cryptography` to 48 — verified harmless (torch 2.11+cu128 /
   ultralytics / cv2 / numpy all still import & run, CUDA OK). MajsoulMax's protos run under
   `auto`'s protobuf 7 (its `==3.20.1` pin isn't needed at runtime). Default `--skins-env auto`;
   override for any other env that has mitmproxy.
2. **protobuf-7 compat (THE root cause of "unlock doesn't work")** — MajsoulMax was written for
   `protobuf==3.20.1` and calls `MessageToDict(..., including_default_value_fields=True)`, a kwarg
   **removed in protobuf 5+**. Under `auto`'s protobuf 7 it throws on *every* message parse, so
   `res_type` never populates and every unlock response fails the pairing assert → **unlock
   silently no-ops** (parses fine offline; only breaks on live traffic). `SkinProxy` auto-applies
   the rename (`→ always_print_fields_with_no_presence`) to `liqi_new.py`/`mod.py` on every launch
   via `scripts/capture/patch_majsoulmax.py::ensure_protobuf7`. (This — not the liqi version — was
   the real bug; verified by replaying the failed session's 34 messages: 0→34 parse OK after the
   rename.)
3. **Current liqi protocol** — MajsoulMax ships a stale liqi (`v0.11.219.w`); `SkinProxy` also
   bakes the current files (`liqi.json/.proto/liqi_pb2.py/lqc.lqbin`) from
   **`_external/autoliqi-asserts/`** into `MajsoulMax/proto/` each launch for max method coverage.
   MajsoulMax is deprecated (Majsoul web → Unity WebGL); maintained sources are **AutoLiqi**
   (protocol) + **MajsoulData** (`max_data.yaml` catalog, extracted from the WebGL client). When
   Majsoul updates and parsing breaks, re-run AutoLiqi and refresh `_external/autoliqi-asserts/`.
   (Fallback: `offline=False` uses MajsoulMax's deprecated GitHub auto-update.)
4. **Full unlock catalog** — `load_lqc_lqbin` parses only 488 skins from the WebGL-era `lqc.lqbin`;
   MajsoulData's `max_data.yaml` has the complete **494**. `SkinProxy` copies
   **`_external/MajsoulData-asserts/max_data.yaml`** → `proto/max_data.yaml` and
   `patch_majsoulmax::ensure_max_data` makes mod.py override its parsed catalog with it, so *all*
   skins/titles/装扮 unlock. Refresh that file (re-run MajsoulData) when new skins ship.
5. **RES framing (THE root cause of "--skins kills autoplay")** — `BaseMessage` is proto3, so
   mod.py's write-back for a *modified* non-Notify frame (`buf[:3] + msg_block.SerializeToString()`)
   drops the **empty `method_name` field** that every native Majsoul RES carries (`0a 00` on the
   wire). The browser tolerates it, but the autoplay tap parses frames with MahjongCopilot's
   positional `liqi.py`, which asserts block[0] is the (empty) method name — so the
   always-rewritten `authGame` RES was dropped (`parse err: AssertionError`), GameState never
   learned the hero seat (`self.seat` stuck at its default 0; log: `operation seat N !=
   self.seat 0`) and Mortal never acted. `SkinProxy` auto-applies
   `patch_majsoulmax::ensure_res_framing`: the write-back re-frames via `liqi_new.toProtobuf`
   with both blocks explicit — byte-identical to native framing (verified against real captured
   frames end-to-end through MahjongCopilot's parser; see `tests/test_skins.py`).
6. **Trust the CA** — done automatically on first `--skins` run: `SkinProxy` points MajsoulMax's
   `mitmdump` at MahjongCopilot's `mitm_config` confdir and calls MahjongCopilot's
   `install_root_cert` (`certutil -addstore Root`, may prompt UAC once). If you've used
   MahjongCopilot's own MITM before, the cert is already trusted and this is a no-op. If the
   install fails (no admin), `--skins` prints a warning and Majsoul TLS will error until you trust
   `<mjc>/mitm_config/mitmproxy-ca-cert.cer` manually.

MajsoulMax ships its catalog (`proto/lqc.lqbin`) + liqi files, so runs are fully offline — no
network auto-update.

## Usage

```bash
PY=C:/Users/zsx/miniforge3/envs/auto/python.exe

# Manual: unlock everything; set skins/牌背/桌布 in-lobby and they persist for the session.
PYTHONPATH=. $PY scripts/capture/autoplay_ai.py --skins --server jp

# Per-game randomization (hero seat + table 牌面/牌背/桌布/场景 vary every game):
PYTHONPATH=. $PY scripts/capture/autoplay_ai.py --skins --skins-randomize --server jp

# …and randomize EVERY seat's 立绘 incl. AI opponents (verify opponent rendering first, §3):
PYTHONPATH=. $PY scripts/capture/autoplay_ai.py --skins --skins-randomize --skins-all-seats --server jp
```

Flags: `--skins-port` (23410), `--skins-env` (majsoulmax), `--skins-dir` (`_external/MajsoulMax`),
`--skins-slots` (default `13,7,6,8`).

## How it fits together

| Piece | Role |
|-------|------|
| `majsoul_eye/capture/skins.py` — `SkinProxy` | Runs `mitmdump` in the majsoulmax env (cwd=MajsoulMax), seeds offline config, shares MJC's CA, waits for port+cert, tears the subprocess down. Browser-agnostic; cert-install injected as a callable. |
| `scripts/capture/autoplay_ai.py` `--skins*` | Builds the `ensure_cert` closure (MJC `common.utils`), enters `SkinProxy`, passes `proxy_str` to the existing `browser.start(...)`, stops it in `finally`. |
| `scripts/capture/build_skin_config.py` | In-env catalog reader: parses `lqc.lqbin` → `random_character.pool` (every character+skin) + `views` random slots → merges into `settings.mod.yaml`. |
| `scripts/capture/patch_majsoulmax.py` | Tracked, idempotent patcher for `_external`'s **gitignored** MajsoulMax (protobuf-7 kwarg rename, `max_data.yaml` catalog override, RES framing fix, `random_all_seats`). Auto-applied on every `SkinProxy` launch; recoverable source of those edits. |

With `--skins`, each `game<N>/metadata.json` records — next to `language` — both the config and the
**actual per-game skins** (parsed from the rewritten `authGame` RES the browser receives):
`"skins": {"enabled", "randomize", "slots", "all_seats", "table": {slot: item_id …}, "characters":
[{account_id, charid, skin, robot} × seats], "hero_account_id"}`. `table` = the HERO's view
decorations that render the whole table (`7`=牌背 `6`=桌布 `8`=场景); `characters` = every seat's
立绘. So you can stratify the dataset by the exact tile-back / desktop / scene / portraits each
game actually used. The hero is found by `hero_account_id` (from the `authGame` REQ) — the RES's
`players` list is account_id-sorted, NOT hero-first (seat order lives in `seat_list`), so
`players[0]` is usually a stranger. Metadata written before this fix (2026-07-05, ai_session2)
recorded `players[0]`'s views as `table` (usually `{}`); `scripts/capture/backfill_skin_meta.py`
re-derived the correct values from each game's recorded wire (already applied to ai_session2).

## Decoration slot reference (`ItemDefinitionItem.type` == view slot)

| slot | kind | count | tile-relevant | randomized by default |
|-----:|------|------:|:---:|:---:|
| 7 | 牌背 (tile back) | 57 | ★★★ | ✅ |
| 6 | 桌布 (desktop) | 69 | ★★ | ✅ |
| 8 | 场景 (scene) | 20 | ★ | ✅ |
| 13 | 牌面 (tile face) | 3 | ★★★ | ❌ opt-in (changes the recognizer's target) |
| 3 | 手 (hand/paw) | 23 | ~ | opt-in via `--skins-slots` |
| 0 | 立直棒 | 60 | ~ | opt-in |
| 5 | 头像框 | 87 | ✗ (HUD) | never (breaks mod.py's slot-5 read) |
| 1/2/4 | 和牌/立直/入场特效 | — | transient | no |
| 9 | 主题BGM | 22 | audio | no |
| 10 | 鸣牌指示 | 3 | ~ | opt-in |

## Verification runbook (needs a burner login — do these live)

1. **Setup/cert**: first `--skins` run — browser loads Majsoul *through the proxy* and login
   succeeds (no `ERR_CERT_AUTHORITY_INVALID`). `captures/raw/ai_session/run_N/skin_proxy.log`
   shows the mod plugin loaded.
2. **Swap visible**: with `--skins --skins-randomize`, enter a bot game; a captured
   `game<N>/frames/*.png` visibly shows swapped 牌面/牌背/桌布 vs a plain run.
3. **Step 0 — opponent rendering (gates `--skins-all-seats`)**: run
   `--skins --skins-randomize --skins-all-seats`, enter a vs-电脑 game, and check the three AI
   opponents show **different, non-default** 立绘. If they do → all-seats works. If they render
   default → injected skins don't render on opponents; drop `--skins-all-seats` (hero + table
   still vary) and rely on Majsoul's native per-opponent decoration randomization.
4. **Label integrity (critical)**: capture one bot game with `--skins` and one without; diff the
   two `game<N>.jsonl` GTRecords — tile / discard / actor fields must be **identical** (only pixels
   differ). This is the proof skins don't corrupt labels.
5. **Teardown**: after Ctrl-C, no orphan `mitmdump` remains on `--skins-port`.

## Compliance / ban risk

MajsoulMax modifies live traffic and self-warns it may cause account bans. Use **burner accounts**
and **bot games** only. Do not extract/redistribute Majsoul art assets.
