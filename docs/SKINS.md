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

1. **Isolated env** (MajsoulMax needs `protobuf==3.20.1` + `mitmproxy`, incompatible with our
   capture env — hence a subprocess in its own env):
   ```
   conda create -n majsoulmax python=3.11 -y
   conda run -n majsoulmax pip install --no-cache-dir "mitmproxy>=10,<11" "protobuf==3.20.1" requests ruamel.yaml loguru
   ```
2. **Trust the CA** — done automatically on first `--skins` run: `SkinProxy` points MajsoulMax's
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
| `scripts/capture/patch_majsoulmax.py` | Tracked, idempotent patcher for `_external`'s **gitignored** `plugin/mod.py` (adds `random_all_seats`). Auto-applied when `--seats all`; recoverable source of that edit. |

## Decoration slot reference (`ItemDefinitionItem.type` == view slot)

| slot | kind | count | tile-relevant | randomized by default |
|-----:|------|------:|:---:|:---:|
| 13 | 牌面 (tile face) | 3 | ★★★ | ✅ |
| 7 | 牌背 (tile back) | 56 | ★★★ | ✅ |
| 6 | 桌布 (desktop) | 68 | ★★ | ✅ |
| 8 | 场景 (scene) | 20 | ★ | ✅ |
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
