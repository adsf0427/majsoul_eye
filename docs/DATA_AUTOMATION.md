# majsoul_eye — 无人值守数据采集方案（浏览器 autoplay + 进程内 WS 抓包）

> 目的：消除数据管线里**唯一**还需人工的环节——产生截图。把"人工驾驶浏览器对局"
> 换成**浏览器自动对局**，截图与 GT 顺着同一条 WebSocket 搭车采集。
> 参考实现：`D:\code\phoenix\MahjongCopilot`（latorc/MahjongCopilot，**GPLv3**，
> 非 Akagi 的 AGPL+Commons Clause）。本方案三轮调研 + 对抗式核验产出，见会话记录。
> **历史注记（2026-07）**：文中的 `scripts/spike_ws_tap.py` / `spike_decode_*.py` / `spike_autoplay.py`
> 已完成验证并从仓库删除（结论固化进 `scripts/autoplay_ai.py` 与 `scripts/convert_mjcopilot.py`）；
> 需要时可在 git 历史（`861db68`）中找回。

## 1. 核心：一个 Playwright Chromium = 自驱 + 抓包 + 截图

MahjongCopilot 的浏览器 autoplay 已实现完整的 **自动加入→逐手出牌→结算确认→续局**
循环（[`game/automation.py`](../../MahjongCopilot/game/automation.py)）。我们在**同一个
浏览器进程**里再挂"抓 liqi WS 帧 → GT"和"截图"，三者共用一个进程时钟，**彻底去掉
Akagi、mitmproxy、proxinject、CA 证书、独立抓屏器**。出牌强度无关 → 随机合法出牌，
不需要 Mortal。

```
ONE headed Playwright Chromium (launch_persistent_context)
  DRIVE   : RandomBot.react() → automate_action()    [复用 automation.py + Positions]
            随机合法出牌（tsumogiri=最右张恒合法），无 Mortal/libriichi
  CAPTURE : page.on('websocket') → ws.on('framereceived')   ← 唯一全新组件
            → liqi.parse(bytes) → liqi→MJAI
  GT      : MJAI → 本仓库 state/replay.py:Replayer → 四家 BoardState
  SYNC    : capture/sync.py:FrameSyncer（保留）→ page.screenshot()，按全局 seq 命名
  LABEL   : label/{autolabel,river,meld}.py + coords.py（不变）
```

## 2. 复用 / 改写 / 新建 / 删除

| 动作 | 内容 | 依据 |
|---|---|---|
| **复用（本仓库）** | `state/replay.py:Replayer`（四家 GT，MahjongCopilot 的 `GameState` 只有己方手牌，无河/副露）、`capture/sync.py:FrameSyncer`、`label/*`、`coords.py` | 已是我们的、Akagi-free |
| **复用/改写（参考）** | `automation.py` 自驱循环 + `class Positions`（16:9 归一化坐标）；`browser.py` 的 mouse/screenshot/page.evaluate | GPLv3 → clean-room 重写，坐标表当数据 |
| **改写** | `game_state.py` 的 liqi→MJAI 转换（aka-dora、东家 14 张拆分、立直延迟），喂给我们的 Replayer；MJAI 从 `game_state.mjai_pending_input_msgs` **收割**，不是 bot 返回值 | — |
| **新建①** | **进程内 WS tap**：`page.on('websocket')` → `framereceived` → 入队 → 解码。MahjongCopilot 全仓库 0 处 CDP 代码，`crx/` 是空占位 | 唯一真正的新工作 |
| **新建②** | `RandomBot(Bot)` ~30 行：`react` 返回 `{'type':'dahai','actor':seat,'pai':<牌>,'tsumogiri':True}`，所有鸣牌提示返回 `{'type':'none'}` | 无 `meta_options`/`consumed`/libriichi |
| **删除** | `mitm.py`、`proxinject.py`、`crx/`、`bot/`+`libriichi`/`libriichi3p`（**AGPL-3.0**，真正的 copyleft 陷阱）、`randomize_action`（只温度化模型 top-3，非均匀随机；设 `ai_randomize_choice=0`） | 去耦合、去依赖 |

## 3. 三个承重风险（核验为"很可能但未实证"，须先 spike）

1. **WS tap 抓不抓得到对局 socket（gating）**：Playwright `page.on('websocket')` 只暴露
   **主线程** WS（issue #37048），CDP 亦有偶发不回调（puppeteer #11456）。证据强烈
   指向雀魂 socket 在主线程（userscript 能从页面 realm override `window.WebSocket`），
   但无直接实证。→ 见 `scripts/spike_ws_tap.py`。失败则退到
   `new_cdp_session`+`Network.enable`+`webSocketFrameReceived`(base64)，或
   `window.WebSocket` init-script shim。
2. **两个 socket**：大厅 WS（`.lq.Lobby.oauth2Login`）与对局 WS（`.lq.FastTest.authGame`）
   分开（"单 socket"说法**已被推翻**）→ 按每个 `ws` 对象嗅探首个 method 路由。流量是
   wss://(TLS)，XOR 只混淆内层 `ActionPrototype.data`（能抓是因 TLS 在渲染器内已解密）。
3. **时间对齐不是白送**（"单进程=对齐"降级为 partly-true）：协议先于动画渲染——这正是
   `sync.py` 的核心风险 → **保留 FrameSyncer**（debounce-to-quiet + 帧差）。另：Playwright
   sync API 非线程安全，WS 回调与 `page.screenshot()` 在同一浏览器线程串行 → tap **入队不阻塞**。

## 4. License 与封号

- **License**：MahjongCopilot 是 **GPLv3**。**勿将 `liqi.py`/`automation.py` 原样拷入要
  发布的产品**（会传染 GPLv3）。liqi 解析器从 MIT 的
  `github.com/MahjongRepository/mahjong_soul_api` vendor（或自己重 dump `liqi.json`）；
  automation 当 clean-room 参考重写。
- **`navigator.webdriver` 现暴露为 `true`**（去掉 `--enable-automation` 只去提示条）→ 加
  `--disable-blink-features=AutomationControlled` + init-script 改写。
- **封号姿态**：无 TLS-MITM/代理/证书/进程注入；随机出牌=无 AI 一致性指纹。**唯一偏离**
  DESIGN.md §7 被动偏好：autoplay 只导航**段位场**（无友人战/人机导航代码）→ 建议铜/银、
  限场次、勿 24/7、用小号。

## 5. Spike 清单（各自可独立验证）

1. 在 `auto` env 装 `playwright`（`requirements.txt` 第 16 行现注释掉；参考 pin 1.42.0）。
2. **`scripts/spike_ws_tap.py`（gating）**：`goto` 前注册监听，打印每个 `ws.url` + 首批
   `framereceived`。**PASS = 实战中收到一帧 `.lq.FastTest.*` 二进制帧**。同时跑 CDP 回退
   对比，定位 worker-vs-main-thread。
3. **解码 round-trip**：把抓到的对局帧喂 vendor 的 `LiqiProto.parse` → 得到合法
   `ActionPrototype` dict；确认每 socket 内 REQ/RES `msg_id` 配对有序。
4. **MJAI 方言 diff**：转换出的 MJAI 流逐字段对比 `state/replay.py` 期望
   （`tsumogiri` vs `moqie`、`consumed` 顺序、`dora` 位置）。事件名已对齐，字段需确认。
5. **RandomBot 端到端（4p）**：续局模板需**按你的分辨率/皮肤/语言重新截取**
   （`img_proc.py:GameVisual` 的 MAIN_MENU 是视觉模板匹配，非协议）。
6. **接 FrameSyncer**：quiet 时 `page.screenshot()` + 快照 Replayer 状态，按全局 `seq` 命名。
7. **整局不变量门**：`Replayer.check_invariants()`，违例帧丢弃/人工复核；帧数≈动作数。
8. **隐身 + 限流**：上 `--disable-blink-features=AutomationControlled` + webdriver init-script。

## 6. Replay（牌谱）—— 次要，留待以后识别

一次性 blob 拉取（`.lq.Lobby.fetchGameRecord` → `GameDetailRecords`，version 0=`records[]` /
210715=`actions[]`），**非**逐步 WS 流；`RecordNewRound.tiles0..3` = 四家全手牌（完整 GT）。
解码后是同一 `Record*` 方言 → 我们的 Replayer 可直接消费（仅需 ~30 行 blob 解码器）。
揭示对手手牌 = **`record_show_hand` 开关**（不是切座位；`changeMainbody` 只移相机）。
web 端**仍是 Laya**，JS 注入可用但版本脆 → GT 走 blob 解码（引擎无关），JS 只用于驱动渲染。
识别器以后须处理：**控制条遮挡、揭示的对家手牌条、可变 hero 座位（≠永远在底）**。
