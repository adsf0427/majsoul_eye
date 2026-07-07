"""Interactive manual calibration of opponent hand-row tile-back quads (one-shot tool).

Generates a SELF-CONTAINED HTML annotator (frame embedded as base64) and opens it
in the default browser — the conda `auto` env ships opencv-python-HEADLESS, so a
cv2.imshow UI is not available. In the page you click the 4 corners of each
opponent concealed tile; the quads download as a JSON that the backs model
(majsoul_eye/annotate/backs.py) is then re-derived from (per-slot templates
instead of uniform fullwarp cells — captures each tile's true lean, plus the
separated drawn-tile slot the automatic calibration skipped).

Session plan (frames pre-picked from run_8/game1, all 1920x1080 native):
  # 1) one settled all-13 frame -> the 3x13 row slots:
  PYTHONPATH=. python scripts/annotate/calibrate_backs_manual.py captures/raw/ai_session/run_8/game1 28 --pos 1 2 3
  # 2) one first-turn thinking frame per seat -> that seat's separated drawn tile:
  PYTHONPATH=. python scripts/annotate/calibrate_backs_manual.py captures/raw/ai_session/run_8/game1 29 --pos 1 --drawn-only
  PYTHONPATH=. python scripts/annotate/calibrate_backs_manual.py captures/raw/ai_session/run_8/game1 31 --pos 2 --drawn-only
  PYTHONPATH=. python scripts/annotate/calibrate_backs_manual.py captures/raw/ai_session/run_8/game1 21 --pos 3 --drawn-only
  (backup holding frames if one looks mid-sort: pos1 seq39/152, pos2 seq43/156, pos3 seq33/158;
   backup plain13: seq30/155/157)

Page controls (also shown in the page header):
  left-click     record the next corner
  mouse wheel    zoom (to cursor); right-drag = pan
  arrow keys     nudge the LAST recorded point by 1 image px
  u              undo last point (or reopen the previous finished quad)
  下载 JSON      save the session's quads (auto-fires once when all are done)

Click order per tile: 4 corners in SCREEN orientation TL -> TR -> BR -> BL.
Slot order: across = left->right; right/left seats = top->bottom. Progress
survives refresh via localStorage. Save each downloaded JSON into
out/backs_calib/ — the ingest step merges every backs_manual_*.json it finds.
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import cv2

from majsoul_eye.capture.gtframes import load_frames

HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#111;font:14px system-ui,sans-serif}
  /* bottom bar: the top of the frame is the ACROSS hand row — don't cover it */
  #hud{position:fixed;bottom:0;left:0;right:0;z-index:9;background:#000c;color:#8f8;padding:8px 12px;line-height:1.6}
  #task{font-size:17px;font-weight:600}
  #hud button{margin-right:8px;padding:3px 14px;font-size:14px}
  #cv{display:block;cursor:crosshair}
</style></head><body>
<div id="hud">
  <span id="task"></span><br>
  <button id="undo">撤销 (u)</button><button id="dl">下载 JSON</button>
  <button id="reset">清空本页进度</button>
  <span id="hint">左键=点角 | 滚轮=缩放 | 右键拖=平移 | 方向键=微调最后一点 | 进度自动保存(可刷新)</span>
</div>
<canvas id="cv"></canvas>
<script>
const IMG_B64 = "__IMG_B64__";
const TASKS = __TASKS__;              // [[pos, slot], ...] slot "0".."12"|"drawn"
const SOURCE = "__SOURCE__";
const OUT_NAME = "__OUT_NAME__";
const LS_KEY = "backs_calib::" + SOURCE + "::" + OUT_NAME;
const POS_NAME = {1:"右家(从上到下)", 2:"对面(从左到右)", 3:"左家(从上到下)"};
const DRAWN_HINT = {1:"最上方那张分离的牌", 2:"最左侧那张分离的牌", 3:"最下方那张分离的牌"};
const CORNERS = ["左上","右上","右下","左下"];

let quads = {}, ti = 0, pts = [];
try { const s = JSON.parse(localStorage.getItem(LS_KEY));
      if (s) { quads = s.quads||{}; ti = s.ti||0; pts = s.pts||[]; } } catch(e){}

const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const img = new Image();
let scale = 1, ox = 0, oy = 0, cursor = null, panning = null, downAt = null, autoDl = false;

function fit(){
  cv.width = innerWidth; cv.height = innerHeight;
  if (img.width) { scale = Math.min(cv.width/img.width, cv.height/img.height);
                   ox = (cv.width - img.width*scale)/2; oy = (cv.height - img.height*scale)/2; }
  draw();
}
img.onload = fit; addEventListener("resize", fit);
img.src = "data:image/png;base64," + IMG_B64;

const key = i => TASKS[i][0] + ":" + TASKS[i][1];
const done = () => ti >= TASKS.length;
function save(){ localStorage.setItem(LS_KEY, JSON.stringify({quads, ti, pts})); }

function taskText(){
  if (done()) return "全部完成 ✔ 已自动下载 JSON（也可再点“下载 JSON”）";
  const [pos, slot] = TASKS[ti];
  const what = slot === "drawn" ? "摸牌位（" + DRAWN_HINT[pos] + "）"
                                : "第 " + (parseInt(slot)+1) + "/13 张";
  return "pos " + pos + " " + POS_NAME[pos] + " — " + what +
         " — 第 " + (pts.length+1) + "/4 个角（" + CORNERS[pts.length] + "）";
}

function draw(){
  ctx.setTransform(1,0,0,1,0,0); ctx.fillStyle = "#111"; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.setTransform(scale,0,0,scale,ox,oy);
  ctx.drawImage(img,0,0);
  ctx.lineWidth = 1.4/scale;
  for (const k in quads){
    const q = quads[k];
    ctx.strokeStyle = q.source === SOURCE ? "#0e0" : "#0cc";
    ctx.beginPath();
    q.pts.forEach((p,i)=> i ? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1]));
    ctx.closePath(); ctx.stroke();
  }
  ctx.fillStyle = "#f33"; ctx.strokeStyle = "#f33";
  pts.forEach((p,i)=>{ ctx.beginPath(); ctx.arc(p[0],p[1],2.5/scale,0,7); ctx.fill();
                       if(i){ ctx.beginPath(); ctx.moveTo(...pts[i-1]); ctx.lineTo(...p); ctx.stroke(); }});
  if (cursor){                                     // full crosshair for precise aiming
    ctx.strokeStyle = "#ff0a"; ctx.lineWidth = 1/scale; ctx.beginPath();
    ctx.moveTo(cursor[0], 0); ctx.lineTo(cursor[0], img.height);
    ctx.moveTo(0, cursor[1]); ctx.lineTo(img.width, cursor[1]); ctx.stroke();
  }
  document.getElementById("task").textContent =
    taskText() + "   [" + Math.min(ti,TASKS.length) + "/" + TASKS.length + " 格 | 缩放 " + scale.toFixed(1) + "x]";
}

const toImg = (sx,sy) => [(sx-ox)/scale, (sy-oy)/scale];

cv.addEventListener("pointerdown", e => {
  if (e.button === 2) { panning = [e.clientX, e.clientY]; return; }
  if (e.button === 0) downAt = [e.clientX, e.clientY];
});
addEventListener("pointerup", e => {
  if (e.button === 2) { panning = null; return; }
  if (e.button !== 0 || !downAt) return;
  const moved = Math.hypot(e.clientX-downAt[0], e.clientY-downAt[1]); downAt = null;
  if (moved > 4 || done()) return;
  const [x,y] = toImg(e.clientX, e.clientY);
  pts.push([Math.round(x*10)/10, Math.round(y*10)/10]);
  if (pts.length === 4){
    const [pos, slot] = TASKS[ti];
    quads[key(ti)] = {pos: pos, slot: slot, source: SOURCE, pts: pts};
    pts = []; ti++;
    if (done() && !autoDl){ autoDl = true; download(); }
  }
  save(); draw();
});
cv.addEventListener("pointermove", e => {
  if (panning){ ox += e.clientX-panning[0]; oy += e.clientY-panning[1];
                panning = [e.clientX, e.clientY]; }
  cursor = toImg(e.clientX, e.clientY);
  draw();
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.25 : 0.8;
  const ns = Math.min(16, Math.max(0.2, scale*f));
  ox = e.clientX - (e.clientX-ox)*ns/scale; oy = e.clientY - (e.clientY-oy)*ns/scale;
  scale = ns; draw();
}, {passive:false});
cv.addEventListener("contextmenu", e => e.preventDefault());

function undo(){
  if (pts.length) pts.pop();
  else if (ti > 0){ ti--; const old = quads[key(ti)]; delete quads[key(ti)];
                    pts = old ? old.pts.slice(0,3) : []; }
  save(); draw();
}
function nudge(dx,dy){ if (pts.length){ pts[pts.length-1][0]+=dx; pts[pts.length-1][1]+=dy; save(); draw(); } }
addEventListener("keydown", e => {
  if (e.key === "u") undo();
  else if (e.key === "ArrowLeft"){ nudge(-1,0); e.preventDefault(); }
  else if (e.key === "ArrowRight"){ nudge(1,0); e.preventDefault(); }
  else if (e.key === "ArrowUp"){ nudge(0,-1); e.preventDefault(); }
  else if (e.key === "ArrowDown"){ nudge(0,1); e.preventDefault(); }
});
document.getElementById("undo").onclick = undo;
document.getElementById("reset").onclick = () => {
  if (confirm("清空本页全部进度？")) { quads={}; ti=0; pts=[]; save(); draw(); } };
function download(){
  const payload = {meta:{space:"original px @1920x1080", source:SOURCE}, quads:quads};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(payload,null,1)], {type:"application/json"}));
  a.download = OUT_NAME; a.click();
}
document.getElementById("dl").onclick = download;
</script></body></html>
"""


def canon_quad(pts):
    """Reorder 4 points to visual-clockwise starting at the top-left-most corner
    (min x+y) — guards against click-order variation; matches crop_quad's
    [TL,TR,BR,BL] convention. Image coords (y down): increasing atan2 = visual CW."""
    import numpy as np
    p = np.float32(pts)
    c = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
    p = p[order]
    start = int(np.argmin(p.sum(axis=1)))
    return np.roll(p, -start, axis=0)


def ingest(calib_dir: str) -> None:
    """Merge every backs_manual_*.json in `calib_dir`, canonicalize corners, map
    original px -> fullwarp, validate (per-seat spacing uniformity, drawn gap,
    comparison vs the automatic BACK_ROWS fit) and print a ready-to-paste
    constants block for majsoul_eye/annotate/backs.py."""
    import glob as g

    import numpy as np

    from majsoul_eye.annotate import pipeline as P
    from majsoul_eye.annotate.backs import BACK_ROWS

    hom = P.build_homographies(1920, 1080)
    quads: dict = {}
    for f in sorted(g.glob(os.path.join(calib_dir, "backs_manual_*.json"))):
        with open(f, encoding="utf-8") as fh:
            quads.update(json.load(fh)["quads"])     # later files win
    rows: dict = {}
    drawn: dict = {}
    for q in quads.values():
        fw = P.original_to_fullwarp(canon_quad(q["pts"]), hom["H_full"])
        fw = [[round(float(x), 1) for x in p] for p in fw]
        if q["slot"] == "drawn":
            drawn[q["pos"]] = fw
        else:
            rows.setdefault(q["pos"], {})[int(q["slot"])] = fw

    for pos in sorted(rows):
        r = BACK_ROWS[pos]
        ai = 0 if r["along"] == "x" else 1
        slots = rows[pos]
        cents = {i: np.float32(slots[i]).mean(axis=0) for i in sorted(slots)}
        gaps = [abs(cents[i + 1][ai] - cents[i][ai]) for i in sorted(cents) if i + 1 in cents]
        alongs = [p[ai] for i in slots for p in slots[i]]
        crosses = [p[1 - ai] for i in slots for p in slots[i]]
        print(f"pos {pos}: n={len(slots)}  centroid pitch fw={np.mean(gaps):.2f} "
              f"(min {min(gaps):.1f} max {max(gaps):.1f})  vs auto {r['pitch']:.2f}   "
              f"along [{min(alongs):.0f},{max(alongs):.0f}] vs auto "
              f"[{r['along0']:.0f},{r['along1']:.0f}]   cross [{min(crosses):.0f},{max(crosses):.0f}] "
              f"vs auto [{r['cross'][0]:.0f},{r['cross'][1]:.0f}]")
        if pos in drawn:
            dc = np.float32(drawn[pos]).mean(axis=0)
            last = cents[max(cents)]
            print(f"  drawn: centroid gap beyond slot12 = {abs(dc[ai]-last[ai])-np.mean(gaps):+.1f} "
                  f"(vs auto DRAWN_GAP=25)")

    sources = sorted({q["source"] for q in quads.values()})
    lines = [
        '"""GENERATED by scripts/annotate/calibrate_backs_manual.py --ingest',
        "— do not edit by hand; re-click + re-ingest instead.",
        "",
        "Manually clicked opponent hand-row tile-back quads (FULLWARP px, canonical",
        "corner order TL,TR,BR,BL visual-clockwise). Slot 0 = the anchored end",
        "(player's left); BACK_DRAWN_QUADS = the separated drawn-tile slot at the",
        f"13-row moving end. Source frames: {', '.join(sources)}.",
        '"""',
        "",
        "BACK_SLOT_QUADS = {",
    ]
    for pos in sorted(rows):
        lines.append(f"    {pos}: [")
        lines += [f"        {rows[pos][i]}," for i in sorted(rows[pos])]
        lines.append("    ],")
    lines.append("}")
    lines.append("")
    lines.append("BACK_DRAWN_QUADS = {")
    lines += [f"    {pos}: {drawn[pos]}," for pos in sorted(drawn)]
    lines.append("}")
    out = "majsoul_eye/annotate/_backs_manual.py"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}  ({sum(len(v) for v in rows.values())} row quads + "
          f"{len(drawn)} drawn quads)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingest", action="store_true",
                    help="merge out/backs_calib/backs_manual_*.json -> validate + print "
                         "the fullwarp per-slot constants block (no GUI)")
    ap.add_argument("frames_dir", nargs="?", help="game dir containing frames.jsonl")
    ap.add_argument("seq", type=int, nargs="?", help="frame seq to annotate on")
    ap.add_argument("--pos", type=int, nargs="+", choices=(1, 2, 3),
                    help="screen positions to annotate (1=right 2=across 3=left)")
    ap.add_argument("--drawn-only", action="store_true",
                    help="only the separated drawn tile of each --pos (holding frames)")
    ap.add_argument("--out-dir", default="out/backs_calib",
                    help="where the generated HTML lands (save the downloaded JSONs here too)")
    ap.add_argument("--no-open", action="store_true", help="only generate, don't open the browser")
    args = ap.parse_args()

    if args.ingest:
        ingest(args.out_dir)
        return
    if args.frames_dir is None or args.seq is None or not args.pos:
        ap.error("frames_dir, seq and --pos are required (unless --ingest)")

    frames = load_frames(args.frames_dir)
    if args.seq not in frames:
        raise SystemExit(f"seq {args.seq} has no ok frame in {args.frames_dir}")
    img = cv2.imread(frames[args.seq])
    if img is None:
        raise SystemExit(f"cannot read {frames[args.seq]}")
    if (img.shape[1], img.shape[0]) != (1920, 1080):
        print(f"resizing {img.shape[1]}x{img.shape[0]} -> 1920x1080 (calibration space)")
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit("png encode failed")

    tasks = [[p, s] for p in args.pos
             for s in (["drawn"] if args.drawn_only else [str(i) for i in range(13)])]
    tag = f"seq{args.seq}_pos{''.join(map(str, args.pos))}" + ("_drawn" if args.drawn_only else "")
    source = f"{args.frames_dir.replace(os.sep, '/')}#{args.seq}"
    out_name = f"backs_manual_{tag}.json"

    html = (HTML_TEMPLATE
            .replace("__TITLE__", f"backs calib {tag}")
            .replace("__IMG_B64__", base64.b64encode(buf.tobytes()).decode())
            .replace("__TASKS__", json.dumps(tasks))
            .replace("__SOURCE__", source)
            .replace("__OUT_NAME__", out_name))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"backs_calib_{tag}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{len(tasks)} quads to click; annotator -> {path}")
    print(f"finish -> browser downloads {out_name}; move it into {args.out_dir}/")
    if not args.no_open:
        os.startfile(os.path.abspath(path))  # windows default browser


if __name__ == "__main__":
    main()
