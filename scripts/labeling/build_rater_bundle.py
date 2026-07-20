"""Build a self-contained HTML rating bundle from a rendered window set.

Input : a directory produced by render_context.py  (images/ + manifest.csv)
Output: rating_bundle_<name>.zip  =  index.html + images/ + README.txt

The rater unzips it, double-clicks index.html, labels, clicks Export, and sends back one CSV.
No install, no server, works offline and behind a firewall.

The window list is BAKED INTO the HTML as a JS array rather than fetched: a page opened from file://
cannot fetch a sibling CSV (browsers block it as cross-origin), so fetching would silently show a
blank page on the rater's machine.
"""
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

from neurodent.results.scoring import CATEGORIES, DEFAULT_CATEGORY, LABEL_COL_PREFIX

README = """\
EEG window rating

1. Unzip this folder somewhere.
2. Double-click index.html. It opens in your browser. Nothing to install.
3. Type your name when asked.
4. For each window: every channel starts as "clean". Change only the channels that are not clean,
   then press Next (or the right arrow key).
5. Progress saves automatically. The bar/strip under the toolbar shows which windows you have seen
   (grey = not seen, green = seen, orange = flagged); click a square to jump, or "Next unseen" to
   pick up where you left off. You can close the tab and come back.
6. When you are done (or want to send progress so far), click "Export CSV" and email the file back.
   To continue on another computer, open the page there and click "Import CSV" on that file.

Click "Rubric" at any time for what the categories mean.
"""

RUBRIC = [
    ("clean", "Normal EEG for that channel. Keep it. This is the default."),
    ("bad", "Artifact: contamination that is not brain activity. Movement, muscle, electrode pop or "
            "step, flatline or disconnection, saturation. NOT plain mains hum: these traces are "
            "already notched at 60 Hz, so it is gone before you see it. Only gross residual mains "
            "contamination counts, and it shows as harmonics (120/180 Hz) in the PSD panel."),
    ("event", "Real brain activity that looks dramatic: epileptiform discharge, spike train, seizure. "
              "It is abnormal but it is signal, not noise. KEEP it. Use the wide overview strip at "
              "the top to see whether the thing evolves over time (an event does; an artifact "
              "usually does not)."),
    ("unsure", "You genuinely cannot tell. Use it. Do not guess."),
]

HTML = """<!doctype html>
<meta charset="utf-8">
<title>EEG window rating</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;margin:0;padding:12px 16px;color:#111}
 header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;border-bottom:1px solid #ddd;padding-bottom:8px}
 h1{font-size:15px;margin:0;font-weight:600}
 .meta{color:#555;font-size:13px}
 button{font:13px system-ui;padding:5px 10px;border:1px solid #bbb;background:#fafafa;border-radius:4px;cursor:pointer}
 button:hover{background:#f0f0f0}
 .main{display:flex;gap:10px;align-items:flex-start;margin-top:8px}
 /* Capping max-width keeps the stage shrink-wrapped to the image, so the bands stay glued to it,
    while leaving room for the control column. */
 .stage{position:relative;line-height:0;flex:none}
 .stage img{max-width:calc(100vw - 300px);max-height:86vh;width:auto;height:auto;display:block;
            border:1px solid #eee;cursor:zoom-in}
 /* One band per channel, over the rows matplotlib reported (geometry.json). Lights up when the rater
    is on that channel's buttons, so the row and its controls are visibly the same thing. */
 .bands{position:absolute;inset:0;pointer-events:none}
 .band{position:absolute;left:0;right:0;transition:background .07s}
 .band.hot{background:rgba(255,176,0,.15);outline:1px solid rgba(230,150,0,.55)}
 /* Buttons sit level with their channel's row, not in a table to correlate against. */
 .ctrls{position:relative;width:250px;flex:none}
 .rowctl{position:absolute;left:0;right:0;transform:translateY(-50%);display:flex;align-items:center;
         gap:2px;padding:2px 3px;border-radius:4px}
 .rowctl:hover{background:#f6f6f6}
 .rowctl .chn{font:11px ui-monospace,monospace;color:#555;width:24px;flex:none;text-align:right;
              overflow:hidden;white-space:nowrap}
 .opt{display:inline-block;padding:3px 6px;border:1px solid #ccc;border-radius:3px;
      cursor:pointer;font-size:11px;background:#fff;user-select:none;line-height:1.2}
 .opt.sel[data-v=clean] {background:#e8f5e9;border-color:#4c9a54;font-weight:600}
 .opt.sel[data-v=bad]   {background:#fdecea;border-color:#c0392b;font-weight:600}
 .opt.sel[data-v=event] {background:#fff4e0;border-color:#d68910;font-weight:600}
 .opt.sel[data-v=unsure]{background:#eceff1;border-color:#607d8b;font-weight:600}
 .quick,.nav{display:flex;gap:5px;align-items:center}
 .sep{width:1px;height:20px;background:#ddd}
 #ovl,#zoom{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;overflow:auto;padding:40px;z-index:9}
 #ovl .box{background:#fff;max-width:760px;margin:0 auto;padding:20px 24px;border-radius:6px}
 #ovl h2{font-size:15px;margin:0 0 10px}
 #ovl dt{font-weight:600;margin-top:10px}
 #ovl img{max-width:100%;border:1px solid #ddd;margin-top:6px}
 #keys{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;overflow:auto;padding:40px;z-index:9}
 #keys .box{background:#fff;max-width:420px;margin:0 auto;padding:20px 24px;border-radius:6px}
 #keys h2{font-size:15px;margin:0 0 10px} #keys dt{font-weight:600;margin-top:10px}
 .rightcol{display:flex;flex-direction:column;gap:6px;flex:none}
 #zoom{background:rgba(0,0,0,.85);padding:0;cursor:zoom-out}
 #zoom img{max-width:none;width:auto;display:block}
 .warn{color:#b34700;font-size:12px}
 /* progress: a bar (seen fraction) + a per-window strip you can click to jump. */
 .prog2{display:flex;align-items:center;gap:10px;margin:6px 0 0}
 .pbar{flex:none;width:160px;height:8px;background:#eee;border-radius:4px;overflow:hidden}
 .pfill{height:100%;width:0;background:#4c9a54;transition:width .1s}
 .strip{display:flex;gap:1px;flex-wrap:wrap;flex:1;min-width:0}
 .tick{width:9px;height:14px;border-radius:2px;background:#e0e0e0;cursor:pointer}  /* unseen */
 .tick.seen{background:#bfe3c4}                 /* seen, all clean */
 .tick.flag{background:#e6a23c}                 /* seen, has a non-clean label */
 .tick.cur{outline:2px solid #333;outline-offset:0}
</style>

<header>
  <h1>EEG window rating</h1>
  <span class="meta" id="who"></span>
  <span class="meta" id="prog"></span>
  <span class="meta" id="seen"></span>
  <span class="meta" id="where"></span>
  <span class="sep"></span>
  <span class="nav">
    <button id="prev">&larr; Prev</button>
    <button id="next">Next &rarr;</button>
    <button id="jumpUnseen" title="Jump to the first window you have not seen yet">Next unseen</button>
  </span>
  <span class="meta warn" id="unsaved"></span>
  <span style="flex:1"></span>
  <button id="scaleBtn" title="Toggle per-channel / shared amplitude scale (Space)">Scale: split</button>
  <button id="keysBtn" title="Keyboard shortcuts">Hotkeys</button>
  <button id="rubricBtn">Rubric</button>
  <button id="expBtn">Export CSV</button>
  <button id="impBtn">Import CSV</button>
  <input type="file" id="impFile" accept=".csv" style="display:none">
</header>
<p class="meta" style="margin:6px 0 0">Each channel's buttons sit level with that channel's trace.
   Everything starts as <b>clean</b>; change only what is not, then Next. Arrow keys navigate.
   Click the figure to enlarge.</p>
<div class="prog2">
  <span class="pbar"><span class="pfill" id="pfill"></span></span>
  <div class="strip" id="strip" title="one square per window; grey = not seen, green = seen, orange = flagged. Click to jump."></div>
</div>

<div class="main">
  <div class="stage" id="stage">
    <img id="img" alt="window">
    <div class="bands" id="bands"></div>
  </div>
  <div class="rightcol">
    <div class="ctrls" id="ctrls"></div>
    <span class="quick">
      <button data-all="clean">All clean</button>
      <button data-all="bad">All bad</button>
      <button data-all="event">All event</button>
    </span>
  </div>
</div>

<div id="zoom"><img id="zoomImg" alt="full size"></div>

<div id="ovl"><div class="box">
  <h2>Rubric</h2><dl id="rubric"></dl>
  <button id="closeOvl">Close</button>
</div></div>

<div id="keys"><div class="box">
  <h2>Shortcuts</h2>
  <dl>
    <dt>hover a channel, then 1 / 2 / 3 / 4</dt><dd>clean / bad / event / unsure</dd>
    <dt>&larr; &rarr;</dt><dd>previous / next window</dd>
    <dt>Space</dt><dd>toggle per-channel / shared amplitude scale</dd>
  </dl>
  <button id="closeKeys">Close</button>
</div></div>

<script>
const WINDOWS  = __WINDOWS__;
const CHANNELS = __CHANNELS__;
const CATS     = __CATS__;
const DEFAULT  = __DEFAULT__;
const RUBRIC   = __RUBRIC__;
const BUNDLE   = __BUNDLE__;
const LABEL_PREFIX = __LABEL_PREFIX__;
const GEOM     = __GEOM__;    // {recording: {rows:[{channel, top, bottom}]}} fractions from image top

// Channel names share a long prefix/suffix ("EEG E13-REF2") that truncates to a useless "EEG E1..."
// in a narrow column. Strip whatever all channels share -> "13". The full name is on the trace row
// and in the tooltip.
const shortName = (() => {
  if (CHANNELS.length < 2) return c => c;
  const a = CHANNELS[0];
  let p = 0;
  while (p < a.length && CHANNELS.every(c => c[p] === a[p])) p++;
  let s = 0;
  while (s < a.length - p && CHANNELS.every(c => c[c.length-1-s] === a[a.length-1-s])) s++;
  return c => c.slice(p, c.length - s) || c;
})();

// Where channel `ch` sits, as fractions of image height (from geometry.json, required at build).
function rowBox(recording, ch){
  return GEOM[recording].rows.find(r => r.channel === ch);
}
// The channels THIS recording actually has (from its geometry). A mixed bundle spans recordings with
// different channel counts, so per-window logic keys on this -- NOT the global (union) CHANNELS list,
// which may include slots a given recording doesn't have.
function channelsOf(recording){
  return GEOM[recording].rows.map(r => r.channel);
}

const KEY = "eegrate:" + BUNDLE;
let rater = localStorage.getItem(KEY + ":rater") || "";
while (!rater) { rater = (prompt("Your name or initials:") || "").trim(); }
localStorage.setItem(KEY + ":rater", rater);

// deterministic per-rater shuffle: raters see different orders, scoring keys on window id anyway
function hash(s){ let h=2166136261; for(const c of s) h=Math.imul(h^c.charCodeAt(0),16777619); return h>>>0; }
let seed = hash(rater) || 1;
const rnd = () => (seed = (seed*1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const order = WINDOWS.map((_,i)=>i);
for (let i=order.length-1; i>0; i--){ const j=Math.floor(rnd()*(i+1)); [order[i],order[j]]=[order[j],order[i]]; }
const dispPos = [];                        // window array index -> its position in this rater's order
order.forEach((wi, dp) => { dispPos[wi] = dp; });

const wkey = w => w.recording + "|" + w.window;
let state = JSON.parse(localStorage.getItem(KEY + ":state") || "{}");   // {wkey: {ch: cat}}
let secs  = JSON.parse(localStorage.getItem(KEY + ":secs")  || "{}");   // {wkey: seconds}
let seen  = JSON.parse(localStorage.getItem(KEY + ":seen")  || "{}");   // {wkey: 1}  windows reviewed
let dirty = localStorage.getItem(KEY + ":dirty") === "1";
let pos   = parseInt(localStorage.getItem(KEY + ":pos") || "0", 10) || 0;
let shownAt = Date.now();

const isFlagged = w => { const l = state[wkey(w)]; return !!l && channelsOf(w.recording).some(c => l[c] && l[c] !== "clean"); };
const firstUnseenPos = () => { for (let dp = 0; dp < order.length; dp++) if (!seen[wkey(WINDOWS[order[dp]])]) return dp; return -1; };
// A window is "seen" only when the rater ENGAGES it (advances away, or labels it) -- NOT merely by
// being rendered. Otherwise landing on the resume window (import / reload) would fabricate an
// all-clean "reviewed" judgment the rater never made, and export/import would not be idempotent.
const markSeen = () => { seen[wkey(WINDOWS[order[pos]])] = 1; };

let hovered = null;                                    // channel row under the cursor, for the 1/2/3/4 keys
const HAS_SHARED = __HAS_SHARED__;
let shared = false;                                    // per-channel (false) vs one shared amplitude scale
const imgOf = w => "images/" + (shared ? w.image.replace(".png", "__shared.png") : w.image);
function setScale(on){
  if (!HAS_SHARED) return;
  shared = on;
  document.getElementById("img").src = imgOf(WINDOWS[order[pos]]);
  document.getElementById("scaleBtn").textContent = "Scale: " + (shared ? "shared" : "split");
}
function setLabel(ch, cat){                             // in-place row update; no full re-render, so the
  markSeen();                                          // hovered channel stays live for rapid key entry
  const lab = labelsFor(WINDOWS[order[pos]]);
  lab[ch] = cat; dirty = true;
  const rc = [...document.querySelectorAll(".rowctl")].find(r => r.querySelector(".chn").title === ch);
  if (rc) rc.querySelectorAll(".opt").forEach(o => o.classList.toggle("sel", o.dataset.v === cat));
  refreshProgress(); save();
}

function labelsFor(w){
  const k = wkey(w);
  if (!state[k]) { state[k] = {}; channelsOf(w.recording).forEach(c => state[k][c] = DEFAULT); }
  return state[k];
}
function save(){
  localStorage.setItem(KEY + ":state", JSON.stringify(state));
  localStorage.setItem(KEY + ":secs",  JSON.stringify(secs));
  localStorage.setItem(KEY + ":seen",  JSON.stringify(seen));
  localStorage.setItem(KEY + ":pos",   String(pos));
  localStorage.setItem(KEY + ":dirty", dirty ? "1" : "0");
  document.getElementById("unsaved").textContent = dirty ? "unexported changes" : "";
}

// ---- progress: strip (one tick per window, click to jump) + bar + counts ----
const strip = document.getElementById("strip");
const ticks = order.map((wi, dp) => {
  const t = document.createElement("div");
  t.className = "tick";
  t.title = "window " + WINDOWS[wi].window;
  t.onclick = () => { markSeen(); stampTime(); pos = dp; render(); };
  strip.appendChild(t);
  return t;
});
function tickClass(dp){
  const w = WINDOWS[order[dp]], k = wkey(w);
  return "tick" + (seen[k] ? (isFlagged(w) ? " flag" : " seen") : "") + (dp === pos ? " cur" : "");
}
function refreshProgress(){
  ticks.forEach((t, dp) => { const c = tickClass(dp); if (t.className !== c) t.className = c; });
  const nSeen = order.reduce((a, wi) => a + (seen[wkey(WINDOWS[wi])] ? 1 : 0), 0);
  const nFlag = order.reduce((a, wi) => a + (isFlagged(WINDOWS[wi]) ? 1 : 0), 0);
  document.getElementById("pfill").style.width = (100 * nSeen / order.length) + "%";
  document.getElementById("seen").textContent = "seen " + nSeen + "/" + order.length +
    "  •  flagged " + nFlag;
}
function stampTime(){
  const w = WINDOWS[order[pos]], k = wkey(w);
  secs[k] = (secs[k] || 0) + Math.round((Date.now() - shownAt) / 1000);
  shownAt = Date.now();
}

// Rows are positioned in % of the control column's height, so it must match the image's RENDERED
// height, which depends on the viewport.
function syncHeight(){
  const img = document.getElementById("img");
  if (img.clientHeight) document.getElementById("ctrls").style.height = img.clientHeight + "px";
}
document.getElementById("img").addEventListener("load", syncHeight);
window.addEventListener("resize", syncHeight);

function render(){
  const w = WINDOWS[order[pos]];
  document.getElementById("img").src = imgOf(w);
  document.getElementById("who").textContent   = "rater: " + rater;
  document.getElementById("prog").textContent  = (pos+1) + " / " + WINDOWS.length;
  document.getElementById("where").textContent = "item " + (hash(wkey(w)) % 46656).toString(36);  // opaque id (hides recording/genotype/time)

  const lab = labelsFor(w);
  const bands = document.getElementById("bands"), ctrls = document.getElementById("ctrls");
  bands.innerHTML = ""; ctrls.innerHTML = "";

  channelsOf(w.recording).forEach((ch) => {               // only THIS window's real channels (mixed bundle)
    const box = rowBox(w.recording, ch);

    const band = document.createElement("div");            // highlight over that channel's trace
    band.className = "band";
    band.style.top    = (box.top * 100) + "%";
    band.style.height = ((box.bottom - box.top) * 100) + "%";
    bands.appendChild(band);

    const rc = document.createElement("div");              // its buttons, level with it
    rc.className = "rowctl";
    rc.style.top = (((box.top + box.bottom) / 2) * 100) + "%";
    rc.onmouseenter = () => { band.classList.add("hot"); hovered = ch; };
    rc.onmouseleave = () => { band.classList.remove("hot"); if (hovered === ch) hovered = null; };

    const nm = document.createElement("span");
    nm.className = "chn"; nm.textContent = shortName(ch); nm.title = ch;
    rc.appendChild(nm);

    CATS.forEach(cat => {
      const b = document.createElement("span");
      b.className = "opt" + (lab[ch] === cat ? " sel" : "");
      b.dataset.v = cat; b.textContent = cat;
      b.onclick = () => setLabel(ch, cat);
      rc.appendChild(b);
    });
    ctrls.appendChild(rc);
  });
  syncHeight();
  refreshProgress();
  save();
}
function go(d){ markSeen(); stampTime(); pos = Math.min(WINDOWS.length-1, Math.max(0, pos+d)); render(); }
document.getElementById("jumpUnseen").onclick = () => {
  const dp = firstUnseenPos();
  if (dp < 0) { alert("You have seen every window."); return; }
  markSeen(); stampTime(); pos = dp; render();
};

document.querySelectorAll("[data-all]").forEach(b => b.onclick = () => {
  markSeen();
  const w = WINDOWS[order[pos]];
  const lab = labelsFor(w);
  channelsOf(w.recording).forEach(c => lab[c] = b.dataset.all);   // only this window's real channels
  dirty = true; save(); render();
});
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(+1);

// Zoom is an overlay, not an inline resize, which would break the row alignment.
const zoom = document.getElementById("zoom");
document.getElementById("img").onclick = () => {
  document.getElementById("zoomImg").src = document.getElementById("img").src;
  zoom.style.display = "block";
};
zoom.onclick = () => { zoom.style.display = "none"; };

const KEYCAT = {}; CATS.forEach((c, i) => KEYCAT[String(i + 1)] = c);   // 1/2/3/4 -> the categories in order
document.onkeydown = e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "Escape") { zoom.style.display = "none"; document.getElementById("ovl").style.display = "none"; document.getElementById("keys").style.display = "none"; return; }
  if (e.key === " ") { e.preventDefault(); setScale(!shared); return; }         // toggle scale
  if (e.key in KEYCAT) { if (hovered) setLabel(hovered, KEYCAT[e.key]); return; } // label the hovered channel
  if (e.key === "ArrowRight" || e.key === "Enter") go(+1);
  if (e.key === "ArrowLeft") go(-1);
};
document.getElementById("scaleBtn").onclick = () => setScale(!shared);
if (!HAS_SHARED) document.getElementById("scaleBtn").style.display = "none";
document.getElementById("keysBtn").onclick = () => document.getElementById("keys").style.display = "block";
document.getElementById("closeKeys").onclick = () => document.getElementById("keys").style.display = "none";

// ---- rubric ----
const dl = document.getElementById("rubric");
RUBRIC.forEach(([name, desc]) => {
  dl.insertAdjacentHTML("beforeend", "<dt>" + name + "</dt><dd>" + desc +
    '</dd><dd><img src="images/rubric/' + name + '.png" onerror="this.remove()"></dd>');
});
document.getElementById("rubricBtn").onclick = () => document.getElementById("ovl").style.display = "block";
document.getElementById("closeOvl").onclick  = () => document.getElementById("ovl").style.display = "none";

// ---- export / import: the CSV is the complete save file (every window, seen flag, display order) ----
const HEAD = ["image","recording","window","t_start_s","display_order"]
  .concat(CHANNELS.map(c => LABEL_PREFIX + c))
  .concat(["seen","rater","seconds_spent","ts"]);
function exportCsv(){
  stampTime();
  const now = new Date().toISOString();
  const lines = [HEAD.join(",")];
  WINDOWS.forEach((w, wi) => {                          // EVERY window, so nothing is ambiguous
    const k = wkey(w), sv = seen[k] ? 1 : 0, lab = state[k];
    const has = new Set(channelsOf(w.recording));        // slots this window's recording actually has
    // blank = not seen (!= clean); also blank for union slots this window doesn't have (mixed bundle)
    const labels = CHANNELS.map(c => (sv && lab && has.has(c)) ? (lab[c] || "") : "");
    const row = [w.image, w.recording, w.window, w.t_start_s, dispPos[wi]]
      .concat(labels)
      .concat([sv, rater, secs[k] || 0, now]);
    lines.push(row.map(v => /[",]/.test(String(v)) ? '"' + String(v).replace(/"/g,'""') + '"' : v).join(","));
  });
  const blob = new Blob([lines.join("\\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_" + BUNDLE + "_" + rater.replace(/\\W+/g,"") + ".csv";
  a.click();
  dirty = false; save();
}
document.getElementById("expBtn").onclick = exportCsv;
document.getElementById("impBtn").onclick = () => document.getElementById("impFile").click();
document.getElementById("impFile").onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    const rows = rd.result.split(/\\r?\\n/).filter(Boolean);
    const head = rows.shift().split(",");
    const iRec = head.indexOf("recording"), iWin = head.indexOf("window");
    const iSec = head.indexOf("seconds_spent"), iSeen = head.indexOf("seen");
    let n = 0;
    rows.forEach(line => {
      const v = line.split(",");
      const k = v[iRec] + "|" + parseInt(v[iWin], 10);
      const lab = {};
      CHANNELS.forEach(c => { const i = head.indexOf(LABEL_PREFIX + c); if (i >= 0 && v[i]) lab[c] = v[i]; });
      const wasSeen = v[iSeen] === "1" || v[iSeen] === "true";
      if (wasSeen) {
        seen[k] = 1;
        state[k] = Object.keys(lab).length ? lab : (() => { const o = {}; CHANNELS.forEach(c => o[c] = DEFAULT); return o; })();
        n++;
      }
      if (iSec >= 0 && v[iSec]) secs[k] = parseInt(v[iSec], 10) || 0;
    });
    dirty = false;
    const dp = firstUnseenPos(); pos = dp >= 0 ? dp : 0;      // resume at the first window not yet seen
    save(); render();
    alert("Restored " + n + " seen windows; resuming at your first unseen window.");
  };
  rd.readAsText(f);
};
window.onbeforeunload = () => dirty ? "You have unexported changes." : undefined;

render();
</script>
"""


def build(rendered_dir, out_zip=None, name=None):
    rendered_dir = Path(rendered_dir)
    manifest = rendered_dir / "manifest.csv"
    img_dir = rendered_dir / "images"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest}; run render_context.py first")

    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("manifest is empty")
    channels = [c[len(LABEL_COL_PREFIX):] for c in rows[0] if c.startswith(LABEL_COL_PREFIX)]
    windows = [{"image": r["image"], "recording": r["recording"],
                "window": int(r["window"]), "t_start_s": r["t_start_s"]} for r in rows]

    missing = [w["image"] for w in windows if not (img_dir / w["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest images missing from {img_dir}, e.g. {missing[:3]}")
    if len({w["image"] for w in windows}) != len(windows):
        raise SystemExit("duplicate image filenames in manifest (recording prefix collision)")

    # Shared-scale twin per window (for the rater's Space toggle). Present iff every window has one.
    shared_name = lambda img: img.replace(".png", "__shared.png")
    has_shared = all((img_dir / shared_name(w["image"])).exists() for w in windows)

    # Where each channel's trace landed, so the page puts that channel's buttons level with it.
    gpath = rendered_dir / "geometry.json"
    if not gpath.exists():
        raise SystemExit(f"no geometry.json in {rendered_dir}; re-run render_context.py")
    geometry = json.loads(gpath.read_text())
    ungeom = sorted({w["recording"] for w in windows} - set(geometry))
    if ungeom:
        raise SystemExit(f"geometry.json is missing recording(s): {ungeom}")

    name = name or rendered_dir.name
    html = (HTML
            .replace("__WINDOWS__", json.dumps(windows))
            .replace("__CHANNELS__", json.dumps(channels))
            .replace("__CATS__", json.dumps(CATEGORIES))
            .replace("__DEFAULT__", json.dumps(DEFAULT_CATEGORY))
            .replace("__RUBRIC__", json.dumps(RUBRIC))
            .replace("__BUNDLE__", json.dumps(name))
            .replace("__LABEL_PREFIX__", json.dumps(LABEL_COL_PREFIX))
            .replace("__HAS_SHARED__", json.dumps(has_shared))
            .replace("__GEOM__", json.dumps(geometry)))

    stage = rendered_dir / "_bundle"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "images").mkdir(parents=True)
    for w in windows:
        shutil.copy2(img_dir / w["image"], stage / "images" / w["image"])
        if has_shared:
            shutil.copy2(img_dir / shared_name(w["image"]), stage / "images" / shared_name(w["image"]))
    rubric_src = img_dir / "rubric"
    if rubric_src.is_dir():
        shutil.copytree(rubric_src, stage / "images" / "rubric")
    (stage / "index.html").write_text(html)
    (stage / "README.txt").write_text(README)

    out_zip = Path(out_zip or rendered_dir / f"rating_bundle_{name}.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    shutil.rmtree(stage)

    mb = out_zip.stat().st_size / 1e6
    print(f"{out_zip}  ({len(windows)} windows, {len(channels)} channels, {mb:.1f} MB)")
    return out_zip


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "results/labeling_pilot")
