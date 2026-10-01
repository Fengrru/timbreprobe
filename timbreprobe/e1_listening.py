"""E1 -- perceptual monotonicity listening test material.

Protocol (triplet 2AFC)
-----------------------
Each trial plays a reference A and two options; the listener answers which
option sounds *more similar to A*.  The two options are drawn so that their
embedding distances to A fall in different quantile bins of the pool's
distance distribution -- that is what turns the task into a measurement of
monotonicity: if the embedding tracks human similarity, agreement with the
embedding's ordering should be well above chance, and the psychometric curve
over distance-contrast levels should be monotone.

The generated `listening_test.html` is self-contained (triplets embedded in
the page, audio as sibling .wav files -- no server and no CORS needed) and
never contains the answer key; `key.json` is written separately so the
experimenter (not the page) holds the ground truth.

Analysis: `analyze(responses, material_dir)` joins participant response files
with the key and reports agreement per contrast level with Wilson intervals.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from . import features, synth
from .config import Config


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_e1_material(cfg: Config, corpus_dir: Path, out_dir: Path,
                      n_triplets: int | None = None, pool_size: int = 600,
                      e1_dur: float = 0.5, seed: int = 0) -> dict:
    """Sample stratified triplets, render their audio, write HTML + key."""
    from .corpus import load_corpus

    out_dir.mkdir(parents=True, exist_ok=True)
    n_triplets = n_triplets or cfg.e1_n_triplets
    corpus, _ = load_corpus(corpus_dir)
    rng = np.random.default_rng(seed)

    pool_idx = rng.choice(corpus.u.shape[0], size=min(pool_size, corpus.u.shape[0]),
                          replace=False)
    u_pool = torch.as_tensor(corpus.u[pool_idx], dtype=torch.float32)

    # render the pool at the listening duration and embed *that* audio, so the
    # distances the test relies on refer to exactly what the listener hears
    with torch.no_grad():
        w = synth.render(u_pool, sr=cfg.sr, dur=e1_dur, chunk_b=cfg.chunk_b)
        feats = features.embed(w, synth.reference_f0(u_pool), sr=cfg.sr)
    stats = features.fit_normalizer(feats)
    z = features.normalize(feats, stats)
    d = torch.cdist(z, z).numpy()
    iu = np.triu_indices(len(pool_idx), k=1)
    pair_d = d[iu]
    edges = np.quantile(pair_d, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0], edges[-1] = edges[0] - 1e-6, edges[-1] + 1e-6

    def bin_of(value: float) -> int:
        return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, 3))

    bin_mat = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, 3)
    np.fill_diagonal(bin_mat, -1)
    used = np.zeros_like(bin_mat, dtype=bool)
    idx_all = np.arange(len(pool_idx))

    triplets: list[dict] = []
    attempts = 0
    while len(triplets) < n_triplets and attempts < n_triplets * 200:
        attempts += 1
        a = int(rng.integers(0, len(pool_idx)))
        b1, b2 = rng.choice(4, size=2, replace=False)
        row = bin_mat[a]
        free = ~used[a]
        cand1 = idx_all[(row == b1) & free]
        cand2 = idx_all[(row == b2) & free]
        if not len(cand1) or not len(cand2):
            continue
        b = int(rng.choice(cand1))
        c = int(rng.choice(cand2))
        used[a, b] = used[b, a] = True
        used[a, c] = used[c, a] = True
        # the closer option goes to a random side to decorrelate position
        closer_is_b = d[a, b] < d[a, c]
        side_of_b = int(rng.integers(1, 3))
        triplets.append({
            "a": a, "b": b, "c": c,
            "d_ab": float(d[a, b]), "d_ac": float(d[a, c]),
            "bin_ab": bin_of(d[a, b]), "bin_ac": bin_of(d[a, c]),
            "side_of_b": side_of_b,
            "answer_side": side_of_b if closer_is_b else (3 - side_of_b),
        })

    # ---- render audio ------------------------------------------------------
    sel = sorted({t["a"] for t in triplets} | {t["b"] for t in triplets}
                 | {t["c"] for t in triplets})
    with torch.no_grad():
        wav = synth.render(u_pool[sel], sr=cfg.sr, dur=e1_dur, chunk_b=cfg.chunk_b).numpy()
    import soundfile as sf
    for i, orig in enumerate(sel):
        sig = wav[i] / max(1e-6, float(np.abs(wav[i]).max())) * 0.9
        sf.write(out_dir / f"p{orig:04d}.wav", sig.astype(np.float32), cfg.sr,
                 subtype="PCM_16")

    page_trials = []
    key_trials = []
    for k, t in enumerate(triplets):
        files = {name: f"p{t[name]:04d}.wav" for name in ("a", "b", "c")}
        page_trials.append({
            "id": k,
            "ref": files["a"],
            "opt1": files["b"] if t["side_of_b"] == 1 else files["c"],
            "opt2": files["b"] if t["side_of_b"] == 2 else files["c"],
        })
        key_trials.append({
            "id": k, "answer_side": t["answer_side"],
            "d_ab": t["d_ab"], "d_ac": t["d_ac"],
            "bin_ab": t["bin_ab"], "bin_ac": t["bin_ac"],
            "contrast": abs(t["bin_ab"] - t["bin_ac"]),
            "u_a": corpus.u[pool_idx[t["a"]]].tolist(),
            "u_b": corpus.u[pool_idx[t["b"]]].tolist(),
            "u_c": corpus.u[pool_idx[t["c"]]].tolist(),
        })

    (out_dir / "key.json").write_text(json.dumps({
        "n_trials": len(key_trials), "edges": [float(e) for e in edges],
        "e1_dur": e1_dur, "seed": seed, "trials": key_trials,
    }, indent=2), encoding="utf-8")

    html = _HTML_TEMPLATE.replace("__TRIALS_JSON__", json.dumps(page_trials))
    (out_dir / "listening_test.html").write_text(html, encoding="utf-8")
    print(f"[e1] {len(page_trials)} triplets, {len(sel)} wav files -> {out_dir}")
    return {"n_trials": len(page_trials), "edges": [float(e) for e in edges]}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze(response_files: Sequence[Path], material_dir: Path) -> dict[str, object]:
    """Join participant responses with the key; report agreement per contrast."""
    key = json.loads((material_dir / "key.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in key["trials"]}
    per_participant = []
    for rf in response_files:
        data = json.loads(Path(rf).read_text(encoding="utf-8"))
        correct = [r for r in data["responses"]
                   if r.get("choice") == by_id[r["trial"]]["answer_side"]]
        per_participant.append({
            "participant": data.get("participant_id", Path(rf).stem),
            "n": len(data["responses"]),
            "agreement": len(correct) / max(1, len(data["responses"])),
        })

    levels: dict[int, list[bool]] = {}
    all_hits: list[bool] = []
    for rf in response_files:
        data = json.loads(Path(rf).read_text(encoding="utf-8"))
        for r in data["responses"]:
            t = by_id[r["trial"]]
            hit = r.get("choice") == t["answer_side"]
            all_hits.append(hit)
            levels.setdefault(int(t["contrast"]), []).append(hit)

    def rate(hits: list[bool]) -> dict:
        k, n = sum(hits), len(hits)
        lo, hi = _wilson(k, n)
        return {"n": n, "agreement": k / n if n else float("nan"),
                "ci95": [lo, hi]}

    return {
        "per_participant": per_participant,
        "overall": rate(all_hits),
        "by_contrast": {str(c): rate(h) for c, h in sorted(levels.items())},
        "n_participants": len(response_files),
    }


# ---------------------------------------------------------------------------
# self-contained test page (vanilla JS, no server required)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>音色相似度听音实验 / Timbre similarity test</title>
<style>
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.35rem; }
  .card { border: 1px solid #ddd; border-radius: 10px; padding: 1.2rem; margin: 1rem 0; }
  button { font-size: 1rem; padding: .55rem 1rem; margin: .25rem; border-radius: 8px;
           border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }
  button:disabled { opacity: .45; cursor: default; }
  button.played { background: #e6f4ea; border-color: #7bbf8a; }
  .opts { display: flex; gap: 1rem; margin-top: .75rem; }
  .opts > div { flex: 1; border: 1px dashed #ccc; border-radius: 10px; padding: .75rem; }
  .bar { height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: #4a90d9; width: 0; }
  .muted { color: #666; font-size: .9rem; }
  #done { display: none; }
</style>
</head>
<body>
<h1>音色相似度听音实验 / Timbre similarity test</h1>
<div id="intro" class="card">
  <p>你会听到 <b>A</b>、<b>1</b>、<b>2</b> 三个音色（都是同一个音高上的持续音）。
     请判断：<b>1 和 2 当中，哪个听起来更像 A？</b></p>
  <p class="muted">You will hear a reference A and two options. Choose the option
     that sounds more similar to A. Use headphones if possible.
     Press keys 1 / 2 to answer.</p>
  <p class="muted">建议戴耳机，音量适中，全程约 10 分钟。
     所有声音都是合成器音色，没有对错之分，凭直觉即可。</p>
  <label>被试编号 / Participant ID:
    <input id="pid" placeholder="e.g. P01" style="padding:.4rem;font-size:1rem">
  </label>
  <p><button id="start">开始 / Start</button></p>
</div>

<div id="stage" class="card" style="display:none">
  <div class="bar"><i id="prog"></i></div>
  <p class="muted" id="counter"></p>
  <p><button id="playA">▶ A</button>
     <button id="play1" disabled>▶ 1</button>
     <button id="play2" disabled>▶ 2</button>
     <button id="replay" disabled>↻ 全部重放</button></p>
  <div class="opts">
    <div><p><b>1</b> 更像 A</p><button class="ans" data-choice="1" disabled>选择 1</button></div>
    <div><p><b>2</b> 更像 A</p><button class="ans" data-choice="2" disabled>选择 2</button></div>
  </div>
  <p class="muted">按键盘 1 / 2 也可以作答 · 播放完三个声音后即可作答</p>
</div>

<div id="done" class="card">
  <h2>完成，谢谢！/ Done, thank you!</h2>
  <p>请把下面的结果文件交给主试 / hand the downloaded file to the experimenter:</p>
  <p><button id="dl">下载结果 / Download responses</button></p>
  <pre id="summary" class="muted"></pre>
</div>

<script>
const TRIALS = __TRIALS_JSON__;
let i = 0, responses = [], played = {}, t0 = 0;
const $ = id => document.getElementById(id);
const audio = { A: new Audio(), o1: new Audio(), o2: new Audio() };

function loadTrial() {
  const t = TRIALS[i];
  audio.A.src = t.ref; audio.o1.src = t.opt1; audio.o2.src = t.opt2;
  played = {A: false, o1: false, o2: false};
  $("play1").disabled = false; $("play2").disabled = false; $("replay").disabled = false;
  document.querySelectorAll("button.ans").forEach(b => b.disabled = true);
  ["playA","play1","play2"].forEach(b => $(b).classList.remove("played"));
  $("counter").textContent = `第 ${i+1} / ${TRIALS.length} 题`;
  $("prog").style.width = (100 * i / TRIALS.length) + "%";
  t0 = performance.now();
}
function maybeEnable() {
  if (played.A && played.o1 && played.o2)
    document.querySelectorAll("button.ans").forEach(b => b.disabled = false);
}
function play(key, btn) {
  const a = key === "A" ? audio.A : (key === "o1" ? audio.o1 : audio.o2);
  a.currentTime = 0; a.play();
  played[key] = true; $(btn).classList.add("played"); maybeEnable();
}
function answer(choice) {
  responses.push({trial: TRIALS[i].id, choice: parseInt(choice),
                  rt_ms: Math.round(performance.now() - t0)});
  i++;
  if (i >= TRIALS.length) { $("stage").style.display = "none";
    $("done").style.display = "block";
    $("summary").textContent = `共 ${responses.length} 题`; }
  else loadTrial();
}
$("start").onclick = () => {
  if (!$("pid").value.trim()) { alert("请填写被试编号 / please enter participant ID"); return; }
  $("intro").style.display = "none"; $("stage").style.display = "block"; loadTrial();
};
$("playA").onclick = () => play("A", "playA");
$("play1").onclick = () => play("o1", "play1");
$("play2").onclick = () => play("o2", "play2");
$("replay").onclick = () => { play("A","playA"); setTimeout(()=>play("o1","play1"), 900);
  setTimeout(()=>play("o2","play2"), 1800); };
document.querySelectorAll("button.ans").forEach(b => b.onclick = () => answer(b.dataset.choice));
document.addEventListener("keydown", e => {
  if (!document.querySelectorAll("button.ans")[0].disabled) {
    if (e.key === "1") answer(1); if (e.key === "2") answer(2);
  }});
$("dl").onclick = () => {
  const blob = new Blob([JSON.stringify({
    participant_id: $("pid").value.trim(), finished_at: new Date().toISOString(),
    n_trials: TRIALS.length, responses}, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `responses_${$("pid").value.trim() || "anon"}.json`;
  a.click();
};
</script>
</body>
</html>
"""
