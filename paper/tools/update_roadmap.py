"""Add the metric-weighting finding to the roadmap (paper §7 and ROADMAP.md)."""

from __future__ import annotations

from pathlib import Path

BS = chr(92)
SENT = "@" + "@"


def tex(t: str) -> str:
    return t.replace(SENT, BS)


# ---- paper §7: fold the metric result into Phase 2 ------------------------
p = Path("main.tex")
s = p.read_text(encoding="utf-8")
old = tex("""\paragraph{Phase 2 --- Replace the embedding (CLAP, MuQ) and re-run E2/E3
unchanged.} @@emph{Motivation:} the objective's geometry sets the search
difficulty; learned spaces may be smoother and better aligned.
@@emph{Deliverable:} three embeddings (handcrafted / CLAP / MuQ) evaluated by the
same E1 gate and the same E2 protocol.""")
new = tex("""\paragraph{Phase 2 --- Fix the objective: metric weighting first, then the
embedding.} @@emph{Motivation:} the diagnostics (@@S@@ref{sec:res-diagnostics})
attribute the plateau to how the objective is @@emph{weighted} rather than to the
search: equalising the feature blocks improved the oracle from $0.875$ to
$0.809$ at the same render budget, the only intervention in this paper that
improved the gate without touching the search.  @@emph{Deliverable:} (a) a
systematic weighting study over the 13 blocks (and, once listener data exists, a
metric @@emph{learned} from E1 triplets); (b) three embeddings
(handcrafted / CLAP / MuQ), each with and without block balancing, evaluated by
the same E1 gate and the same E2 protocol.""")
assert old in s, "phase 2 anchor"
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("paper §7 updated")

# ---- ROADMAP.md: new first-step subsection + decision note ----------------
p = Path("ROADMAP.md")
s = p.read_text(encoding="utf-8")
old = """### 1.2 换嵌入，重跑 E2/E3（与 1.1 并行，不需要人类）"""
new = """### 1.2 先修目标函数：度量加权（最便宜、已见效果）

诊断实验给出本文唯一一个"不改搜索算法就能变好"的干预：把 13 个特征块的贡献**均衡化**
（每维权重 $1/\\sqrt{d_B}$，再做尺度归一），真梯度的中位比值从 **0.875 → 0.809**，
success@0.5 从 0.06 **翻倍到 0.12**——同一个搜索算法、同一批目标、同样的渲染预算。

```bash
python -m timbreprobe.cli e2 --preset smoke --metric balanced   # -> out/e2_smoke_balanced
```

* **为什么**：85 维里有 64 维是 log-mel，包络失配主导了被优化的那个数；
  而"波形/奇偶谐波"这类真正决定音色身份的区分只占 1 维、权重极小。
  残差分解显示每维超重最多的是谐波比（2.34×）、过零率（1.79×）、斜率（1.75×）。
* **完成标准**：在 13 个块上做系统性加权搜索（或按 E1 数据学习度量），
  让真梯度中位比值稳定 **< 0.75**。
* **失败就怎么办**：若加权扫描无进一步改善，说明瓶颈回到可达性（进第二阶段潜空间）。

### 1.3 换嵌入，重跑 E2/E3（与 1.1/1.2 并行，不需要人类）"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace("### 1.3 修代理（与 1.2 并行；先做便宜的判断实验）",
              "### 1.4 修代理（与 1.2/1.3 并行；先做便宜的判断实验）", 1)
p.write_text(s, encoding="utf-8")
print("ROADMAP.md updated")
