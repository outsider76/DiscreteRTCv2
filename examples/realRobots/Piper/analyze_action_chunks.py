#!/usr/bin/env python3
"""Analyze Piper action chunks in a live-client text log.

The report is intentionally self-contained: it embeds the parsed data and uses
plain Canvas/JavaScript, so it can be opened offline without Plotly or a CDN.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "ouput_action_sync.txt"

# Piper MDH parameters from pyAgxArm/utiles/mdh_kinematics.py.  Each row is
# (d, a, alpha, theta_offset), and T = Rx(alpha) Tx(a) Rz(theta) Tz(d).
PIPER_MDH = np.asarray(
    [
        (0.123, 0.0, 0.0, 0.0),
        (0.0, 0.0, -math.pi / 2.0, -3.0058060377846343),
        (0.0, 0.28503, 0.0, -1.793849405199772),
        (0.25075, -0.02198, math.pi / 2.0, 0.0),
        (0.0, 0.0, -math.pi / 2.0, 0.0),
        (0.091, 0.0, math.pi / 2.0, 0.0),
    ],
    dtype=float,
)

EXECUTE_RE = re.compile(
    r"\[EXECUTE\]\s+#(?P<index>\d+)\s+"
    r"chunk=(?P<chunk>\d+)\s+step=(?P<step>\d+)/(?P<chunk_len>\d+)\s+"
    r"state=\[(?P<state>[^\]]+)\]\s+"
    r"predicted=\[(?P<predicted>[^\]]+)\]\s+"
    r"command=\[(?P<command>[^\]]+)\]",
    re.MULTILINE,
)
INFER_RE = re.compile(
    r"\[INFER\]\s+chunk=(?P<chunk>\d+)\s+latency=(?P<latency>[0-9.]+)s;\s+"
    r"executing\s+(?P<steps>\d+)\s+steps\s+at\s+(?P<rate>[0-9.]+)\s+Hz"
)
SOURCES = ("state", "predicted", "command")


def parse_vector(raw: str, label: str) -> np.ndarray:
    values = np.fromstring(raw.replace("\n", " "), sep=" ")
    if values.shape != (7,):
        raise ValueError(f"{label}: expected 7 values, got {values.shape}: {raw!r}")
    return values


def parse_log(path: Path) -> tuple[list[dict[str, Any]], dict[int, float], float, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    for match in EXECUTE_RE.finditer(text):
        item: dict[str, Any] = {
            "index": int(match["index"]),
            "chunk": int(match["chunk"]),
            "step": int(match["step"]),
            "chunk_len": int(match["chunk_len"]),
        }
        for source in SOURCES:
            item[source] = parse_vector(match[source], f"record {item['index']} {source}")
        records.append(item)
    if not records:
        raise ValueError(f"No [EXECUTE] records found in {path}")

    indices = [r["index"] for r in records]
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError("[EXECUTE] record numbers are not continuous")

    inference: dict[int, float] = {}
    rates: list[float] = []
    infer_steps: list[int] = []
    for match in INFER_RE.finditer(text):
        inference[int(match["chunk"])] = float(match["latency"])
        rates.append(float(match["rate"]))
        infer_steps.append(int(match["steps"]))
    rate_hz = float(np.median(rates)) if rates else 20.0
    chunk_len = int(round(float(np.median(infer_steps)))) if infer_steps else records[0]["chunk_len"]
    return records, inference, rate_hz, chunk_len


def mdh_matrix(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.asarray(
        [
            [ct, -st, 0.0, a],
            [ca * st, ca * ct, -sa, -sa * d],
            [sa * st, sa * ct, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.eye(4)
    for joint, (d, a, alpha, offset) in zip(joints[:6], PIPER_MDH):
        transform = transform @ mdh_matrix(d, a, alpha, float(joint + offset))
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    # Robust eigenvector formulation; output order is [x, y, z, w].
    r = rotation
    k = np.asarray(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[1, 2] - r[2, 1]],
            [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[2, 0] - r[0, 2]],
            [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[0, 1] - r[1, 0]],
            [r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0], r[0, 0] + r[1, 1] + r[2, 2]],
        ],
        dtype=float,
    ) / 3.0
    values, vectors = np.linalg.eigh(k)
    quat = vectors[:, int(np.argmax(values))]
    if quat[3] < 0:
        quat = -quat
    return quat


def orientation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def build_analysis(
    records: list[dict[str, Any]], inference: dict[int, float], rate_hz: float, chunk_len: int
) -> dict[str, Any]:
    compact_records: list[dict[str, Any]] = []
    poses: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {source: [] for source in SOURCES}
    for record in records:
        compact: dict[str, Any] = {
            "i": record["index"], "c": record["chunk"], "s": record["step"], "n": record["chunk_len"]
        }
        for source in SOURCES:
            pos, rot = forward_kinematics(record[source])
            poses[source].append((pos, rot))
            compact[source] = {
                "j": np.round(record[source], 6).tolist(),
                "p": np.round(pos, 7).tolist(),
                "q": np.round(matrix_to_quaternion(rot), 7).tolist(),
            }
        compact_records.append(compact)

    transitions: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
    stats: dict[str, Any] = {}
    per_step: dict[str, Any] = {}
    for source in SOURCES:
        for k in range(1, len(records)):
            previous, current = records[k - 1], records[k]
            prev_pos, prev_rot = poses[source][k - 1]
            pos, rot = poses[source][k]
            joint_diff = current[source][:6] - previous[source][:6]
            boundary = current["chunk"] != previous["chunk"]
            nominal_dt = 1.0 / rate_hz
            infer_s = inference.get(current["chunk"], 0.0) if boundary else 0.0
            transitions[source].append(
                {
                    "i": current["index"],
                    "c": current["chunk"],
                    "s": current["step"],
                    "b": boundary,
                    "dp": float(np.linalg.norm(pos - prev_pos) * 1000.0),
                    "dr": orientation_delta_deg(prev_rot, rot),
                    "dj": float(np.max(np.abs(joint_diff))),
                    "djl2": float(np.linalg.norm(joint_diff)),
                    "dt": nominal_dt + infer_s,
                    "gap": infer_s,
                }
            )

        within = [t for t in transitions[source] if not t["b"]]
        boundary = [t for t in transitions[source] if t["b"]]
        stats[source] = {}
        for metric in ("dp", "dr", "dj", "djl2"):
            within_summary = summary([float(t[metric]) for t in within])
            boundary_summary = summary([float(t[metric]) for t in boundary])
            stats[source][metric] = {
                "within": within_summary,
                "boundary": boundary_summary,
                "median_ratio": safe_ratio(float(boundary_summary["median"]), float(within_summary["median"])),
                "p95_ratio": safe_ratio(float(boundary_summary["p95"]), float(within_summary["p95"])),
            }
        per_step[source] = {}
        for step in range(1, chunk_len + 1):
            selected = [t for t in transitions[source] if t["s"] == step]
            per_step[source][str(step)] = {
                "dp": summary([float(t["dp"]) for t in selected]),
                "dr": summary([float(t["dr"]) for t in selected]),
                "dj": summary([float(t["dj"]) for t in selected]),
            }

    inference_values = list(inference.values())
    infer_summary = summary([value * 1000.0 for value in inference_values])
    command_dp = stats["command"]["dp"]
    predicted_dp = stats["predicted"]["dp"]
    state_dp = stats["state"]["dp"]
    spatial_ratio = float(command_dp["median_ratio"])
    raw_ratio = float(predicted_dp["median_ratio"])
    observed_ratio = float(state_dp["median_ratio"])

    if spatial_ratio >= 1.5:
        verdict = "chunk 边界的空间不连续更明显"
        explanation = (
            f"实际发送 command 在边界处的 EE 位移中位数是 chunk 内的 {spatial_ratio:.2f} 倍；"
            "因此除了等待停顿，重规划后的第 1 步也带来了更大的空间跳变。"
        )
    elif spatial_ratio <= 1.15:
        verdict = "主要不是 chunk 边界的空间跳变"
        explanation = (
            f"实际发送 command 的边界/内部 EE 位移中位数比仅为 {spatial_ratio:.2f}；"
            "chunk 边界主要增加了推理等待，明显抖动更可能来自 chunk 内动作形状、跟踪误差或底层控制。"
        )
    else:
        verdict = "chunk 内变化与边界效应混合存在"
        explanation = (
            f"实际发送 command 的边界/内部 EE 位移中位数比为 {spatial_ratio:.2f}，"
            "边界有放大但不足以单独解释全部抖动。"
        )
    gap_median = float(infer_summary["median"])
    timing_note = (
        f"每个 chunk 后还存在约 {gap_median:.0f} ms 的推理等待（中位数），"
        f"所以边界命令间隔估算约为 {1000.0 / rate_hz + gap_median:.0f} ms，"
        f"而 chunk 内标称间隔为 {1000.0 / rate_hz:.0f} ms。这个时间停顿会造成速度不连续。"
    )

    boundary_commands = sorted(
        [t for t in transitions["command"] if t["b"]], key=lambda item: item["dp"], reverse=True
    )[:12]
    return {
        "meta": {
            "records": len(records),
            "chunks": len({record["chunk"] for record in records}),
            "rate_hz": rate_hz,
            "chunk_len": chunk_len,
            "inference_ms": infer_summary,
            "first_index": records[0]["index"],
            "last_index": records[-1]["index"],
        },
        "records": compact_records,
        "transitions": transitions,
        "stats": stats,
        "per_step": per_step,
        "top_boundaries": boundary_commands,
        "conclusion": {
            "verdict": verdict,
            "explanation": explanation,
            "timing": timing_note,
            "ratios": {"predicted": raw_ratio, "command": spatial_ratio, "state": observed_ratio},
        },
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Piper action chunk / EE pose 分析</title>
<style>
:root{--bg:#0b1020;--panel:#121a2d;--panel2:#18223a;--text:#e8edf7;--muted:#9eabc2;--grid:#2a3652;--blue:#55a7ff;--orange:#ffb454;--green:#5dd39e;--red:#ff637d}
*{box-sizing:border-box} body{margin:0;background:linear-gradient(145deg,#080d19,#10182b);color:var(--text);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1500px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;flex-wrap:wrap}h1{font-size:26px;margin:0}h2{font-size:18px;margin:0 0 12px}.sub,.note{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}.card,.panel{background:rgba(18,26,45,.92);border:1px solid #263451;border-radius:12px;padding:16px;box-shadow:0 8px 30px #0003}.card b{display:block;font-size:24px}.card span{color:var(--muted)}
.verdict{border-left:4px solid var(--orange);padding:13px 16px;background:#192238;border-radius:7px;margin:15px 0}.verdict strong{color:#ffd18d;font-size:17px}.grid2{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(350px,.7fr);gap:14px}.stack{display:grid;gap:14px}.canvas-wrap{position:relative;height:600px;background:#090f1d;border-radius:8px;overflow:hidden}.chart-wrap{height:280px;position:relative;background:#0b1222;border-radius:8px}.chart-wrap.small{height:250px}canvas{width:100%;height:100%;display:block}
.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.toolbar label{cursor:pointer}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.tip{display:none;position:absolute;pointer-events:none;background:#050914eb;border:1px solid #425271;border-radius:7px;padding:7px 9px;white-space:pre;font:12px/1.45 ui-monospace,monospace;z-index:5}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #2a3652}th:first-child,td:first-child{text-align:left}th{color:#b9c7dd}.scroll{overflow:auto}.pill{display:inline-block;padding:2px 7px;border-radius:12px;background:#24314e;color:#cbd7e9}.method{font-size:13px;color:var(--muted)}code{color:#a9d1ff}@media(max-width:950px){.grid2{grid-template-columns:1fr}.canvas-wrap{height:480px}}
</style></head><body><main>
<div class="top"><div><h1>Piper action chunk → EE pose 分析</h1><div class="sub" id="sourcePath"></div></div><div class="pill" id="generatedAt"></div></div>
<div class="cards" id="cards"></div>
<div class="verdict"><strong id="verdict"></strong><div id="explanation"></div><div id="timing"></div></div>
<div class="grid2"><section class="panel"><h2>EE 轨迹（拖动旋转，滚轮缩放）</h2>
 <div class="toolbar"><label><input type="checkbox" data-series="state" checked> <i class="dot" style="background:var(--green)"></i>state</label><label><input type="checkbox" data-series="predicted" checked> <i class="dot" style="background:var(--orange)"></i>predicted</label><label><input type="checkbox" data-series="command" checked> <i class="dot" style="background:var(--blue)"></i>command</label><span class="note">红点 = 新 chunk 的 step 1</span><button id="reset3d">重置视角</button></div>
 <div class="canvas-wrap"><canvas id="scene"></canvas><div class="tip" id="sceneTip"></div></div></section>
 <div class="stack"><section class="panel"><h2>如何读结论</h2><p><b>predicted</b> 是模型原始输出；<b>command</b> 是经过每关节步长限制后实际发布的目标；<b>state</b> 是反馈。边界比较的是上一 chunk 的 step 8 → 下一 chunk 的 step 1，其余为 chunk 内转换。</p><p class="note">空间 gap 与时间 gap 是两件事：即使相邻 command 很接近，推理期间不发新命令也会出现停顿/速度突变。</p><div id="ratios"></div></section>
 <section class="panel"><h2>边界 EE 位移最大的 command</h2><div class="scroll"><table><thead><tr><th>到达</th><th>位移 mm</th><th>旋转 °</th><th>max |Δq|</th><th>推理 ms</th></tr></thead><tbody id="topRows"></tbody></table></div></section></div></div>
<div class="grid2" style="margin-top:14px"><section class="panel"><h2>相邻 EE 位移</h2><div class="toolbar">显示：<select id="metricSource"><option value="command">command</option><option value="predicted">predicted</option><option value="state">state</option></select><span class="note">红点/竖线为 chunk 边界</span></div><div class="chart-wrap"><canvas id="metricChart"></canvas><div class="tip" id="metricTip"></div></div></section>
<section class="panel"><h2>按目标 step 分组的平均 EE 位移</h2><div class="note">step 1 = chunk 边界；step 2–8 = chunk 内部</div><div class="chart-wrap"><canvas id="stepChart"></canvas></div></section></div>
<section class="panel" style="margin-top:14px"><h2>command 每个关节的相邻变化 |Δq|</h2><div class="legend" id="jointLegend"></div><div class="chart-wrap small"><canvas id="jointChart"></canvas><div class="tip" id="jointTip"></div></div></section>
<section class="panel" style="margin-top:14px"><h2>统计对比</h2><div class="scroll"><table><thead><tr><th>序列 / 指标</th><th>chunk 内 median</th><th>边界 median</th><th>边界/内部</th><th>chunk 内 p95</th><th>边界 p95</th><th>最大</th></tr></thead><tbody id="statsRows"></tbody></table></div></section>
<section class="panel method" style="margin-top:14px"><h2>方法和限制</h2><ul><li>用 Piper 的 6 轴 MDH 参数做前向运动学，末端定义为 link6 / gripper base；第 7 个值是夹爪开度，不参与 EE 位姿 FK。</li><li>旋转变化为两个旋转矩阵之间的 SO(3) 测地角。位置单位为 mm，关节单位为 rad。</li><li>日志没有每条发布命令的 wall-clock 时间戳。chunk 内间隔按日志中的控制频率估算；边界额外间隔按对应 <code>[INFER] latency</code> 估算。因此本报告能可靠判断“目标轨迹是否跳变”，但对真机速度/加速度抖动只能做时序推断。</li><li>若要确认物理抖动来源，建议后续同时记录 command/state 的 ROS 时间戳，并计算 EE 速度和加速度。</li></ul></section>
</main><script>
const D=__DATA__; const COLORS={state:'#5dd39e',predicted:'#ffb454',command:'#55a7ff'}, JCOL=['#55a7ff','#ffb454','#5dd39e','#dc78ff','#ff637d','#63d5ff'];
const fmt=(x,n=2)=>Number(x).toFixed(n), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.querySelector('#sourcePath').textContent=D.input; document.querySelector('#generatedAt').textContent=D.generated;
document.querySelector('#verdict').textContent=D.conclusion.verdict; document.querySelector('#explanation').textContent=D.conclusion.explanation; document.querySelector('#timing').textContent=D.conclusion.timing;
const m=D.meta; document.querySelector('#cards').innerHTML=[['记录',m.records],['Action chunks',m.chunks],['控制频率',fmt(m.rate_hz,1)+' Hz'],['Chunk 长度',m.chunk_len+' steps'],['推理延迟 median',fmt(m.inference_ms.median,0)+' ms'],['推理延迟 p95',fmt(m.inference_ms.p95,0)+' ms']].map(x=>`<div class="card"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('');
document.querySelector('#ratios').innerHTML='<table><tr><th>序列</th><th>边界/内部位移 median</th></tr>'+['predicted','command','state'].map(s=>`<tr><td>${s}</td><td>${fmt(D.conclusion.ratios[s],2)}×</td></tr>`).join('')+'</table>';
document.querySelector('#topRows').innerHTML=D.top_boundaries.map(t=>`<tr><td>chunk ${t.c} / step ${t.s}</td><td>${fmt(t.dp)}</td><td>${fmt(t.dr)}</td><td>${fmt(t.dj,4)}</td><td>${fmt(t.gap*1000,0)}</td></tr>`).join('');
const units={dp:'mm',dr:'°',dj:'rad'}, names={dp:'EE 位移',dr:'EE 旋转',dj:'max |Δq|'}; let rows='';
for(const s of ['predicted','command','state'])for(const k of ['dp','dr','dj']){const x=D.stats[s][k]; rows+=`<tr><td>${s} / ${names[k]}</td><td>${fmt(x.within.median,3)} ${units[k]}</td><td>${fmt(x.boundary.median,3)} ${units[k]}</td><td>${fmt(x.median_ratio,2)}×</td><td>${fmt(x.within.p95,3)}</td><td>${fmt(x.boundary.p95,3)}</td><td>${fmt(Math.max(x.within.max,x.boundary.max),3)}</td></tr>`} document.querySelector('#statsRows').innerHTML=rows;

function fitCanvas(canvas){const dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w*dpr||canvas.height!==h*dpr){canvas.width=w*dpr;canvas.height=h*dpr}const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return[c,w,h]}
const scene=document.querySelector('#scene'), stip=document.querySelector('#sceneTip'); let yaw=-.75,pitch=.55,zoom=1,drag=null,projected=[];
const allP=D.records.flatMap(r=>['state','predicted','command'].map(s=>r[s].p)); const mins=[0,1,2].map(k=>Math.min(...allP.map(p=>p[k]))),maxs=[0,1,2].map(k=>Math.max(...allP.map(p=>p[k]))),center=mins.map((v,k)=>(v+maxs[k])/2),extent=Math.max(...mins.map((v,k)=>maxs[k]-v),.08);
function proj(p,w,h){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];let X=Math.cos(yaw)*x-Math.sin(yaw)*y,Y=Math.sin(yaw)*x+Math.cos(yaw)*y;let Z=Math.cos(pitch)*z-Math.sin(pitch)*Y,depth=Math.sin(pitch)*z+Math.cos(pitch)*Y;let sc=Math.min(w,h)*.72/extent*zoom;return[w/2+X*sc,h/2-Z*sc,depth]}
function sceneDraw(){const[c,w,h]=fitCanvas(scene);c.clearRect(0,0,w,h);c.fillStyle='#090f1d';c.fillRect(0,0,w,h);projected=[];
 const z=mins[2];c.strokeStyle='#23304a';c.lineWidth=1;for(let n=0;n<=10;n++){let f=n/10,a=[mins[0]+(maxs[0]-mins[0])*f,mins[1],z],b=[mins[0]+(maxs[0]-mins[0])*f,maxs[1],z],u=proj(a,w,h),v=proj(b,w,h);c.beginPath();c.moveTo(u[0],u[1]);c.lineTo(v[0],v[1]);c.stroke();a=[mins[0],mins[1]+(maxs[1]-mins[1])*f,z];b=[maxs[0],mins[1]+(maxs[1]-mins[1])*f,z];u=proj(a,w,h);v=proj(b,w,h);c.beginPath();c.moveTo(u[0],u[1]);c.lineTo(v[0],v[1]);c.stroke()}
 for(const s of ['state','predicted','command']){if(!document.querySelector(`[data-series=${s}]`).checked)continue;const pts=D.records.map((r,k)=>{const p=proj(r[s].p,w,h);projected.push({x:p[0],y:p[1],r,k,s});return p});c.strokeStyle=COLORS[s];c.globalAlpha=.82;c.lineWidth=s==='command'?2.2:1.5;c.beginPath();pts.forEach((p,k)=>k?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]));c.stroke();c.globalAlpha=1;D.records.forEach((r,k)=>{if(r.s===1&&k){const p=pts[k];c.fillStyle='#ff637d';c.beginPath();c.arc(p[0],p[1],3.2,0,Math.PI*2);c.fill()}})}
 c.font='12px system-ui';[['X',[maxs[0],center[1],center[2]],'#ff637d'],['Y',[center[0],maxs[1],center[2]],'#5dd39e'],['Z',[center[0],center[1],maxs[2]],'#55a7ff']].forEach(a=>{const p=proj(a[1],w,h);c.fillStyle=a[2];c.fillText(a[0],p[0]+4,p[1]-4)})}
scene.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,yaw,pitch];scene.setPointerCapture(e.pointerId)});scene.addEventListener('pointermove',e=>{if(drag){yaw=drag[2]+(e.clientX-drag[0])*.008;pitch=Math.max(-1.45,Math.min(1.45,drag[3]+(e.clientY-drag[1])*.008));sceneDraw();return}let best=null,bd=100;for(const p of projected){let d=Math.hypot(e.offsetX-p.x,e.offsetY-p.y);if(d<bd){bd=d;best=p}}if(best&&bd<10){const r=best.r,x=r[best.s];stip.style.display='block';stip.style.left=(e.offsetX+12)+'px';stip.style.top=(e.offsetY+12)+'px';stip.textContent=`${best.s}  #${r.i} chunk=${r.c} step=${r.s}/${r.n}\nxyz [m] = ${x.p.map(v=>fmt(v,4)).join(', ')}\nq1..q6 = ${x.j.slice(0,6).map(v=>fmt(v,3)).join(', ')}`}else stip.style.display='none'});scene.addEventListener('pointerup',()=>drag=null);scene.addEventListener('pointerleave',()=>{drag=null;stip.style.display='none'});scene.addEventListener('wheel',e=>{e.preventDefault();zoom*=e.deltaY>0?.9:1.1;zoom=Math.max(.35,Math.min(5,zoom));sceneDraw()},{passive:false});document.querySelector('#reset3d').onclick=()=>{yaw=-.75;pitch=.55;zoom=1;sceneDraw()};document.querySelectorAll('[data-series]').forEach(x=>x.onchange=sceneDraw);

function lineChart(canvas,tip,source){const[c,w,h]=fitCanvas(canvas),pad={l:55,r:15,t:15,b:35},T=D.transitions[source],vals=T.map(x=>x.dp),ymax=Math.max(...vals)*1.08||1;c.clearRect(0,0,w,h);c.fillStyle='#0b1222';c.fillRect(0,0,w,h);c.strokeStyle='#2a3652';c.fillStyle='#9eabc2';c.font='11px system-ui';for(let n=0;n<=4;n++){let y=pad.t+(h-pad.t-pad.b)*n/4,v=ymax*(1-n/4);c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText(fmt(v,1),5,y+4)}const xy=(v,k)=>[pad.l+(w-pad.l-pad.r)*k/Math.max(1,T.length-1),pad.t+(h-pad.t-pad.b)*(1-v/ymax)];c.strokeStyle=COLORS[source];c.lineWidth=1.5;c.beginPath();T.forEach((t,k)=>{const p=xy(t.dp,k);k?c.lineTo(...p):c.moveTo(...p)});c.stroke();T.forEach((t,k)=>{if(t.b){const p=xy(t.dp,k);c.strokeStyle='#ff637d44';c.beginPath();c.moveTo(p[0],pad.t);c.lineTo(p[0],h-pad.b);c.stroke();c.fillStyle='#ff637d';c.beginPath();c.arc(...p,3,0,Math.PI*2);c.fill()}});c.fillStyle='#9eabc2';c.fillText('EE Δposition [mm]',pad.l,11);canvas._chart={T,xy,pad,ymax,tip}}
function stepChart(){const canvas=document.querySelector('#stepChart'),[c,w,h]=fitCanvas(canvas),src=document.querySelector('#metricSource').value,pad={l:48,r:12,t:18,b:35},items=Object.entries(D.per_step[src]).map(([s,x])=>[+s,x.dp.mean]),max=Math.max(...items.map(x=>x[1]))*1.18||1;c.clearRect(0,0,w,h);c.fillStyle='#0b1222';c.fillRect(0,0,w,h);let bw=(w-pad.l-pad.r)/items.length*.68;items.forEach(([s,v],k)=>{let x=pad.l+(w-pad.l-pad.r)*(k+.5)/items.length,y=pad.t+(h-pad.t-pad.b)*(1-v/max);c.fillStyle=s===1?'#ff637d':COLORS[src];c.fillRect(x-bw/2,y,bw,h-pad.b-y);c.fillStyle='#c5d0e2';c.textAlign='center';c.fillText('step '+s,x,h-13);c.fillText(fmt(v,1),x,y-5)});c.textAlign='left';c.fillStyle='#9eabc2';c.fillText('mean EE Δposition [mm]',8,12)}
function jointChart(){const canvas=document.querySelector('#jointChart'),[c,w,h]=fitCanvas(canvas),pad={l:50,r:12,t:15,b:30},R=D.records;let A=[];for(let k=1;k<R.length;k++)A.push(R[k].command.j.slice(0,6).map((v,j)=>Math.abs(v-R[k-1].command.j[j])));let max=Math.max(...A.flat())*1.08||.01;c.clearRect(0,0,w,h);c.fillStyle='#0b1222';c.fillRect(0,0,w,h);const xy=(v,k)=>[pad.l+(w-pad.l-pad.r)*k/Math.max(1,A.length-1),pad.t+(h-pad.t-pad.b)*(1-v/max)];for(let j=0;j<6;j++){c.strokeStyle=JCOL[j];c.lineWidth=1.2;c.beginPath();A.forEach((a,k)=>{const p=xy(a[j],k);k?c.lineTo(...p):c.moveTo(...p)});c.stroke()}D.transitions.command.forEach((t,k)=>{if(t.b){let x=xy(0,k)[0];c.strokeStyle='#ff637d55';c.beginPath();c.moveTo(x,pad.t);c.lineTo(x,h-pad.b);c.stroke()}});c.fillStyle='#9eabc2';c.fillText('|Δq| [rad]',5,12);canvas._joint={A,xy,pad,max}}
document.querySelector('#jointLegend').innerHTML=JCOL.map((c,k)=>`<span><i class="dot" style="background:${c}"></i>joint ${k+1}</span>`).join('');
const metric=document.querySelector('#metricChart'),mtip=document.querySelector('#metricTip');function redraw(){let s=document.querySelector('#metricSource').value;lineChart(metric,mtip,s);stepChart()}document.querySelector('#metricSource').onchange=redraw;
metric.addEventListener('pointermove',e=>{const o=metric._chart;if(!o)return;let k=Math.round((e.offsetX-o.pad.l)/(metric.clientWidth-o.pad.l-o.pad.r)*(o.T.length-1));if(k>=0&&k<o.T.length){let t=o.T[k];mtip.style.display='block';mtip.style.left=(e.offsetX+10)+'px';mtip.style.top=(e.offsetY+10)+'px';mtip.textContent=`#${t.i} chunk=${t.c} step=${t.s}${t.b?'  [边界]':''}\nEE Δp=${fmt(t.dp,3)} mm, ΔR=${fmt(t.dr,3)}°\nmax |Δq|=${fmt(t.dj,4)} rad\n估算 Δt=${fmt(t.dt*1000,0)} ms`}});metric.addEventListener('pointerleave',()=>mtip.style.display='none');
window.addEventListener('resize',()=>{sceneDraw();redraw();jointChart()});sceneDraw();redraw();jointChart();
</script></body></html>'''


def write_report(analysis: dict[str, Any], input_path: Path, output_path: Path) -> None:
    from datetime import datetime

    payload = dict(analysis)
    payload["input"] = str(input_path)
    payload["generated"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", r"<\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(HTML_TEMPLATE.replace("__DATA__", serialized), encoding="utf-8")


def print_summary(analysis: dict[str, Any], output_path: Path) -> None:
    meta, conclusion = analysis["meta"], analysis["conclusion"]
    print(f"Parsed {meta['records']} records in {meta['chunks']} chunks ({meta['rate_hz']:.1f} Hz).")
    print(f"Inference latency: median={meta['inference_ms']['median']:.1f} ms, p95={meta['inference_ms']['p95']:.1f} ms")
    for source in SOURCES:
        metric = analysis["stats"][source]["dp"]
        print(
            f"{source:9s} EE delta median: within={metric['within']['median']:.3f} mm, "
            f"boundary={metric['boundary']['median']:.3f} mm, ratio={metric['median_ratio']:.2f}x"
        )
    print(f"Verdict: {conclusion['verdict']}")
    print(f"Wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Piper live-client log")
    parser.add_argument("--output", type=Path, help="Output HTML (default: <input>_ee_analysis.html)")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_ee_analysis.html")
    )
    records, inference, rate_hz, chunk_len = parse_log(input_path)
    analysis = build_analysis(records, inference, rate_hz, chunk_len)
    write_report(analysis, input_path, output_path)
    print_summary(analysis, output_path)


if __name__ == "__main__":
    main()
