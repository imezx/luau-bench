from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from luau_bench.evaluator import BenchmarkRun


def save_html(run: BenchmarkRun, path: Path) -> Path:
    data = _build_data(run)
    html = _render(data)
    path.write_text(html, encoding="utf-8")
    return path


def _build_data(run: BenchmarkRun) -> dict[str, Any]:
    tasks = []
    for tr in sorted(run.task_results, key=lambda t: t.task_name):
        primary = tr.primary_metric
        primary_value = tr.metrics.get(primary, 0.0) if primary else 0.0

        ci_lo = ci_hi = None
        if primary and primary in tr.std_errors:
            se = tr.std_errors[primary]
            ci_lo = se.get("ci_lower")
            ci_hi = se.get("ci_upper")

        metric_rows = [
            {"name": k, "value": round(v, 2)}
            for k, v in tr.metrics.items()
            if isinstance(v, (int, float))
        ]

        exec_details: Optional[list] = tr.metric_extras.get("_exec_details")
        analyze_details: Optional[list] = tr.metric_extras.get("_analyze_details")
        effective_n: int = tr.metric_extras.get("_effective_n", 1)

        samples = []
        for doc_idx, dr in enumerate(tr.doc_results):
            flat_idx = doc_idx * effective_n

            exec_entry: Optional[dict] = None
            if exec_details and flat_idx < len(exec_details) and exec_details[flat_idx]:
                raw = exec_details[flat_idx]
                exec_entry = {
                    "passed": raw.get("passed", 0),
                    "failed": raw.get("failed", 0),
                    "errors": raw.get("errors", 0),
                    "total": raw.get("total", 0),
                    "runtime_ms": round(raw.get("runtime_ms", 0.0), 1),
                    "timed_out": raw.get("timed_out", False),
                    "stderr": (raw.get("stderr") or "")[:400],
                    "tests": [
                        {
                            "name": t.get("test", ""),
                            "status": t.get("status", ""),
                            "message": t.get("message", ""),
                        }
                        for t in raw.get("details", [])
                    ],
                }

            analyze_entry: Optional[dict] = None
            if analyze_details and doc_idx < len(analyze_details) and analyze_details[doc_idx]:
                diags = analyze_details[doc_idx]
                if isinstance(diags, list):
                    errors = sum(1 for d in diags if d.get("severity") == "error")
                    warnings = sum(1 for d in diags if d.get("severity") == "warning")
                    analyze_entry = {
                        "errors": errors,
                        "warnings": warnings,
                        "diagnostics": [
                            {
                                "line": d.get("line", 0),
                                "col": d.get("column", 0),
                                "severity": d.get("severity", ""),
                                "code": d.get("code", ""),
                                "message": d.get("message", ""),
                            }
                            for d in diags[:20]
                        ],
                    }

            samples.append(
                {
                    "description": (dr.doc.get("description", "") or str(list(dr.doc.keys())[:3]))[
                        :200
                    ],
                    "prediction": dr.prediction[:1000],
                    "reference": dr.reference[:400],
                    "raw": dr.raw_generation[:800] if dr.raw_generation else "",
                    "exec": exec_entry,
                    "analyze": analyze_entry,
                }
            )

        tasks.append(
            {
                "name": tr.task_name,
                "version": tr.version,
                "num_docs": tr.num_docs,
                "weight": getattr(tr, "weight", 1.0),
                "primary": primary,
                "primary_value": round(primary_value, 2),
                "ci_lo": round(ci_lo, 1) if ci_lo is not None else None,
                "ci_hi": round(ci_hi, 1) if ci_hi is not None else None,
                "error": tr.error,
                "metrics": metric_rows,
                "samples": samples,
            }
        )

    composite = run.composite_score
    composite_lo = composite_hi = None
    if run.composite_se:
        composite_lo = round(run.composite_se.get("ci_lower", 0.0), 4)
        composite_hi = round(run.composite_se.get("ci_upper", 0.0), 4)

    return {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "provider": run.provider,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": run.started_at,
        "finished_at": run.finished_at or run.started_at,
        "elapsed_s": round((run.finished_at or run.started_at) - run.started_at, 1),
        "composite": round(composite, 4) if composite is not None else None,
        "composite_lo": composite_lo,
        "composite_hi": composite_hi,
        "tasks": tasks,
    }


def _render(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Luau Bench - {data["model_id"]}</title>
<style>
  :root {{
    --bg:          #faf9f6;
    --surface:     #ffffff;
    --surface-alt: #fdf8f0;
    --border:      #e8d9bc;
    --border-soft: #f0e8d4;
    --text:        #1c1917;
    --text-muted:  #78716c;
    --text-light:  #a8a29e;

    --amber:       #b45309;
    --amber-light: #fef3c7;
    --amber-mid:   #fde68a;
    --amber-line:  #d97706;

    --green:       #166534;
    --green-light: #dcfce7;
    --green-line:  #16a34a;

    --red:         #991b1b;
    --red-light:   #fee2e2;
    --red-line:    #dc2626;

    --code-bg:     #f8f4ec;
    --code-text:   #44403c;
    --code-border: #e2d4b8;

    --radius-sm:   6px;
    --radius-md:   10px;
    --radius-lg:   14px;

    --shadow-sm:   0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md:   0 4px 12px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.04);
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.65;
    padding: 0;
  }}

  .page {{
    max-width: 1080px;
    margin: 0 auto;
    padding: 40px 28px 80px;
  }}

  .header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
    flex-wrap: wrap;
    padding-bottom: 28px;
    border-bottom: 2px solid var(--border);
    margin-bottom: 36px;
  }}
  .header-left h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -.02em;
    margin-bottom: 4px;
  }}
  .header-left h1 span {{
    color: rgb(172, 147, 98);
  }}
  .header-meta {{
    color: var(--text-muted);
    font-size: .8rem;
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
  }}
  .header-meta span {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .header-meta .dot {{
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--border);
  }}

  .composite-pill {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 28px;
    box-shadow: var(--shadow-sm);
    min-width: 130px;
  }}
  .composite-pill .score {{
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -.04em;
    line-height: 1;
  }}
  .composite-pill .score-label {{
    font-size: .7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text-muted);
    margin-top: 5px;
  }}
  .composite-pill .ci-line {{
    font-size: .7rem;
    color: var(--text-light);
    margin-top: 4px;
  }}
  .score-good  {{ color: var(--green); }}
  .score-ok    {{ color: var(--amber-line); }}
  .score-poor  {{ color: var(--red); }}
  .score-none  {{ color: var(--text-muted); }}

  .section-title {{
    font-size: .7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--text-muted);
    margin: 40px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-title::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-soft);
  }}

  .card {{
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    margin-bottom: 24px;
  }}
  .card-inner {{
    padding: 20px 24px;
  }}

  .stats-row {{
    display: flex;
    gap: 0;
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: 24px;
    background: var(--surface);
    box-shadow: var(--shadow-sm);
  }}
  .stat-cell {{
    flex: 1;
    padding: 18px 20px;
    text-align: center;
    border-right: 1px solid var(--border-soft);
    position: relative;
  }}
  .stat-cell:last-child {{ border-right: none; }}
  .stat-cell::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--border-soft);
  }}
  .stat-cell.accent::before {{ background: var(--amber-line); }}
  .stat-num {{
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -.03em;
    color: var(--text);
    line-height: 1;
  }}
  .stat-label {{
    font-size: .68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text-muted);
    margin-top: 5px;
  }}

  .chart-wrap {{
    padding: 20px 24px;
  }}

  .results-table {{
    width: 100%;
    border-collapse: collapse;
  }}
  .results-table thead tr {{
    background: var(--surface-alt);
    border-bottom: 1.5px solid var(--border);
  }}
  .results-table th {{
    padding: 10px 16px;
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-muted);
    text-align: left;
    white-space: nowrap;
  }}
  .results-table td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-soft);
    vertical-align: middle;
  }}
  .results-table tbody tr:last-child td {{ border-bottom: none; }}
  .results-table tbody tr:hover td {{
    background: var(--surface-alt);
  }}
  .task-name {{
    font-weight: 600;
    font-size: .88rem;
  }}
  .task-version {{
    font-size: .75rem;
    color: var(--text-light);
    background: var(--border-soft);
    padding: 2px 7px;
    border-radius: 20px;
  }}

  .score-cell {{
    display: flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
  }}
  .score-num {{
    font-weight: 700;
    font-size: .9rem;
    min-width: 44px;
  }}
  .bar-track {{
    flex: 1;
    min-width: 80px;
    max-width: 140px;
    height: 6px;
    background: var(--border-soft);
    border-radius: 99px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width .3s ease;
  }}
  .fill-good {{ background: var(--green-line); }}
  .fill-ok   {{ background: var(--amber-line); }}
  .fill-poor {{ background: var(--red-line);  }}

  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: .73rem;
    font-weight: 600;
    white-space: nowrap;
  }}
  .badge-pass  {{ background: var(--green-light); color: var(--green); }}
  .badge-fail  {{ background: var(--red-light);   color: var(--red);  }}
  .badge-warn  {{ background: var(--amber-light);  color: var(--amber); }}
  .badge-neutral {{ background: var(--border-soft); color: var(--text-muted); }}

  .metric-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }}
  .metric-pill {{
    font-size: .7rem;
    color: var(--text-muted);
    background: var(--bg);
    border: 1px solid var(--border-soft);
    border-radius: 20px;
    padding: 2px 8px;
  }}

  .task-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 22px;
    border-bottom: 1px solid var(--border-soft);
    gap: 12px;
  }}
  .task-card-title {{
    font-size: .95rem;
    font-weight: 700;
  }}

  details.sample-detail {{
    border-bottom: 1px solid var(--border-soft);
  }}
  details.sample-detail:last-child {{ border-bottom: none; }}

  summary.sample-summary {{
    list-style: none;
    cursor: pointer;
    padding: 12px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    user-select: none;
    font-size: .83rem;
    color: var(--text-muted);
    transition: background .12s;
  }}
  summary.sample-summary:hover {{ background: var(--surface-alt); }}
  summary.sample-summary::-webkit-details-marker {{ display: none; }}
  summary.sample-summary .chevron {{
    width: 18px; height: 18px;
    border: 1.5px solid var(--border);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .6rem;
    color: var(--text-light);
    flex-shrink: 0;
    transition: transform .2s;
  }}
  details.sample-detail[open] summary.sample-summary .chevron {{
    transform: rotate(180deg);
  }}
  .sample-summary-left {{
    display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;
  }}
  .sample-doc-num {{
    font-weight: 700;
    color: var(--text);
    white-space: nowrap;
  }}
  .sample-desc {{
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }}

  .sample-body {{
    padding: 16px 22px 20px;
    background: var(--surface-alt);
    border-top: 1px solid var(--border-soft);
  }}
  .sample-section {{ margin-top: 18px; }}
  .sample-section:first-child {{ margin-top: 0; }}

  .sample-label {{
    font-size: .67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text-light);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .sample-label span {{ color: var(--text-muted); font-weight: 400; text-transform: none; letter-spacing: 0; }}

  pre.code-block {{
    background: var(--code-bg);
    border: 1px solid var(--code-border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    overflow-x: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Menlo', monospace;
    font-size: .78rem;
    line-height: 1.6;
    color: var(--code-text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 340px;
    overflow-y: auto;
  }}

  .test-list {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 8px;
  }}
  .test-row {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: .8rem;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    background: var(--surface);
    border: 1px solid var(--border-soft);
  }}
  .test-icon {{ flex-shrink: 0; font-size: .75rem; }}
  .test-name {{ flex: 1; color: var(--text); }}
  .test-msg  {{ color: var(--text-muted); font-size: .74rem; font-style: italic; }}
  .test-pass {{ border-color: #bbf7d0; background: #f0fdf4; }}
  .test-fail {{ border-color: #fecaca; background: #fff1f2; }}
  .test-error {{ border-color: var(--amber-mid); background: var(--amber-light); }}

  .diag-list {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 8px;
  }}
  .diag-row {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: .78rem;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    background: var(--surface);
    border: 1px solid var(--border-soft);
  }}
  .diag-loc {{ color: var(--text-light); font-size: .72rem; min-width: 52px; white-space: nowrap; }}
  .diag-msg {{ flex: 1; }}
  .diag-error   {{ border-color: #fecaca; color: var(--red); }}
  .diag-warning {{ border-color: var(--amber-mid); color: var(--amber); }}

  .offline-notice {{
    background: var(--amber-light);
    border: 1px solid var(--amber-mid);
    border-radius: var(--radius-md);
    padding: 10px 14px;
    margin-bottom: 20px;
    font-size: .8rem;
    color: var(--amber);
    display: none;
  }}

  @media (max-width: 640px) {{
    .page {{ padding: 20px 16px 60px; }}
    .header {{ flex-direction: column; }}
    .stats-row {{ flex-wrap: wrap; }}
    .stat-cell {{ flex: 1 1 40%; }}
  }}
</style>
</head>
<body>
<div class="page">

  <header class="header">
    <div class="header-left">
      <h1 style="color: rgb(46, 129, 224)">Luau <span>Bench</span></h1>
      <div class="header-meta" id="headerMeta"></div>
    </div>
    <div class="composite-pill" id="compositePill"></div>
  </header>

  <div class="offline-notice" id="offlineNotice">
    Chart.js could not load - charts need an internet connection on first view.
    All table data is still shown below.
  </div>

  <div class="stats-row" id="statsRow"></div>

  <div class="section-title">Score overview</div>
  <div class="card">
    <div class="chart-wrap">
      <canvas id="overviewChart" height="160"></canvas>
    </div>
  </div>

  <div class="section-title">Task results</div>
  <div class="card" style="overflow:auto;">
    <table class="results-table" id="resultsTable"></table>
  </div>

  <div id="samplesSection"></div>

</div>

<script>
const DATA = {data_json};

function esc(s) {{
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}}
function scoreClass(v) {{ return v >= 70 ? "score-good" : v >= 40 ? "score-ok" : "score-poor"; }}
function fillClass(v)  {{ return v >= 70 ? "fill-good"  : v >= 40 ? "fill-ok"  : "fill-poor"; }}
function fmtDur(s) {{
  if (s < 60) return s.toFixed(1) + "s";
  return Math.floor(s/60) + "m " + (s%60|0) + "s";
}}
function fmtDate(iso) {{
  try {{ return new Date(iso).toLocaleString(undefined, {{dateStyle:"medium",timeStyle:"short"}}); }}
  catch {{ return iso; }}
}}

(function() {{
  const d = DATA;
  const dots = [
    `<span>${{esc(d.model_id)}}</span>`,
    `<span class="dot"></span><span>Provider: ${{esc(d.provider)}}</span>`,
    `<span class="dot"></span><span>Run: ${{esc(d.run_id)}}</span>`,
    `<span class="dot"></span><span>${{fmtDur(d.elapsed_s)}}</span>`,
    `<span class="dot"></span><span>${{fmtDate(d.generated_at)}}</span>`,
  ];
  document.getElementById("headerMeta").innerHTML = dots.join("");
}})();

(function() {{
  const c = DATA.composite;
  let html;
  if (c === null || c === undefined) {{
    html = `<div class="score score-none">N/A</div>
            <div class="score-label">Composite</div>`;
  }} else {{
    const cls = scoreClass(c);
    const ci = (DATA.composite_lo !== null && DATA.composite_hi !== null)
      ? `<div class="ci-line">[${{DATA.composite_lo.toFixed(1)}}, ${{DATA.composite_hi.toFixed(1)}}] 95% CI</div>`
      : "";
    html = `<div class="score ${{cls}}">${{c.toFixed(1)}}%</div>
            <div class="score-label">Composite</div>${{ci}}`;
  }}
  document.getElementById("compositePill").innerHTML = html;
}})();

(function() {{
  const tasks = DATA.tasks;
  const nOk  = tasks.filter(t => !t.error).length;
  const nErr = tasks.length - nOk;
  const avgScore = nOk > 0
    ? (tasks.filter(t=>!t.error).reduce((s,t)=>s+(t.primary_value||0),0)/nOk).toFixed(1)+"%" : "n/a";
  const totalDocs = tasks.reduce((s,t)=>s+t.num_docs,0);

  const cells = [
    {{ num: tasks.length,    label: "Tasks",      accent: true }},
    {{ num: nOk,             label: "Completed"  }},
    {{ num: nErr||"n/a",       label: "Errors"     }},
    {{ num: totalDocs,       label: "Documents"  }},
    {{ num: avgScore,        label: "Avg Score"  }},
  ];

  document.getElementById("statsRow").innerHTML = cells.map(c =>
    `<div class="stat-cell${{c.accent?" accent":""}}">
      <div class="stat-num">${{esc(c.num)}}</div>
      <div class="stat-label">${{esc(c.label)}}</div>
    </div>`
  ).join("");
}})();

(function() {{
  const head = `<thead><tr>
    <th>Task</th><th>Ver</th><th>Docs</th>
    <th>Primary score</th><th>Metrics</th>
  </tr></thead>`;

  const rows = DATA.tasks.map(t => {{
    const v  = t.primary_value ?? 0;
    const fc = fillClass(v);
    const sc = scoreClass(v);

    let scoreCell;
    if (t.error) {{
      scoreCell = `<span class="badge badge-fail">ERROR</span>
                   <span style="color:var(--text-muted);font-size:.75rem;margin-left:6px">${{esc(t.error.slice(0,80))}}</span>`;
    }} else {{
      const ci = (t.ci_lo !== null && t.ci_hi !== null)
        ? `<span style="font-size:.7rem;color:var(--text-light);margin-left:6px">[${{t.ci_lo.toFixed(1)}}, ${{t.ci_hi.toFixed(1)}}]</span>` : "";
      scoreCell = `<div class="score-cell">
        <span class="score-num ${{sc}}">${{v.toFixed(1)}}%</span>
        <div class="bar-track"><div class="bar-fill ${{fc}}" style="width:${{v}}%"></div></div>
        ${{ci}}
      </div>`;
    }}

    const pills = t.metrics.slice(0,6).map(m =>
      `<span class="metric-pill">${{esc(m.name)}} ${{m.value.toFixed(1)}}%</span>`
    ).join("");

    return `<tr>
      <td><span class="task-name">${{esc(t.name)}}</span></td>
      <td><span class="task-version">v${{t.version}}</span></td>
      <td style="color:var(--text-muted)">${{t.num_docs}}</td>
      <td>${{scoreCell}}</td>
      <td><div class="metric-pills">${{pills}}</div></td>
    </tr>`;
  }}).join("");

  document.getElementById("resultsTable").innerHTML = head + `<tbody>${{rows}}</tbody>`;
}})();

(function() {{
  const tasksWithSamples = DATA.tasks.filter(t => t.samples && t.samples.length > 0);
  if (!tasksWithSamples.length) return;

  const container = document.getElementById("samplesSection");
  let html = `<div class="section-title">Sample outputs</div>`;

  tasksWithSamples.forEach(t => {{
    const v   = t.primary_value ?? 0;
    const sc  = scoreClass(v);
    const badge = t.error
      ? `<span class="badge badge-fail">ERROR</span>`
      : `<span class="badge badge-neutral ${{sc.replace("score-","score-")}}" style="font-size:.78rem">${{v.toFixed(1)}}%</span>`;

    html += `<div class="card" style="margin-bottom:20px">
      <div class="task-card-header">
        <span class="task-card-title">${{esc(t.name)}}</span>
        ${{badge}}
      </div>`;

    t.samples.forEach((s, i) => {{
      const execBadge = s.exec
        ? (s.exec.timed_out
            ? `<span class="badge badge-fail">Timeout</span>`
            : s.exec.passed === s.exec.total && s.exec.total > 0
              ? `<span class="badge badge-pass">${{s.exec.passed}}/${{s.exec.total}} passed</span>`
              : `<span class="badge badge-fail">${{s.exec.passed}}/${{s.exec.total}} passed</span>`)
        : "";

      html += `<details class="sample-detail">
        <summary class="sample-summary">
          <div class="sample-summary-left">
            <span class="sample-doc-num">Doc ${{i+1}}</span>
            <span class="sample-desc">${{esc(s.description.slice(0,90))}}${{s.description.length>90?"...":""}}</span>
            ${{execBadge}}
          </div>
          <span class="chevron">▾</span>
        </summary>
        <div class="sample-body">`;

      if (s.raw) {{
        html += `<div class="sample-section">
          <div class="sample-label">Raw output <span>${{s.raw.length}} chars</span></div>
          <pre class="code-block">${{esc(s.raw)}}</pre>
        </div>`;
      }}

      html += `<div class="sample-section">
        <div class="sample-label">Extracted code <span>${{(s.prediction||"").length}} chars</span></div>
        <pre class="code-block">${{esc(s.prediction || "(empty)")}}</pre>
      </div>`;

      if (s.reference) {{
        html += `<div class="sample-section">
          <div class="sample-label">Reference</div>
          <pre class="code-block">${{esc(s.reference)}}</pre>
        </div>`;
      }}

      if (s.exec) {{
        const ex = s.exec;
        const total = ex.total;
        if (total > 0 || ex.timed_out) {{
          const statusBadge = ex.timed_out
            ? `<span class="badge badge-fail">Timeout</span>`
            : ex.passed === total
              ? `<span class="badge badge-pass">All ${{total}} tests passed</span>`
              : `<span class="badge badge-fail">${{ex.passed}}/${{total}} passed</span>`;

          const testRows = ex.tests.map(tt => {{
            const rowCls = tt.status === "pass" ? "test-pass" : tt.status === "fail" ? "test-fail" : "test-error";
            const icon = tt.status === "pass" ? "pass" : tt.status === "fail" ? "fail" : "err";
            const msg = tt.message ? `<span class="test-msg">${{esc(tt.message.slice(0,120))}}</span>` : "";
            return `<div class="test-row ${{rowCls}}">
              <span class="test-icon">${{icon}}</span>
              <span class="test-name">${{esc(tt.name)}}</span>
              ${{msg}}
            </div>`;
          }}).join("");

          const stderr = ex.stderr
            ? `<div class="sample-label" style="margin-top:12px">stderr</div>
               <pre class="code-block">${{esc(ex.stderr)}}</pre>`
            : "";

          html += `<div class="sample-section">
            <div class="sample-label">Execution <span>${{ex.runtime_ms}}ms</span></div>
            ${{statusBadge}}
            <div class="test-list">${{testRows}}</div>
            ${{stderr}}
          </div>`;
        }}
      }}

      if (s.analyze) {{
        const az = s.analyze;
        if (az.errors === 0 && az.warnings === 0) {{
          html += `<div class="sample-section">
            <div class="sample-label">Static analysis</div>
            <span class="badge badge-pass">Clean - no issues</span>
          </div>`;
        }} else {{
          const diagRows = az.diagnostics.map(d => {{
            const cls = d.severity === "error" ? "diag-error" : "diag-warning";
            return `<div class="diag-row ${{cls}}">
              <span class="diag-loc">${{d.line}}:${{d.col}}</span>
              <span class="diag-msg">${{esc(d.message.slice(0,120))}}</span>
            </div>`;
          }}).join("");
          html += `<div class="sample-section">
            <div class="sample-label">Static analysis
              <span>${{az.errors}} error(s), ${{az.warnings}} warning(s)</span>
            </div>
            <div class="diag-list">${{diagRows}}</div>
          </div>`;
        }}
      }}

      html += `</div></details>`;
    }});

    html += `</div>`;
  }});

  container.innerHTML = html;
}})();

function buildChart(Chart) {{
  const tasks = DATA.tasks.filter(t => !t.error && t.primary !== null);
  const labels = tasks.map(t => t.name);
  const vals = tasks.map(t => t.primary_value ?? 0);

  const barColor = "rgba(180, 83, 9, 0.75)";
  const barHover = "rgba(180, 83, 9, 1)";
  const gridColor = "#f0e8d4";
  const tickColor = "#78716c";

  new Chart(document.getElementById("overviewChart"), {{
    type: "bar",
    data: {{
      labels,
      datasets: [{{
        label: "Primary score (%)",
        data: vals,
        backgroundColor: vals.map(v =>
          v >= 70 ? "rgba(22,163,74,.75)"
                  : v >= 40 ? "rgba(180,83,9,.75)"
                            : "rgba(220,38,38,.75)"
        ),
        hoverBackgroundColor: vals.map(v =>
          v >= 70 ? "rgba(22,163,74,1)"
                  : v >= 40 ? "rgba(180,83,9,1)"
                            : "rgba(220,38,38,1)"
        ),
        borderRadius: 5,
        borderSkipped: false,
      }}]
    }},
    options: {{
      indexAxis: "y",
      responsive: true,
      animation: {{ duration: 500, easing: "easeOutQuart" }},
      layout: {{ padding: {{ right: 16 }} }},
      scales: {{
        x: {{
          min: 0, max: 100,
          grid: {{ color: gridColor }},
          border: {{ display: false }},
          ticks: {{ color: tickColor, callback: v => v + "%" }},
        }},
        y: {{
          grid: {{ display: false }},
          border: {{ display: false }},
          ticks: {{ color: tickColor, font: {{ size: 12, weight: "500" }} }},
        }},
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: "#ffffff",
          borderColor: "#e8d9bc",
          borderWidth: 1,
          titleColor: "#1c1917",
          bodyColor: "#78716c",
          padding: 10,
          callbacks: {{
            label: ctx => ` ${{ctx.parsed.x.toFixed(1)}}%`,
            afterLabel: ctx => {{
              const t = tasks[ctx.dataIndex];
              return (t.ci_lo !== null && t.ci_hi !== null)
                ? ` 95% CI: [${{t.ci_lo.toFixed(1)}}, ${{t.ci_hi.toFixed(1)}}]`
                : "";
            }}
          }}
        }}
      }}
    }}
  }});
}}

const cs = document.createElement("script");
cs.src = "https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js";
cs.onload = () => buildChart(Chart);
cs.onerror = () => {{ document.getElementById("offlineNotice").style.display = "block"; }};
document.head.appendChild(cs);
</script>
</body>
</html>"""
