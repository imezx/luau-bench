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
<title>Luau Bench — {data["model_id"]}</title>
<style>
  :root {{
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2a2d3a;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --green:   #22c55e;
    --amber:   #f59e0b;
    --red:     #ef4444;
    --blue:    #3b82f6;
    --accent:  #818cf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 14px; line-height: 1.6;
    padding: 24px;
  }}
  h1 {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin: 32px 0 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
  h3 {{ font-size: .95rem; font-weight: 600; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
  }}
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px; margin-top: 16px;
  }}
  .stat {{ text-align: center; }}
  .stat-val {{ font-size: 2rem; font-weight: 800; }}
  .stat-lbl {{ color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }}
  .badge {{
    display: inline-block; border-radius: 8px; padding: 4px 14px;
    font-weight: 700; font-size: 1.8rem;
  }}
  .badge-green {{ background: #14532d; color: var(--green); }}
  .badge-amber {{ background: #451a03; color: var(--amber); }}
  .badge-red   {{ background: #450a0a; color: var(--red);   }}
  .ci {{ font-size: .75rem; color: var(--muted); margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.03); }}
  .bar-wrap {{ background: var(--border); border-radius: 4px; height: 8px; width: 120px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .good  {{ background: var(--green); }}
  .ok    {{ background: var(--amber); }}
  .poor  {{ background: var(--red);   }}
  .error-badge {{ background: #450a0a; color: var(--red); border-radius: 6px; padding: 2px 8px; font-size: .75rem; }}
  .pass-badge {{ background: #14532d; color: var(--green); border-radius: 6px; padding: 2px 8px; font-size: .75rem; }}
  details {{ margin-bottom: 8px; }}
  summary {{
    cursor: pointer; padding: 10px 14px; background: var(--border);
    border-radius: 8px; font-weight: 500; user-select: none;
    list-style: none; display: flex; justify-content: space-between;
    align-items: center;
  }}
  summary::-webkit-details-marker {{ display: none; }}
  summary::after {{ content: "▶"; font-size: .7rem; color: var(--muted); transition: .2s; }}
  details[open] summary::after {{ content: "▼"; }}
  .sample-box {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 0 0 8px 8px; padding: 16px;
  }}
  .sample-section {{ margin-top: 14px; }}
  .sample-section:first-child {{ margin-top: 0; }}
  pre {{
    background: #090b12; border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; overflow-x: auto;
    font-size: .8rem; color: #a9b1d6; white-space: pre-wrap;
    word-break: break-word;
  }}
  .label {{ color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; margin: 10px 0 4px; }}
  .test-row {{ display: flex; align-items: baseline; gap: 8px; padding: 3px 0; font-size: .8rem; }}
  .test-icon {{ width: 14px; flex-shrink: 0; }}
  .test-name {{ color: var(--text); flex: 1; }}
  .test-msg {{ color: var(--muted); font-size: .75rem; }}
  .diag-row {{ display: flex; align-items: baseline; gap: 8px; padding: 3px 0; font-size: .8rem; }}
  .diag-loc {{ color: var(--muted); font-size: .75rem; min-width: 60px; }}
  .diag-msg {{ color: var(--text); flex: 1; }}
  .sev-error {{ color: var(--red); }}
  .sev-warning {{ color: var(--amber); }}
  .offline-warn {{
    background: #451a03; border: 1px solid var(--amber); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px; color: var(--amber); display: none;
  }}
  canvas {{ max-width: 100%; }}
</style>
</head>
<body>

<h1>Luau Bench Report</h1>

<div class="offline-warn" id="offlineWarn">
  Chart.js could not be loaded — charts require an internet connection on first use.
  All data is still available in the table below.
</div>

<div class="card" id="summaryCard"></div>

<h2>Score Overview</h2>
<div class="card"><canvas id="overviewChart" height="180"></canvas></div>

<h2>Task Details</h2>
<div class="card"><table id="taskTable"></table></div>

<div id="samplesSection"></div>

<script>
const DATA = {data_json};

function esc(s) {{
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}}
function colourClass(v) {{
  return v >= 70 ? "good" : v >= 40 ? "ok" : "poor";
}}
function badgeClass(v) {{
  return v >= 70 ? "badge-green" : v >= 40 ? "badge-amber" : "badge-red";
}}
function fmtDuration(s) {{
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60), rem = (s % 60).toFixed(0);
  return m + "m " + rem + "s";
}}

(function buildSummary() {{
  const d   = DATA;
  const c   = d.composite;
  const clo = d.composite_lo;
  const chi = d.composite_hi;
  const n   = d.tasks.length;
  const nOk = d.tasks.filter(t => !t.error).length;

  let compositeHtml = "<em style='color:var(--muted)'>—</em>";
  if (c !== null && c !== undefined) {{
    const bc = badgeClass(c);
    compositeHtml = `<span class="badge ${{bc}}">${{c.toFixed(1)}}</span>`;
    if (clo !== null && chi !== null)
      compositeHtml += `<div class="ci">95% CI [${{clo.toFixed(1)}}, ${{chi.toFixed(1)}}]</div>`;
  }}

  document.getElementById("summaryCard").innerHTML = `
    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px">
      <div>
        <div style="color:var(--muted);font-size:.8rem;margin-bottom:4px">MODEL</div>
        <div style="font-size:1.1rem;font-weight:700">${{esc(d.model_id)}}</div>
        <div style="color:var(--muted);font-size:.78rem;margin-top:4px">
          Provider: ${{esc(d.provider)}} &nbsp;·&nbsp;
          Run ID: ${{esc(d.run_id)}} &nbsp;·&nbsp;
          ${{fmtDuration(d.elapsed_s)}}
        </div>
        <div style="color:var(--muted);font-size:.75rem;margin-top:2px">
          ${{new Date(d.generated_at).toLocaleString()}}
        </div>
      </div>
      <div class="summary-grid" style="margin-top:0;flex:1;max-width:500px">
        <div class="stat">
          <div class="stat-val" style="color:var(--accent)">${{n}}</div>
          <div class="stat-lbl">Tasks</div>
        </div>
        <div class="stat">
          <div class="stat-val" style="color:${{nOk === n ? 'var(--green)' : 'var(--amber)'}}">${{nOk}}</div>
          <div class="stat-lbl">Passed</div>
        </div>
        <div class="stat">
          <div class="stat-val">${{compositeHtml}}</div>
          <div class="stat-lbl">Composite</div>
        </div>
      </div>
    </div>
  `;
}})();

(function buildTable() {{
  const rows = DATA.tasks.map(t => {{
    const v   = t.primary_value ?? 0;
    const cc  = colourClass(v);
    const bar = `<span class="bar-wrap"><span class="bar-fill ${{cc}}" style="width:${{v}}%"></span></span>`;
    const ci  = (t.ci_lo !== null && t.ci_hi !== null)
      ? `<span class="ci">[${{t.ci_lo.toFixed(1)}}, ${{t.ci_hi.toFixed(1)}}]</span>` : "";

    const metricsCells = t.metrics.slice(0, 6)
      .map(m => `${{esc(m.name)}}=${{m.value.toFixed(3)}}`)
      .join("<br>");

    const errCell = t.error
      ? `<span class="error-badge">ERROR</span> ${{esc(t.error).slice(0,80)}}`
      : `${{v.toFixed(1)}}%${{bar}}${{ci}}`;

    return `<tr>
      <td><strong>${{esc(t.name)}}</strong></td>
      <td style="color:var(--muted)">v${{t.version}}</td>
      <td>${{t.num_docs}}</td>
      <td>${{errCell}}</td>
      <td style="font-size:.75rem;color:var(--muted)">${{metricsCells}}</td>
    </tr>`;
  }}).join("");

  document.getElementById("taskTable").innerHTML = `
    <thead><tr>
      <th>Task</th><th>Ver</th><th>Docs</th>
      <th>Primary Score</th><th>All Metrics</th>
    </tr></thead>
    <tbody>${{rows}}</tbody>
  `;
}})();

function renderExec(exec) {{
  if (!exec) return "";
  const total = exec.total;
  if (total === 0 && !exec.timed_out) return "";
  const statusBadge = exec.timed_out
    ? `<span class="error-badge">TIMEOUT</span>`
    : exec.passed === total
      ? `<span class="pass-badge">${{exec.passed}}/${{total}} passed</span>`
      : `<span class="error-badge">${{exec.passed}}/${{total}} passed</span>`;

  let rows = exec.tests.map(t => {{
    const icon = t.status === "pass" ? "✓" : t.status === "fail" ? "✗" : "⚠";
    const col  = t.status === "pass" ? "var(--green)" : t.status === "fail" ? "var(--red)" : "var(--amber)";
    const msg  = t.message ? ` <span class="test-msg">${{esc(t.message.slice(0,120))}}</span>` : "";
    return `<div class="test-row">
      <span class="test-icon" style="color:${{col}}">${{icon}}</span>
      <span class="test-name">${{esc(t.name)}}</span>${{msg}}
    </div>`;
  }}).join("");

  let stderr = "";
  if (exec.stderr) {{
    stderr = `<div class="label" style="margin-top:10px">stderr</div><pre>${{esc(exec.stderr)}}</pre>`;
  }}

  return `<div class="sample-section">
    <div class="label">Execution <span style="float:right;font-size:.8rem">${{exec.runtime_ms}}ms</span></div>
    <div style="margin-bottom:6px">${{statusBadge}}</div>
    ${{rows}}${{stderr}}
  </div>`;
}}

function renderAnalyze(az) {{
  if (!az) return "";
  if (az.errors === 0 && az.warnings === 0) {{
    return `<div class="sample-section">
      <div class="label">Static Analysis</div>
      <span class="pass-badge">Clean — no issues</span>
    </div>`;
  }}
  const rows = az.diagnostics.map(d => {{
    const cls = d.severity === "error" ? "sev-error" : "sev-warning";
    return `<div class="diag-row">
      <span class="diag-loc ${{cls}}">${{d.line}}:${{d.col}}</span>
      <span class="diag-msg">${{esc(d.message.slice(0,120))}}</span>
    </div>`;
  }}).join("");

  return `<div class="sample-section">
    <div class="label">Static Analysis — ${{az.errors}} error(s), ${{az.warnings}} warning(s)</div>
    ${{rows}}
  </div>`;
}}

(function buildSamples() {{
  const tasksWithSamples = DATA.tasks.filter(t => t.samples && t.samples.length > 0);
  if (!tasksWithSamples.length) return;

  const container = document.getElementById("samplesSection");
  let html = "<h2>Sample Outputs</h2>";

  tasksWithSamples.forEach(t => {{
    html += `<div class="card"><h3 style="margin-bottom:14px;color:var(--accent)">${{esc(t.name)}}</h3>`;
    t.samples.forEach((s, i) => {{
      const execHtml    = renderExec(s.exec);
      const analyzeHtml = renderAnalyze(s.analyze);
      html += `<details>
        <summary>Doc ${{i+1}} — ${{esc(s.description.slice(0,80))}}${{s.description.length > 80 ? "…" : ""}}</summary>
        <div class="sample-box">
          ${{s.raw ? `<div class="label">Raw output</div><pre>${{esc(s.raw)}}</pre>` : ""}}
          <div class="sample-section">
            <div class="label">Extracted code</div>
            <pre>${{esc(s.prediction || "(empty)")}}</pre>
          </div>
          ${{s.reference ? `<div class="sample-section"><div class="label">Reference</div><pre>${{esc(s.reference)}}</pre></div>` : ""}}
          ${{execHtml}}
          ${{analyzeHtml}}
        </div>
      </details>`;
    }});
    html += "</div>";
  }});

  container.innerHTML = html;
}})();

function buildChart(Chart) {{
  const tasks  = DATA.tasks.filter(t => !t.error && t.primary !== null);
  const labels = tasks.map(t => t.name);
  const vals   = tasks.map(t => t.primary_value ?? 0);

  const colours = vals.map(v =>
    v >= 70 ? "rgba(34,197,94,.8)" : v >= 40 ? "rgba(245,158,11,.8)" : "rgba(239,68,68,.8)"
  );

  new Chart(document.getElementById("overviewChart"), {{
    type: "bar",
    data: {{
      labels,
      datasets: [{{
        label: "Primary Score (%)",
        data: vals,
        backgroundColor: colours,
        borderRadius: 6,
      }}]
    }},
    options: {{
      indexAxis: "y",
      responsive: true,
      scales: {{
        x: {{ min: 0, max: 100, grid: {{ color: "#2a2d3a" }}, ticks: {{ color: "#94a3b8", callback: v => v + "%" }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: "#e2e8f0" }} }},
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => ` ${{ctx.parsed.x.toFixed(1)}}%`,
            afterLabel: ctx => {{
              const t = tasks[ctx.dataIndex];
              if (t.ci_lo !== null && t.ci_hi !== null)
                return `95% CI: [${{t.ci_lo.toFixed(1)}}%, ${{t.ci_hi.toFixed(1)}}%]`;
              return "";
            }}
          }}
        }}
      }}
    }}
  }});
}}

const chartScript = document.createElement("script");
chartScript.src = "https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js";
chartScript.onload = () => buildChart(Chart);
chartScript.onerror = () => {{
  document.getElementById("offlineWarn").style.display = "block";
}};
document.head.appendChild(chartScript);
</script>
</body>
</html>"""
