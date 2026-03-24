"""
Live training dashboard. Reads checkpoints/train_log.jsonl and serves charts.

Usage: python dashboard.py
Then open http://localhost:5000
"""

import json
import os
from flask import Flask, Response

app = Flask(__name__)

LOG_PATH = os.path.join(os.path.dirname(__file__) or ".", "checkpoints", "train_log.jsonl")


def read_log():
    train = []
    evals = []
    gates = []
    stages = []
    samples = []
    if not os.path.exists(LOG_PATH):
        return train, evals, gates, stages, samples
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            t = entry.get("type")
            if t == "gate_check":
                gates.append(entry)
            elif t == "stage_start":
                stages.append(entry)
            elif t == "samples":
                samples.append(entry)
            elif t in ("eval", "final_eval"):
                evals.append(entry)
            elif "loss" in entry:
                train.append(entry)
    return train, evals, gates, stages, samples


@app.route("/api/data")
def api_data():
    train, evals, gates, stages, samples = read_log()
    return {"train": train, "evals": evals, "gates": gates, "stages": stages, "samples": samples}


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


HTML = """<!DOCTYPE html>
<html>
<head>
<title>NullRoot Training Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Menlo', 'Consolas', monospace; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 4px; font-size: 22px; }
  .subtitle { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card h2 { color: #58a6ff; font-size: 14px; margin-bottom: 12px; }
  canvas { width: 100% !important; height: 250px !important; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; text-align: center; }
  .stat .value { font-size: 24px; color: #58a6ff; font-weight: bold; }
  .stat .label { font-size: 11px; color: #8b949e; margin-top: 4px; }
  .full { grid-column: 1 / -1; }
  #status { color: #3fb950; font-size: 12px; }

  .stage-banner { background: #161b22; border: 1px solid #58a6ff; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .stage-banner h2 { color: #58a6ff; font-size: 16px; margin-bottom: 8px; }
  .stage-pipeline { display: flex; gap: 4px; margin-top: 10px; }
  .stage-pip { flex: 1; height: 8px; border-radius: 4px; background: #21262d; }
  .stage-pip.done { background: #3fb950; }
  .stage-pip.active { background: #58a6ff; animation: pulse 1.5s infinite; }
  .stage-pip.pending { background: #21262d; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

  .tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
  .tbl th { text-align: left; color: #58a6ff; padding: 6px 10px; border-bottom: 1px solid #30363d; }
  .tbl td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
  .pass { color: #3fb950; font-weight: bold; }
  .fail { color: #f85149; }

  .sample { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; margin-bottom: 8px; font-size: 13px; }
  .sample .cmd { color: #58a6ff; font-weight: bold; margin-bottom: 4px; }
  .sample .expected { color: #8b949e; }
  .sample .got { margin-top: 2px; }
  .sample .got.ok { color: #3fb950; }
  .sample .got.wrong { color: #f85149; }
  .sample .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; margin-left: 8px; }
  .sample .badge.ok { background: #3fb95022; color: #3fb950; }
  .sample .badge.wrong { background: #f8514922; color: #f85149; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
</head>
<body>

<h1>NullRoot Training Dashboard</h1>
<p class="subtitle">~35M param transformer with memory bank learning to simulate bash &nbsp; <span id="status">connecting...</span></p>

<div class="stage-banner">
  <h2 id="stage-title">Waiting for data...</h2>
  <div id="stage-desc" style="color:#8b949e; font-size:13px;"></div>
  <div class="stage-pipeline" id="stage-pipeline"></div>
  <div id="stage-defs" style="margin-top:12px; display:grid; grid-template-columns:repeat(4,1fr); gap:6px; font-size:11px;"></div>
</div>

<div class="stats">
  <div class="stat"><div class="value" id="s-step">-</div><div class="label">Global Step</div></div>
  <div class="stat"><div class="value" id="s-loss">-</div><div class="label">Train Loss</div></div>
  <div class="stat"><div class="value" id="s-ppl">-</div><div class="label">Perplexity</div></div>
  <div class="stat"><div class="value" id="s-speed">-</div><div class="label">Tokens/sec</div></div>
</div>
<div class="stats">
  <div class="stat"><div class="value" id="s-gate">-</div><div class="label">Gate Tests</div></div>
  <div class="stat"><div class="value" id="s-lr">-</div><div class="label">Learning Rate</div></div>
  <div class="stat"><div class="value" id="s-gnorm">-</div><div class="label">Grad Norm</div></div>
  <div class="stat"><div class="value" id="s-gpu">-</div><div class="label">GPU Peak (GB)</div></div>
</div>

<div class="grid">
  <div class="card"><h2>Training Loss</h2><canvas id="lossChart"></canvas></div>
  <div class="card"><h2>Perplexity</h2><canvas id="pplChart"></canvas></div>
  <div class="card"><h2>Learning Rate</h2><canvas id="lrChart"></canvas></div>
  <div class="card"><h2>Throughput (tok/s)</h2><canvas id="speedChart"></canvas></div>
</div>

<div class="grid">
  <div class="card full">
    <h2>Gate Checks</h2>
    <table class="tbl" id="gateTable">
      <thead><tr><th>Stage</th><th>Step</th><th>Tests</th><th>All</th><th>Result</th><th>Time</th></tr></thead>
      <tbody id="gateBody"><tr><td colspan="7" style="color:#8b949e">No gate checks yet</td></tr></tbody>
    </table>
  </div>
</div>

<div class="grid">
  <div class="card full">
    <h2>Sample Outputs (latest gate check)</h2>
    <div id="samplesContainer" style="color:#8b949e; font-size:13px;">No samples yet</div>
  </div>
</div>

<script>
const STAGE_DEFS = [
  { name: 'S1', label: 'all commands', cmds: ['mkdir', 'cd', 'ls', 'pwd', 'touch', 'echo', 'cat', 'rm', 'cp', 'mv', 'head', 'wc', 'find', 'grep'] },
];
const STAGE_NAMES = STAGE_DEFS.map(s => s.label);
const STAGE_COLORS = [
  '#3fb950'
];

const chartOpts = (label, color) => ({
  type: 'line',
  data: { datasets: [{ label, data: [], borderColor: color, backgroundColor: color + '22', fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.3 }] },
  options: {
    responsive: true, animation: false,
    scales: {
      x: { type: 'linear', title: { display: false }, ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
    },
    plugins: { legend: { display: false }, annotation: { annotations: {} } }
  }
});

const lossChart = new Chart(document.getElementById('lossChart'), chartOpts('Loss', '#f85149'));
const pplChart  = new Chart(document.getElementById('pplChart'),  chartOpts('PPL',  '#d29922'));
const lrChart   = new Chart(document.getElementById('lrChart'),   chartOpts('LR',   '#3fb950'));
const speedChart= new Chart(document.getElementById('speedChart'),chartOpts('tok/s','#58a6ff'));
const allCharts = [lossChart, pplChart, lrChart, speedChart];

function buildStageAnnotations(stages) {
  // Build vertical line annotations for each stage transition
  const annotations = {};
  stages.forEach((s, i) => {
    if (i === 0) return; // skip first stage, it starts at 0
    annotations['stage' + i] = {
      type: 'line',
      xMin: s.global_step,
      xMax: s.global_step,
      borderColor: STAGE_COLORS[s.stage % STAGE_COLORS.length] + 'aa',
      borderWidth: 2,
      borderDash: [6, 4],
      label: {
        display: true,
        content: 'S' + s.stage,
        position: 'start',
        backgroundColor: STAGE_COLORS[s.stage % STAGE_COLORS.length] + '44',
        color: STAGE_COLORS[s.stage % STAGE_COLORS.length],
        font: { size: 10, family: 'monospace' },
        padding: 3
      }
    };
  });
  return annotations;
}

function update(data) {
  const t = data.train;
  const gates = data.gates || [];
  const stages = data.stages || [];

  // Update stage annotations on all charts
  const annotations = buildStageAnnotations(stages);
  allCharts.forEach(chart => {
    chart.options.plugins.annotation.annotations = annotations;
  });

  // Stage pipeline
  const currentStage = t.length ? t[t.length-1].stage : (stages.length ? stages[stages.length-1].stage : 0);
  const passedStages = new Set(gates.filter(g => g.gate_passed).map(g => g.stage));
  const pipeline = document.getElementById('stage-pipeline');
  pipeline.innerHTML = STAGE_NAMES.map((name, i) => {
    let cls = 'stage-pip ';
    if (passedStages.has(i)) cls += 'done';
    else if (i === currentStage) cls += 'active';
    else cls += 'pending';
    return '<div class="' + cls + '" title="S' + i + ': ' + name + '"></div>';
  }).join('');

  const stageTitle = document.getElementById('stage-title');
  const stageDesc = document.getElementById('stage-desc');
  if (stages.length) {
    const latest = stages[stages.length-1];
    stageTitle.textContent = latest.name || ('Stage ' + latest.stage);
    stageDesc.textContent = 'Stage ' + (currentStage+1) + '/1 | ' + passedStages.size + ' gates passed';
  }

  // Stage definitions
  const defsEl = document.getElementById('stage-defs');
  defsEl.innerHTML = STAGE_DEFS.map((s, i) => {
    let border = STAGE_COLORS[i];
    let opacity = '44';
    let badge = '';
    if (passedStages.has(i)) { opacity = 'aa'; badge = ' \u2713'; }
    else if (i === currentStage) { opacity = 'ff'; badge = ' \u25C0'; }
    return '<div style="border:1px solid ' + border + opacity + '; border-radius:4px; padding:6px; ' +
      (i === currentStage ? 'background:' + border + '11;' : '') + '">' +
      '<div style="color:' + border + '; font-weight:bold; margin-bottom:3px;">' + s.name + badge + '</div>' +
      '<div style="color:#8b949e;">' + s.cmds.join(' ') + '</div></div>';
  }).join('');


  if (!t.length) return;

  function xyData(key) {
    return t.map(e => ({ x: e.step, y: e[key] }));
  }

  function setChart(chart, points) {
    chart.data.datasets[0].data = points;
    chart.update();
  }
  setChart(lossChart, xyData('loss'));
  setChart(pplChart, xyData('ppl'));
  setChart(lrChart, xyData('lr'));
  setChart(speedChart, t.map(e => ({ x: e.step, y: e.tokens_per_sec || e.steps_per_sec || 0 })));

  const last = t[t.length - 1];
  document.getElementById('s-step').textContent = (last.step || 0).toLocaleString();
  document.getElementById('s-loss').textContent = (last.loss || 0).toFixed(4);
  document.getElementById('s-ppl').textContent = (last.ppl || 0).toFixed(1);
  document.getElementById('s-speed').textContent = (last.tokens_per_sec || last.steps_per_sec || 0).toLocaleString();
  document.getElementById('s-lr').textContent = (last.lr || 0).toExponential(2);
  document.getElementById('s-gnorm').textContent = (last.grad_norm || 0).toFixed(2);
  document.getElementById('s-gpu').textContent = (last.gpu_peak_gb || 0).toFixed(1);

  // Latest gate val loss
  if (gates.length) {
    const lastGate = gates[gates.length - 1];
    const el = document.getElementById('s-gate');
    const gc = lastGate.gate_correct || 0;
    const gt = lastGate.gate_total || 1;
    el.textContent = gc + '/' + gt;
    el.style.color = lastGate.gate_passed ? '#3fb950' : '#f85149';
  }

  // Gate table
  const gateBody = document.getElementById('gateBody');
  if (gates.length) {
    gateBody.innerHTML = gates.map(g => {
      const passed = g.gate_passed;
      const gc = g.gate_correct || 0;
      const gt = g.gate_total || 0;
      const ac = g.all_correct || 0;
      const at = g.all_total || 0;
      return '<tr>' +
        '<td>S' + (g.stage+1) + ': ' + (STAGE_NAMES[g.stage]||'?') + '</td>' +
        '<td>' + (g.step||'') + '</td>' +
        '<td>' + gc + '/' + gt + '</td>' +
        '<td>' + ac + '/' + at + '</td>' +
        '<td class="' + (passed ? 'pass' : 'fail') + '">' + (passed ? 'PASS' : 'FAIL') + '</td>' +
        '<td>' + (g.timestamp||'') + '</td></tr>';
    }).join('');
  }

  // Sample outputs
  const samplesEl = document.getElementById('samplesContainer');
  const allSamples = data.samples || [];
  if (allSamples.length) {
    const latest = allSamples[allSamples.length - 1];
    const items = latest.samples || [];
    const correct = items.filter(s => s.match).length;
    samplesEl.innerHTML = '<div style="margin-bottom:8px; color:#c9d1d9;">' +
      correct + '/' + items.length + ' correct</div>' +
      items.map(s => {
        const cls = s.match ? 'ok' : 'wrong';
        function esc(str) { return (str||'').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
        return '<div class="sample">' +
          '<div class="cmd">$ ' + esc(s.command) +
          '<span class="badge ' + cls + '">' + (s.match ? 'MATCH' : 'MISMATCH') + '</span></div>' +
          '<div class="expected">expected: <code>' + esc(s.expected) + '</code></div>' +
          '<div class="got ' + cls + '">got: <code>' + esc(s.generated) + '</code></div>' +
          '</div>';
      }).join('');
  }

  document.getElementById('status').textContent = 'updated ' + new Date().toLocaleTimeString();
}

async function poll() {
  try {
    const res = await fetch('/api/data');
    const data = await res.json();
    update(data);
  } catch(e) {
    document.getElementById('status').textContent = 'error: ' + e.message;
  }
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Dashboard: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
