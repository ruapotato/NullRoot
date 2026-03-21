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
    if not os.path.exists(LOG_PATH):
        return train, evals
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") in ("eval", "final_eval"):
                evals.append(entry)
            elif "loss" in entry:
                train.append(entry)
    return train, evals


@app.route("/api/data")
def api_data():
    train, evals = read_log()
    return {"train": train, "evals": evals}


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
  .eval-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .eval-table th { text-align: left; color: #58a6ff; padding: 6px 10px; border-bottom: 1px solid #30363d; }
  .eval-table td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
  .full { grid-column: 1 / -1; }
  #status { color: #3fb950; font-size: 12px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>

<h1>NullRoot Training Dashboard</h1>
<p class="subtitle">~20M param transformer learning to simulate bash &nbsp; <span id="status">connecting...</span></p>

<div class="stats">
  <div class="stat"><div class="value" id="s-step">-</div><div class="label">Step</div></div>
  <div class="stat"><div class="value" id="s-loss">-</div><div class="label">Train Loss</div></div>
  <div class="stat"><div class="value" id="s-ppl">-</div><div class="label">Perplexity</div></div>
  <div class="stat"><div class="value" id="s-speed">-</div><div class="label">Tokens/sec</div></div>
</div>
<div class="stats">
  <div class="stat"><div class="value" id="s-eta">-</div><div class="label">ETA</div></div>
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
    <h2>Validation Results</h2>
    <table class="eval-table" id="evalTable">
      <thead><tr><th>Step</th><th>Val Loss</th><th>Val PPL</th><th>Tokens</th><th>Time</th></tr></thead>
      <tbody id="evalBody"><tr><td colspan="5" style="color:#8b949e">No evaluations yet</td></tr></tbody>
    </table>
  </div>
</div>

<script>
const chartOpts = (label, color) => ({
  type: 'line',
  data: { labels: [], datasets: [{ label, data: [], borderColor: color, backgroundColor: color + '22', fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.3 }] },
  options: {
    responsive: true, animation: false,
    scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' } }, y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } } },
    plugins: { legend: { display: false } }
  }
});

const lossChart = new Chart(document.getElementById('lossChart'), chartOpts('Loss', '#f85149'));
const pplChart  = new Chart(document.getElementById('pplChart'),  chartOpts('PPL',  '#d29922'));
const lrChart   = new Chart(document.getElementById('lrChart'),   chartOpts('LR',   '#3fb950'));
const speedChart= new Chart(document.getElementById('speedChart'),chartOpts('tok/s','#58a6ff'));

function update(data) {
  const t = data.train;
  if (!t.length) return;

  const steps = t.map(e => e.step);
  const losses = t.map(e => e.loss);
  const ppls = t.map(e => e.ppl);
  const lrs = t.map(e => e.lr);
  const speeds = t.map(e => e.tokens_per_sec);

  function setChart(chart, labels, values) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update();
  }
  setChart(lossChart, steps, losses);
  setChart(pplChart, steps, ppls);
  setChart(lrChart, steps, lrs);
  setChart(speedChart, steps, speeds);

  const last = t[t.length - 1];
  document.getElementById('s-step').textContent = last.step.toLocaleString();
  document.getElementById('s-loss').textContent = last.loss.toFixed(4);
  document.getElementById('s-ppl').textContent = last.ppl.toFixed(1);
  document.getElementById('s-speed').textContent = last.tokens_per_sec.toLocaleString();
  document.getElementById('s-lr').textContent = last.lr.toExponential(2);
  document.getElementById('s-gnorm').textContent = last.grad_norm.toFixed(2);
  document.getElementById('s-gpu').textContent = (last.gpu_peak_gb || 0).toFixed(1);

  const eta_s = last.eta_sec || 0;
  const hrs = Math.floor(eta_s / 3600);
  const mins = Math.floor((eta_s % 3600) / 60);
  document.getElementById('s-eta').textContent = hrs + 'h ' + mins + 'm';

  // Eval table
  const evals = data.evals;
  const tbody = document.getElementById('evalBody');
  if (evals.length) {
    tbody.innerHTML = evals.map(e =>
      '<tr><td>' + e.step + '</td><td>' + (e.val_loss||0).toFixed(4) + '</td><td>' +
      (e.val_ppl||0).toFixed(2) + '</td><td>' + (e.val_tokens||0).toLocaleString() +
      '</td><td>' + (e.timestamp||'') + '</td></tr>'
    ).join('');
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
