"""Web UI for visualizing Stage 2 dialogue detection results.

Usage:
    uv run python scripts/stage2_viewer.py workspace/v2/dialogue_v2_unified.json
    # Open http://localhost:8501
"""

import argparse
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 2 — Dialogue Detection</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; }

.header { padding: 20px 24px; background: #1a1a1a; border-bottom: 1px solid #333; }
.header h1 { font-size: 18px; font-weight: 500; margin-bottom: 8px; }
.stats { display: flex; gap: 32px; font-size: 13px; color: #999; }
.stats .value { color: #4fc3f7; font-weight: 600; font-size: 15px; }

.controls { padding: 12px 24px; background: #151515; border-bottom: 1px solid #282828; display: flex; gap: 16px; align-items: center; font-size: 13px; }
.controls label { color: #888; }
.controls select, .controls input { background: #222; color: #ddd; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 13px; }

.timeline-container { padding: 24px; overflow-x: auto; }
.timeline-label { font-size: 11px; color: #666; margin-bottom: 4px; }

/* Full movie timeline */
.movie-timeline { position: relative; height: 40px; background: #1a1a1a; border-radius: 4px; margin-bottom: 24px; cursor: crosshair; min-width: 100%; }
.movie-timeline .seg { position: absolute; top: 0; height: 100%; border-radius: 2px; transition: opacity 0.1s; }
.movie-timeline .seg:hover { opacity: 0.8; }
.movie-timeline .seg.dialogue { background: #4caf50; }
.movie-timeline .seg.silent { background: #1a1a1a; }

/* Chunk grid */
.chunks { display: flex; flex-direction: column; gap: 2px; }
.chunk-row { display: flex; align-items: stretch; min-height: 28px; }
.chunk-label { width: 90px; flex-shrink: 0; font-size: 11px; color: #666; display: flex; align-items: center; padding-right: 8px; justify-content: flex-end; }
.chunk-bar { flex: 1; display: flex; position: relative; background: #111; border-radius: 2px; overflow: hidden; }
.chunk-bar .seg { height: 100%; transition: opacity 0.1s; }
.chunk-bar .seg:hover { opacity: 0.8; filter: brightness(1.3); }

/* RMS color scale */
.rms-legend { display: flex; align-items: center; gap: 8px; margin: 16px 24px; font-size: 11px; color: #666; }
.rms-gradient { width: 200px; height: 12px; border-radius: 2px; }

/* Tooltip */
.tooltip { position: fixed; background: #222; border: 1px solid #444; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; z-index: 100; max-width: 280px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
.tooltip .t-row { display: flex; justify-content: space-between; gap: 16px; }
.tooltip .t-label { color: #888; }
.tooltip .t-value { color: #fff; font-weight: 500; }
.tooltip .t-dialogue { color: #4caf50; }
.tooltip .t-silent { color: #666; }

/* Time axis */
.time-axis { position: relative; height: 20px; margin-top: 2px; }
.time-axis .tick { position: absolute; font-size: 10px; color: #555; transform: translateX(-50%); }

/* View range indicator */
.view-range { position: absolute; top: 0; height: 100%; border: 2px solid #4fc3f7; border-radius: 3px; pointer-events: none; opacity: 0.4; }
</style>
</head>
<body>

<div class="header">
  <h1 id="title">Stage 2 — Dialogue Detection</h1>
  <div class="stats">
    <div>Duration: <span class="value" id="duration">—</span></div>
    <div>Dialogue: <span class="value" id="dialogue-pct">—</span></div>
    <div>Dialogue time: <span class="value" id="dialogue-sec">—</span></div>
    <div>Chunks: <span class="value" id="num-chunks">—</span></div>
    <div>With dialogue: <span class="value" id="num-dialogue">—</span></div>
    <div>RMS threshold: <span class="value" id="rms-thresh">—</span></div>
  </div>
</div>

<div class="controls">
  <label>Color by:</label>
  <select id="color-mode">
    <option value="dialogue">Dialogue (yes/no)</option>
    <option value="rms">Target RMS (dB)</option>
    <option value="ratio">Target / Residual (dB)</option>
  </select>
  <label>Zoom:</label>
  <input id="zoom" type="range" min="1" max="10" step="0.5" value="1">
  <span id="zoom-label">1x</span>
</div>

<div class="timeline-container" id="timeline-container">
  <div class="timeline-label">Full movie timeline (click to scroll chunks)</div>
  <div class="movie-timeline" id="movie-timeline"></div>

  <div class="timeline-label" id="chunks-label">Chunks — 1-second segments</div>
  <div class="chunks" id="chunks"></div>

  <div class="time-axis" id="time-axis"></div>
</div>

<div class="rms-legend">
  <span>Silent</span>
  <canvas class="rms-gradient" id="rms-gradient" width="200" height="12"></canvas>
  <span>Loud</span>
</div>

<div class="tooltip" id="tooltip" style="display:none"></div>

<script>
let DATA = null;
let colorMode = 'dialogue';

function fmt(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h > 0 ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${m}:${String(s).padStart(2,'0')}`;
}

function rmsColor(rms) {
  // Map RMS from -80 (silent) to -20 (loud) to a color
  const t = Math.max(0, Math.min(1, (rms + 80) / 60));
  if (t < 0.3) return `rgba(40, 40, 40, 0.8)`;
  if (t < 0.5) return `hsl(${30 + t * 60}, 70%, ${20 + t * 40}%)`;
  if (t < 0.7) return `hsl(${60 + t * 40}, 80%, ${30 + t * 30}%)`;
  return `hsl(${100 + t * 20}, 80%, ${35 + t * 25}%)`;
}

function ratioColor(ratio) {
  // Map target-to-residual ratio. Positive = target louder (dialogue likely)
  const t = Math.max(0, Math.min(1, (ratio + 20) / 40));
  if (t < 0.4) return `rgba(40, 40, 40, 0.8)`;
  if (t < 0.5) return `hsl(0, 50%, ${30 + t * 30}%)`;
  return `hsl(${(t - 0.5) * 240}, 80%, ${30 + t * 30}%)`;
}

function segColor(seg) {
  if (colorMode === 'dialogue') return seg.has_dialogue ? '#4caf50' : '#1e1e1e';
  if (colorMode === 'rms') return rmsColor(seg.target_rms_db);
  if (colorMode === 'ratio') return ratioColor(seg.target_to_residual_db);
  return '#333';
}

function render() {
  if (!DATA) return;
  const d = DATA;
  const duration = d.source.duration;
  const chunks = d.chunks || [];
  const stats = d.statistics || {};
  const proc = d.processing || {};

  // Header stats
  document.getElementById('duration').textContent = fmt(duration);
  document.getElementById('dialogue-pct').textContent = `${stats.dialogue_percentage || 0}%`;
  document.getElementById('dialogue-sec').textContent = `${stats.total_dialogue_seconds || 0}s`;
  document.getElementById('num-chunks').textContent = chunks.length;
  document.getElementById('num-dialogue').textContent = chunks.filter(c => c.has_any_dialogue).length;
  document.getElementById('rms-thresh').textContent = `${proc.rms_threshold_db || -40} dB`;
  document.getElementById('title').textContent = `Stage 2 — ${d.source.file || 'unknown'}`;

  const zoom = parseFloat(document.getElementById('zoom').value);
  document.getElementById('zoom-label').textContent = `${zoom}x`;

  // Movie timeline
  const mt = document.getElementById('movie-timeline');
  mt.innerHTML = '';
  mt.style.width = `${zoom * 100}%`;
  const allSegs = [];
  for (const chunk of chunks) {
    for (const seg of chunk.segments) {
      allSegs.push(seg);
    }
  }
  for (const seg of allSegs) {
    const el = document.createElement('div');
    el.className = `seg ${seg.has_dialogue ? 'dialogue' : 'silent'}`;
    el.style.left = `${(seg.start_time / duration) * 100}%`;
    el.style.width = `${((seg.end_time - seg.start_time) / duration) * 100}%`;
    el.style.background = segColor(seg);
    mt.appendChild(el);
  }

  // Chunk bars
  const chunksEl = document.getElementById('chunks');
  chunksEl.innerHTML = '';
  for (const chunk of chunks) {
    const row = document.createElement('div');
    row.className = 'chunk-row';

    const label = document.createElement('div');
    label.className = 'chunk-label';
    label.textContent = `${fmt(chunk.start_time)}`;
    row.appendChild(label);

    const bar = document.createElement('div');
    bar.className = 'chunk-bar';
    bar.style.width = `${zoom * 100}%`;

    const chunkDur = chunk.end_time - chunk.start_time;
    for (const seg of chunk.segments) {
      const segEl = document.createElement('div');
      segEl.className = 'seg';
      const segDur = seg.end_time - seg.start_time;
      segEl.style.width = `${(segDur / chunkDur) * 100}%`;
      segEl.style.background = segColor(seg);
      segEl.dataset.info = JSON.stringify(seg);
      segEl.dataset.chunkIndex = chunk.chunk_index;
      bar.appendChild(segEl);
    }
    row.appendChild(bar);
    chunksEl.appendChild(row);
  }

  // Time axis
  renderTimeAxis(duration, zoom);

  // RMS gradient legend
  renderGradient();
}

function renderTimeAxis(duration, zoom) {
  const axis = document.getElementById('time-axis');
  axis.innerHTML = '';
  axis.style.width = `${zoom * 100}%`;
  axis.style.marginLeft = '90px';

  // Choose tick interval based on duration and zoom
  let interval = 600; // 10 min
  if (duration / zoom < 600) interval = 60;
  if (duration / zoom < 120) interval = 30;
  if (duration * zoom > 3600) interval = 600;

  for (let t = 0; t <= duration; t += interval) {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `${(t / duration) * 100}%`;
    tick.textContent = fmt(t);
    axis.appendChild(tick);
  }
}

function renderGradient() {
  const canvas = document.getElementById('rms-gradient');
  const ctx = canvas.getContext('2d');
  for (let x = 0; x < 200; x++) {
    const t = x / 200;
    if (colorMode === 'dialogue') {
      ctx.fillStyle = t < 0.5 ? '#1e1e1e' : '#4caf50';
    } else if (colorMode === 'rms') {
      ctx.fillStyle = rmsColor(-80 + t * 60);
    } else {
      ctx.fillStyle = ratioColor(-20 + t * 40);
    }
    ctx.fillRect(x, 0, 1, 12);
  }
}

// Tooltip
const tooltip = document.getElementById('tooltip');
document.addEventListener('mouseover', e => {
  const seg = e.target.closest('.seg[data-info]');
  if (!seg) { tooltip.style.display = 'none'; return; }
  const info = JSON.parse(seg.dataset.info);
  tooltip.style.display = 'block';
  tooltip.innerHTML = `
    <div class="t-row"><span class="t-label">Time</span><span class="t-value">${fmt(info.start_time)} — ${fmt(info.end_time)}</span></div>
    <div class="t-row"><span class="t-label">Dialogue</span><span class="${info.has_dialogue ? 't-dialogue' : 't-silent'}">${info.has_dialogue ? 'Yes' : 'No'}</span></div>
    <div class="t-row"><span class="t-label">Target RMS</span><span class="t-value">${info.target_rms_db} dB</span></div>
    <div class="t-row"><span class="t-label">Target/Residual</span><span class="t-value">${info.target_to_residual_db > 0 ? '+' : ''}${info.target_to_residual_db} dB</span></div>
    <div class="t-row"><span class="t-label">Chunk</span><span class="t-value">#${seg.dataset.chunkIndex}</span></div>
  `;
});
document.addEventListener('mousemove', e => {
  if (tooltip.style.display === 'none') return;
  const x = Math.min(e.clientX + 12, window.innerWidth - 300);
  const y = Math.min(e.clientY + 12, window.innerHeight - 120);
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
});
document.addEventListener('mouseout', e => {
  if (e.target.closest('.seg[data-info]')) return;
  tooltip.style.display = 'none';
});

// Controls
document.getElementById('color-mode').addEventListener('change', e => { colorMode = e.target.value; render(); });
document.getElementById('zoom').addEventListener('input', () => render());

// Load data
fetch('/api/data').then(r => r.json()).then(d => { DATA = d; render(); });
</script>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    data = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.data).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress access logs


def main():
    parser = argparse.ArgumentParser(description="Stage 2 dialogue detection viewer")
    parser.add_argument("json_file", help="Path to pipeline output JSON")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        ViewerHandler.data = json.load(f)

    server = HTTPServer((args.host, args.port), ViewerHandler)
    print(f"Serving {json_path.name} at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
