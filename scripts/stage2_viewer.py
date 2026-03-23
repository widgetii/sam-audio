"""Web UI for visualizing dialogue detection and screen time results.

Usage:
    uv run python scripts/stage2_viewer.py workspace/v2/dialogue_v2_unified.json \
        --screen-time workspace/v2/screen_time.json
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
<title>Pipeline Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; }

.header { padding: 20px 24px; background: #1a1a1a; border-bottom: 1px solid #333; }
.header h1 { font-size: 18px; font-weight: 500; margin-bottom: 8px; }
.stats { display: flex; gap: 32px; font-size: 13px; color: #999; flex-wrap: wrap; }
.stats .value { color: #4fc3f7; font-weight: 600; font-size: 15px; }

.tabs { display: flex; border-bottom: 1px solid #333; background: #151515; }
.tab { padding: 10px 24px; font-size: 13px; color: #888; cursor: pointer; border-bottom: 2px solid transparent; }
.tab:hover { color: #ccc; }
.tab.active { color: #4fc3f7; border-bottom-color: #4fc3f7; }

.tab-content { display: none; }
.tab-content.active { display: block; }

.controls { padding: 12px 24px; background: #151515; border-bottom: 1px solid #282828; display: flex; gap: 16px; align-items: center; font-size: 13px; }
.controls label { color: #888; }
.controls select, .controls input { background: #222; color: #ddd; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 13px; }

.section { padding: 24px; }
.section-title { font-size: 14px; color: #888; margin-bottom: 12px; }

/* Full movie timeline */
.movie-timeline { position: relative; height: 40px; background: #1a1a1a; border-radius: 4px; margin-bottom: 16px; cursor: crosshair; min-width: 100%; }
.movie-timeline .seg { position: absolute; top: 0; height: 100%; border-radius: 2px; transition: opacity 0.1s; }
.movie-timeline .seg:hover { opacity: 0.8; }

/* Chunk grid */
.chunks { display: flex; flex-direction: column; gap: 2px; }
.chunk-row { display: flex; align-items: stretch; min-height: 28px; }
.chunk-label { width: 90px; flex-shrink: 0; font-size: 11px; color: #666; display: flex; align-items: center; padding-right: 8px; justify-content: flex-end; }
.chunk-bar { flex: 1; display: flex; position: relative; background: #111; border-radius: 2px; overflow: hidden; }
.chunk-bar .seg { height: 100%; transition: opacity 0.1s; }
.chunk-bar .seg:hover { opacity: 0.8; filter: brightness(1.3); }

/* RMS legend */
.rms-legend { display: flex; align-items: center; gap: 8px; margin: 16px 0; font-size: 11px; color: #666; }
.rms-gradient { width: 200px; height: 12px; border-radius: 2px; }

/* Tooltip */
.tooltip { position: fixed; background: #222; border: 1px solid #444; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; z-index: 100; max-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
.tooltip .t-row { display: flex; justify-content: space-between; gap: 16px; }
.tooltip .t-label { color: #888; }
.tooltip .t-value { color: #fff; font-weight: 500; }
.tooltip .t-dialogue { color: #4caf50; }
.tooltip .t-silent { color: #666; }

/* Time axis */
.time-axis { position: relative; height: 20px; margin-top: 2px; }
.time-axis .tick { position: absolute; font-size: 10px; color: #555; transform: translateX(-50%); }

/* Screen time */
.char-colors { --c0: #e6194b; --c1: #3cb44b; --c2: #4363d8; --c3: #f58231; --c4: #911eb4; --c5: #42d4f4; --c6: #f032e6; --c7: #bfef45; }

.screen-time-bars { display: flex; flex-direction: column; gap: 8px; max-width: 600px; }
.st-row { display: flex; align-items: center; gap: 12px; }
.st-label { width: 80px; font-size: 13px; color: #aaa; text-align: right; }
.st-bar-bg { flex: 1; height: 24px; background: #1a1a1a; border-radius: 4px; overflow: hidden; position: relative; }
.st-bar { height: 100%; border-radius: 4px; transition: width 0.3s; }
.st-value { font-size: 12px; color: #888; width: 100px; }

/* Character presence timeline */
.char-timeline { position: relative; height: 20px; background: #111; border-radius: 3px; margin-bottom: 4px; }
.char-timeline .pres { position: absolute; top: 2px; height: 16px; border-radius: 2px; opacity: 0.8; }

/* Co-occurrence */
.cooccurrence { border-collapse: collapse; font-size: 12px; }
.cooccurrence th, .cooccurrence td { padding: 6px 10px; text-align: center; border: 1px solid #333; }
.cooccurrence th { color: #888; background: #1a1a1a; }
.cooccurrence td { background: #111; }
</style>
</head>
<body>

<div class="header">
  <h1 id="title">Pipeline Viewer</h1>
  <div class="stats">
    <div>Duration: <span class="value" id="duration">—</span></div>
    <div>Dialogue: <span class="value" id="dialogue-pct">—</span></div>
    <div>Dialogue time: <span class="value" id="dialogue-sec">—</span></div>
    <div>Characters: <span class="value" id="num-chars">—</span></div>
  </div>
</div>

<div class="tabs">
  <div class="tab active" data-tab="dialogue">Dialogue Detection</div>
  <div class="tab" data-tab="screentime">Screen Time</div>
</div>

<!-- DIALOGUE TAB -->
<div class="tab-content active" id="tab-dialogue">
  <div class="controls">
    <label>Color by:</label>
    <select id="color-mode">
      <option value="dialogue">Dialogue (yes/no)</option>
      <option value="rms">Target RMS (dB)</option>
      <option value="ratio">Target / Residual (dB)</option>
      <option value="center">Center Channel (dB)</option>
    </select>
    <label>Zoom:</label>
    <input id="zoom" type="range" min="1" max="10" step="0.5" value="1">
    <span id="zoom-label">1x</span>
  </div>

  <div class="section" id="dialogue-section">
    <div class="section-title">Full movie timeline</div>
    <div class="movie-timeline" id="movie-timeline"></div>

    <div class="section-title">Chunks — 1-second segments</div>
    <div class="chunks" id="chunks"></div>
    <div class="time-axis" id="time-axis"></div>

    <div class="rms-legend">
      <span>Silent</span>
      <canvas class="rms-gradient" id="rms-gradient" width="200" height="12"></canvas>
      <span>Loud</span>
    </div>
  </div>
</div>

<!-- SCREEN TIME TAB -->
<div class="tab-content" id="tab-screentime">
  <div class="section char-colors">
    <div class="section-title">Screen Time per Character</div>
    <div class="screen-time-bars" id="st-bars"></div>

    <div class="section-title" style="margin-top: 32px;">Character Presence Timeline</div>
    <div id="char-timelines"></div>
    <div class="time-axis" id="st-time-axis" style="margin-left: 90px;"></div>

    <div class="section-title" style="margin-top: 32px;">Co-occurrence (shared screen time in seconds)</div>
    <div id="cooccurrence-container"></div>

    <div id="no-screentime" style="padding: 40px; color: #666; font-size: 14px; display: none;">
      No screen time data. Run: <code>python scripts/screen_time.py --input VIDEO --pipeline-json PIPELINE.json --output screen_time.json</code>
    </div>
  </div>
</div>

<div class="tooltip" id="tooltip" style="display:none"></div>

<script>
let DATA = null;
let ST_DATA = null;
let colorMode = 'dialogue';

const CHAR_COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#aaffc3','#dcbeff'];

function fmt(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return h > 0 ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${m}:${String(s).padStart(2,'0')}`;
}

function rmsColor(rms) {
  const t = Math.max(0, Math.min(1, (rms + 80) / 60));
  if (t < 0.3) return 'rgba(40,40,40,0.8)';
  if (t < 0.5) return `hsl(${30+t*60},70%,${20+t*40}%)`;
  if (t < 0.7) return `hsl(${60+t*40},80%,${30+t*30}%)`;
  return `hsl(${100+t*20},80%,${35+t*25}%)`;
}
function ratioColor(ratio) {
  const t = Math.max(0, Math.min(1, (ratio + 20) / 40));
  if (t < 0.4) return 'rgba(40,40,40,0.8)';
  if (t < 0.5) return `hsl(0,50%,${30+t*30}%)`;
  return `hsl(${(t-0.5)*240},80%,${30+t*30}%)`;
}
function segColor(seg) {
  if (colorMode === 'dialogue') return seg.has_dialogue ? '#4caf50' : '#1e1e1e';
  if (colorMode === 'rms') return rmsColor(seg.target_rms_db);
  if (colorMode === 'ratio') return ratioColor(seg.target_to_residual_db);
  if (colorMode === 'center') return seg.center_rms_db != null ? rmsColor(seg.center_rms_db) : '#1e1e1e';
  return '#333';
}

function renderDialogue() {
  if (!DATA) return;
  const d = DATA;
  const duration = d.source.duration;
  const chunks = d.chunks || [];
  const stats = d.statistics || {};
  const proc = d.processing || {};

  document.getElementById('duration').textContent = fmt(duration);
  document.getElementById('dialogue-pct').textContent = `${stats.dialogue_percentage || 0}%`;
  document.getElementById('dialogue-sec').textContent = `${stats.total_dialogue_seconds || 0}s`;
  document.getElementById('num-chars').textContent = (d.characters || []).length;
  document.getElementById('title').textContent = d.source.file || 'Pipeline Viewer';

  const zoom = parseFloat(document.getElementById('zoom').value);
  document.getElementById('zoom-label').textContent = `${zoom}x`;

  // Movie timeline
  const mt = document.getElementById('movie-timeline');
  mt.innerHTML = '';
  mt.style.width = `${zoom * 100}%`;
  for (const chunk of chunks) {
    for (const seg of chunk.segments) {
      const el = document.createElement('div');
      el.className = 'seg';
      el.style.left = `${(seg.start_time / duration) * 100}%`;
      el.style.width = `${((seg.end_time - seg.start_time) / duration) * 100}%`;
      el.style.background = segColor(seg);
      mt.appendChild(el);
    }
  }

  // Chunk bars
  const chunksEl = document.getElementById('chunks');
  chunksEl.innerHTML = '';
  for (const chunk of chunks) {
    const row = document.createElement('div');
    row.className = 'chunk-row';
    const label = document.createElement('div');
    label.className = 'chunk-label';
    label.textContent = fmt(chunk.start_time);
    row.appendChild(label);
    const bar = document.createElement('div');
    bar.className = 'chunk-bar';
    bar.style.width = `${zoom * 100}%`;
    const chunkDur = chunk.end_time - chunk.start_time;
    for (const seg of chunk.segments) {
      const segEl = document.createElement('div');
      segEl.className = 'seg';
      segEl.style.width = `${((seg.end_time - seg.start_time) / chunkDur) * 100}%`;
      segEl.style.background = segColor(seg);
      segEl.dataset.info = JSON.stringify(seg);
      segEl.dataset.chunkIndex = chunk.chunk_index;
      bar.appendChild(segEl);
    }
    row.appendChild(bar);
    chunksEl.appendChild(row);
  }

  // Time axis
  const axis = document.getElementById('time-axis');
  axis.innerHTML = '';
  axis.style.width = `${zoom * 100}%`;
  axis.style.marginLeft = '90px';
  let interval = 600;
  if (duration / zoom < 600) interval = 60;
  if (duration * zoom > 3600) interval = 600;
  for (let t = 0; t <= duration; t += interval) {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `${(t / duration) * 100}%`;
    tick.textContent = fmt(t);
    axis.appendChild(tick);
  }

  // Gradient
  const canvas = document.getElementById('rms-gradient');
  const ctx = canvas.getContext('2d');
  for (let x = 0; x < 200; x++) {
    const t = x / 200;
    if (colorMode === 'dialogue') ctx.fillStyle = t < 0.5 ? '#1e1e1e' : '#4caf50';
    else if (colorMode === 'rms' || colorMode === 'center') ctx.fillStyle = rmsColor(-80 + t * 60);
    else ctx.fillStyle = ratioColor(-20 + t * 40);
    ctx.fillRect(x, 0, 1, 12);
  }
}

function renderScreenTime() {
  if (!ST_DATA) {
    document.getElementById('no-screentime').style.display = 'block';
    return;
  }
  document.getElementById('no-screentime').style.display = 'none';
  const st = ST_DATA;
  const duration = st.source.duration;
  const chars = st.characters || [];
  const maxTime = Math.max(...chars.map(c => c.screen_time_sec), 1);

  // Bar chart
  const barsEl = document.getElementById('st-bars');
  barsEl.innerHTML = '';
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    const row = document.createElement('div');
    row.className = 'st-row';
    row.innerHTML = `
      <div class="st-label">char ${c.id}</div>
      <div class="st-bar-bg">
        <div class="st-bar" style="width: ${(c.screen_time_sec / maxTime) * 100}%; background: ${CHAR_COLORS[i % CHAR_COLORS.length]};"></div>
      </div>
      <div class="st-value">${(c.screen_time_sec / 60).toFixed(1)} min (${c.screen_time_pct}%)</div>
    `;
    barsEl.appendChild(row);
  }

  // Presence timelines
  const tlEl = document.getElementById('char-timelines');
  tlEl.innerHTML = '';
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    const color = CHAR_COLORS[i % CHAR_COLORS.length];
    const row = document.createElement('div');
    row.className = 'chunk-row';
    const label = document.createElement('div');
    label.className = 'chunk-label';
    label.innerHTML = `<span style="color:${color}">&#9632;</span> char ${c.id}`;
    row.appendChild(label);
    const tl = document.createElement('div');
    tl.className = 'char-timeline';
    tl.style.flex = '1';
    for (const seg of c.presence_segments) {
      const el = document.createElement('div');
      el.className = 'pres';
      el.style.left = `${(seg[0] / duration) * 100}%`;
      el.style.width = `${((seg[1] - seg[0]) / duration) * 100}%`;
      el.style.background = color;
      tl.appendChild(el);
    }
    row.appendChild(tl);
    tlEl.appendChild(row);
  }

  // Time axis for screen time
  const axis = document.getElementById('st-time-axis');
  axis.innerHTML = '';
  let interval = 600;
  if (duration < 600) interval = 60;
  for (let t = 0; t <= duration; t += interval) {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `${(t / duration) * 100}%`;
    tick.textContent = fmt(t);
    axis.appendChild(tick);
  }

  // Co-occurrence matrix
  const cooc = st.cooccurrence;
  if (!cooc) return;
  const ids = cooc.character_ids;
  const matrix = cooc.matrix;
  const container = document.getElementById('cooccurrence-container');
  let html = '<table class="cooccurrence"><tr><th></th>';
  for (let i = 0; i < ids.length; i++) html += `<th style="color:${CHAR_COLORS[i]}">c${ids[i]}</th>`;
  html += '</tr>';
  for (let i = 0; i < ids.length; i++) {
    html += `<tr><th style="color:${CHAR_COLORS[i]}">char ${ids[i]}</th>`;
    for (let j = 0; j < ids.length; j++) {
      const val = matrix[i][j] * (st.sample_interval || 5);
      const maxVal = matrix[i][i] * (st.sample_interval || 5);
      const intensity = Math.min(val / Math.max(maxVal, 1), 1);
      const bg = i === j ? `rgba(${CHAR_COLORS[i].replace('#','').match(/../g).map(h=>parseInt(h,16)).join(',')},0.3)` : `rgba(255,255,255,${intensity * 0.15})`;
      html += `<td style="background:${bg}">${Math.round(val / 60)}m</td>`;
    }
    html += '</tr>';
  }
  html += '</table>';
  container.innerHTML = html;
}

// Tooltip
const tooltip = document.getElementById('tooltip');
document.addEventListener('mouseover', e => {
  const seg = e.target.closest('.seg[data-info]');
  if (!seg) { tooltip.style.display = 'none'; return; }
  const info = JSON.parse(seg.dataset.info);
  let centerRow = '';
  if (info.center_rms_db != null) {
    centerRow = `<div class="t-row"><span class="t-label">Center RMS</span><span class="t-value">${info.center_rms_db} dB</span></div>`;
  }
  tooltip.style.display = 'block';
  tooltip.innerHTML = `
    <div class="t-row"><span class="t-label">Time</span><span class="t-value">${fmt(info.start_time)} — ${fmt(info.end_time)}</span></div>
    <div class="t-row"><span class="t-label">Dialogue</span><span class="${info.has_dialogue ? 't-dialogue' : 't-silent'}">${info.has_dialogue ? 'Yes' : 'No'}</span></div>
    <div class="t-row"><span class="t-label">Target RMS</span><span class="t-value">${info.target_rms_db} dB</span></div>
    <div class="t-row"><span class="t-label">Target/Residual</span><span class="t-value">${info.target_to_residual_db > 0 ? '+' : ''}${info.target_to_residual_db} dB</span></div>
    ${centerRow}
    <div class="t-row"><span class="t-label">Chunk</span><span class="t-value">#${seg.dataset.chunkIndex}</span></div>
  `;
});
document.addEventListener('mousemove', e => {
  if (tooltip.style.display === 'none') return;
  tooltip.style.left = Math.min(e.clientX + 12, window.innerWidth - 320) + 'px';
  tooltip.style.top = Math.min(e.clientY + 12, window.innerHeight - 140) + 'px';
});
document.addEventListener('mouseout', e => {
  if (e.target.closest('.seg[data-info]')) return;
  tooltip.style.display = 'none';
});

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

// Controls
document.getElementById('color-mode').addEventListener('change', e => { colorMode = e.target.value; renderDialogue(); });
document.getElementById('zoom').addEventListener('input', () => renderDialogue());

// Load data
Promise.all([
  fetch('/api/data').then(r => r.json()),
  fetch('/api/screen_time').then(r => r.ok ? r.json() : null),
]).then(([d, st]) => {
  DATA = d;
  ST_DATA = st;
  renderDialogue();
  renderScreenTime();
});
</script>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    data = None
    screen_time_data = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.data).encode())
        elif parsed.path == "/api/screen_time":
            if self.screen_time_data:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(self.screen_time_data).encode())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Pipeline results viewer")
    parser.add_argument("json_file", help="Path to pipeline output JSON")
    parser.add_argument("--screen-time", default=None, help="Path to screen_time.json")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        ViewerHandler.data = json.load(f)

    if args.screen_time:
        st_path = Path(args.screen_time)
        if st_path.exists():
            with open(st_path) as f:
                ViewerHandler.screen_time_data = json.load(f)
            print(f"Loaded screen time from {st_path.name}")
        else:
            print(f"Warning: {st_path} not found, screen time tab disabled")
    else:
        # Auto-detect in same directory
        auto = json_path.parent / "screen_time.json"
        if auto.exists():
            with open(auto) as f:
                ViewerHandler.screen_time_data = json.load(f)
            print(f"Auto-loaded screen time from {auto.name}")

    server = HTTPServer((args.host, args.port), ViewerHandler)
    print(f"Serving at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
