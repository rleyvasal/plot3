"""Embedded three.js viewer HTML template."""

from __future__ import annotations

_DOC_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}</script>
<style>
html,body{margin:0;height:100%;overflow:hidden;
  font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
#fig{position:relative;width:100vw;height:100vh}
#title{position:absolute;left:14px;top:8px;font-size:14px;font-weight:600;z-index:4}
#canvas-host{position:absolute}
#axes{position:absolute;inset:0;pointer-events:none;z-index:2}
#legend{position:absolute;right:10px;top:8px;z-index:4;padding:6px 9px;
  border-radius:6px;font-size:11px;line-height:1.7}
#legend .sw{display:inline-block;width:9px;height:9px;border-radius:5px;
  margin-right:6px;vertical-align:-1px}
#legend .lg-e{cursor:pointer;user-select:none}
#tip{position:absolute;display:none;z-index:5;pointer-events:none;
  padding:4px 8px;border-radius:5px;font-size:11px;white-space:nowrap}
#hint{position:absolute;left:50%;bottom:46px;transform:translateX(-50%);
  z-index:5;pointer-events:none;padding:5px 10px;border-radius:5px;
  font-size:11px;opacity:0;transition:opacity .25s}
#ramp{height:8px;width:110px;border-radius:4px;margin-top:3px}
</style></head><body>
<div id="fig">
  <div id="title"></div>
  <div id="canvas-host"></div>
  <svg id="axes"></svg>
  <div id="legend" style="display:none"></div>
  <div id="tip"></div>
  <div id="hint"></div>
</div>
__PAYLOADS__
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

const S = __SPEC__;
const T = S.theme;
document.body.style.background = T.surface;
document.body.style.color = T.ink;

async function decode(id, dtype) {
  const node = document.getElementById(id);
  if (!node) return null;
  const s = atob(node.textContent.trim());
  let a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  if (S.gz) {
    const ds = new DecompressionStream('gzip');
    a = new Uint8Array(
      await new Response(new Blob([a]).stream().pipeThrough(ds)).arrayBuffer());
  }
  if (dtype === 'f32') return new Float32Array(a.buffer);
  let u;
  if (S.gz) {                       // undo byte planes + delta
    const m = a.length >> 1;
    u = new Uint16Array(m);
    for (let i = 0; i < m; i++) u[i] = a[i] | (a[m + i] << 8);
    for (let i = 1; i < m; i++) u[i] = (u[i] + u[i - 1]) & 0xffff;
  } else {
    u = new Uint16Array(a.buffer);
  }
  return u;
}
function toNorm(arr) {                       // u16 -> [0,1] f32 (f32 passes through)
  if (arr instanceof Float32Array) return arr;
  const f = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) f[i] = arr[i] / 65535;
  return f;
}
function hex2rgb(h) {
  return [parseInt(h.slice(1,3),16)/255, parseInt(h.slice(3,5),16)/255,
          parseInt(h.slice(5,7),16)/255];
}
const RAMP = (S.color.ramp || []).map(hex2rgb);
function rampAt(t) {
  const k = Math.min(RAMP.length - 1.001, Math.max(0, t * (RAMP.length - 1)));
  const i = Math.floor(k), f = k - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
const PAL = (S.color.palette || []).map(hex2rgb);

// ── payload decode for every layer ──────────────────────────────────────────
const axesList = S.is3d ? ['x','y','z'] : ['x','y'];
for (const L of S.layers) {
  for (const a of axesList) L[a].data = toNorm(await decode(L[a].id, L[a].dtype));
  if (L.color) L.color.data = await decode(L.color.id, 'u16');
}

// per-layer vertex colors (normalized cube space is built per-branch)
function layerColors(L, defRGB) {
  const n = L.n, c = new Float32Array(n * 3);
  if (L.constColor) defRGB = hex2rgb(L.constColor);
  if (!L.color) { for (let i=0;i<n;i++){c[i*3]=defRGB[0];c[i*3+1]=defRGB[1];c[i*3+2]=defRGB[2];} return c; }
  const d = L.color.data;
  if (L.color.kind === 'cat') {
    for (let i = 0; i < n; i++) { const p = PAL[d[i] % PAL.length];
      c[i*3]=p[0]; c[i*3+1]=p[1]; c[i*3+2]=p[2]; }
  } else {
    for (let i = 0; i < n; i++) { const p = rampAt(d[i] / 65535);
      c[i*3]=p[0]; c[i*3+1]=p[1]; c[i*3+2]=p[2]; }
  }
  return c;
}

// ── chrome: title + legend ──────────────────────────────────────────────────
const figEl = document.getElementById('fig');
const titleEl = document.getElementById('title');
if (S.labs.title) titleEl.textContent = S.labs.title;
const legEl = document.getElementById('legend');
// legend click-filtering: category index -> three.js objects
const hiddenCats = new Set();
const catObjs = new Map();
function regCat(ci, obj) {
  if (!catObjs.has(ci)) catObjs.set(ci, []);
  catObjs.get(ci).push(obj);
}
let redraw = () => {};   // 2D assigns its draw(); 3D renders continuously
window.__plot3 = { hiddenCats, catObjs };
if (S.legend) {
  legEl.style.display = 'block';
  legEl.style.background = T.surface + 'e6';
  legEl.style.border = '1px solid ' + T.grid;
  legEl.style.color = T.ink2;
  legEl.innerHTML = (S.labs.color ? '<b style="color:'+T.ink+'">' +
      S.labs.color + '</b>' : '') +
    S.legend.map((e, i) => '<div class="lg-e" data-ci="' + i +
      '"><span class="sw" style="background:' + e.color + '"></span>' +
      e.label + '</div>').join('');
  legEl.addEventListener('click', ev => {
    const row = ev.target.closest('.lg-e');
    if (!row) return;
    const ci = +row.dataset.ci;
    if (hiddenCats.has(ci)) hiddenCats.delete(ci); else hiddenCats.add(ci);
    row.style.opacity = hiddenCats.has(ci) ? 0.35 : 1;
    for (const o of (catObjs.get(ci) || [])) o.visible = !hiddenCats.has(ci);
    tip.style.display = 'none';
    redraw();
  });
} else if (S.color.kind === 'num') {
  legEl.style.display = 'block';
  legEl.style.background = T.surface + 'e6';
  legEl.style.border = '1px solid ' + T.grid;
  legEl.style.color = T.ink2;
  legEl.innerHTML = '<b style="color:'+T.ink+'">' + (S.labs.color||'') +
    '</b><div id="ramp" style="background:linear-gradient(90deg,' +
    S.color.ramp.join(',') + ')"></div>' +
    '<span style="float:left">' + (+S.color.lo.toPrecision(3)) + '</span>' +
    '<span style="float:right">' + (+S.color.hi.toPrecision(3)) + '</span>';
}

function fmt(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toPrecision(3);
  return String(+v.toFixed(6));
}
// numeric colour: normalized ramp position -> data value (inverse transform)
function cval(t) {
  const tr = S.color.trans || 'linear';
  const f = tr === 'sqrt' ? Math.sqrt
    : tr === 'log10' ? (v => Math.log10(Math.max(v, 1e-12))) : (v => v);
  const inv = tr === 'sqrt' ? (v => v * v)
    : tr === 'log10' ? (v => Math.pow(10, v)) : (v => v);
  return inv(f(S.color.lo) + t * (f(S.color.hi) - f(S.color.lo)));
}
function fmtAxis(ax, v) {                     // v in data units
  const sc = S.scales[ax];
  if (sc.kind === 'cat') {
    const i = Math.round(v);
    return (i >= 0 && i < sc.cats.length && Math.abs(v - i) < 0.26) ? sc.cats[i] : '';
  }
  if (sc.kind === 'dt') return new Date(v * 1000).toLocaleString();
  return fmt(v);
}

// nice numeric ticks (JS side for pan/zoom)
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = 10 * mag;
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) { step = m * mag; break; }
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step)
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return out;
}
// ticks for any scale over a visible data range -> [[pos_data, label], ...]
function thin(vis, maxN) {
  if (vis.length <= maxN) return vis;
  const step = Math.ceil(vis.length / maxN);
  return vis.filter((_, i) => i % step === 0);
}
function ticksFor(ax, lo, hi) {
  const sc = S.scales[ax];
  if (sc.kind === 'cat')
    return thin(sc.cats.map((c, i) => [i, c])
      .filter(t => t[0] >= lo && t[0] <= hi), 12);
  if (sc.kind === 'dt') {
    for (const level of sc.ladder) {
      const vis = level.filter(t => t[0] >= lo && t[0] <= hi);
      if (vis.length >= 3 && vis.length <= 14) return vis;
    }
    let vis = sc.ladder[0].filter(t => t[0] >= lo && t[0] <= hi);
    if (vis.length < 3)
      vis = sc.ladder[sc.ladder.length - 1]
        .filter(t => t[0] >= lo && t[0] <= hi);
    return thin(vis, 10);
  }
  return niceTicks(lo, hi, 6).map(t => [t, fmt(t)]);
}
// hover value: enough digits to resolve ~1/300 of the visible span
function fmtSpan(ax, v, spanData) {
  const sc = S.scales[ax];
  if (sc.kind !== 'num') return fmtAxis(ax, v);
  const d = Math.max(0, Math.min(6,
    Math.ceil(-Math.log10(Math.max(1e-12, spanData / 300)))));
  return v.toFixed(d);
}
const dataLo = ax => S.scales[ax].lo, dataHi = ax => S.scales[ax].hi;
const spanOf = ax => (dataHi(ax) - dataLo(ax)) || 1;

const host = document.getElementById('canvas-host');
const svg = document.getElementById('axes');
const tip = document.getElementById('tip');
tip.style.background = T.surface;
tip.style.border = '1px solid ' + T.axis;
tip.style.color = T.ink;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const lineMats = [];

if (!S.is3d) {
  // ═════════════════════════ 2D: ortho + pan/zoom ═════════════════════════
  const M = { l: 58, r: 12, t: 30, b: 40 };
  let W = 100, H = 100;
  const cam = new THREE.OrthographicCamera(-0.03, 1.03, 1.03, -0.03, -10, 10);

  for (const L of S.layers) {
    const n = L.n;
    const cols = layerColors(L, hex2rgb(T.cat[0]));
    const isCat = L.color && L.color.kind === 'cat';
    if (L.kind === 'point') {
      if (isCat) {
        // one Points object per category -> legend click-filtering
        const k = S.color.cats.length;
        const buckets = Array.from({ length: k }, () => []);
        for (let i = 0; i < n; i++) buckets[L.color.data[i] % k].push(i);
        buckets.forEach((idx, ci) => {
          if (!idx.length) return;
          const pos = new Float32Array(idx.length * 3);
          for (let j = 0; j < idx.length; j++) {
            const i = idx[j];
            pos[j*3] = L.x.data[i]; pos[j*3+1] = L.y.data[i]; pos[j*3+2] = 0;
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          const pt = new THREE.Points(g, new THREE.PointsMaterial({
            color: S.color.palette[ci], size: L.size, sizeAttenuation: false,
            transparent: true, opacity: L.alpha }));
          scene.add(pt);
          regCat(ci, pt);
        });
      } else {
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
          pos[i*3] = L.x.data[i]; pos[i*3+1] = L.y.data[i]; pos[i*3+2] = 0;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
        scene.add(new THREE.Points(g, new THREE.PointsMaterial({
          size: L.size, sizeAttenuation: false, vertexColors: true,
          transparent: true, opacity: L.alpha })));
      }
    } else if (L.kind === 'col') {
      // Axis-aligned bars from baseline y0 to y=height (normalized coords).
      const hw = (L.width || 0.08) * 0.5;
      const y0 = (L.y0 != null) ? L.y0 : 0;
      const pos = new Float32Array(n * 6 * 3);
      const col = new Float32Array(n * 6 * 3);
      let p = 0, c = 0;
      for (let i = 0; i < n; i++) {
        const x = L.x.data[i], y = L.y.data[i];
        const x0 = x - hw, x1 = x + hw;
        // two triangles: (x0,y0)-(x1,y0)-(x1,y) and (x0,y0)-(x1,y)-(x0,y)
        const tri = [x0,y0,0, x1,y0,0, x1,y,0, x0,y0,0, x1,y,0, x0,y,0];
        for (let k = 0; k < 18; k++) pos[p++] = tri[k];
        for (let k = 0; k < 6; k++) {
          col[c++] = cols[i*3]; col[c++] = cols[i*3+1]; col[c++] = cols[i*3+2];
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      scene.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({
        vertexColors: true, transparent: true, opacity: L.alpha,
        side: THREE.DoubleSide, depthWrite: false })));
    } else {
      for (const [s0, cnt] of L.groups) {
        if (cnt < 2) continue;
        const flat = new Float32Array(cnt * 3);
        for (let i = 0; i < cnt; i++) {
          flat[i*3] = L.x.data[s0+i]; flat[i*3+1] = L.y.data[s0+i]; flat[i*3+2]=0;
        }
        const lg = new LineGeometry();
        lg.setPositions(Array.from(flat));
        const rgb = [cols[s0*3], cols[s0*3+1], cols[s0*3+2]];
        const lm = new LineMaterial({
          color: new THREE.Color(rgb[0], rgb[1], rgb[2]).getHex(),
          linewidth: L.linewidth, worldUnits: false,
          transparent: true, opacity: L.alpha });
        lineMats.push(lm);
        const ln = new Line2(lg, lm);
        scene.add(ln);
        if (isCat) regCat(L.color.data[s0] % S.color.cats.length, ln);
      }
    }
  }

  function drawAxes() {
    const x0 = dataLo('x') + cam.left  * spanOf('x');
    const x1 = dataLo('x') + cam.right * spanOf('x');
    const y0 = dataLo('y') + cam.bottom * spanOf('y');
    const y1 = dataLo('y') + cam.top    * spanOf('y');
    const px = v => M.l + (v - x0) / (x1 - x0) * W;
    const py = v => M.t + H - (v - y0) / (y1 - y0) * H;
    let s = '';
    // cap tick count by panel size so labels never collide
    const xt = thin(ticksFor('x', x0, x1), Math.max(3, Math.floor(W / 80)));
    const yt = thin(ticksFor('y', y0, y1), Math.max(3, Math.floor(H / 40)));
    for (const [t, lab] of xt) {
      const X = px(t);
      if (X < M.l - 1 || X > M.l + W + 1) continue;
      s += `<line x1="${X}" y1="${M.t}" x2="${X}" y2="${M.t+H}" stroke="${T.grid}"/>`;
      s += `<text x="${X}" y="${M.t+H+14}" fill="${T.muted}" text-anchor="middle">${lab}</text>`;
    }
    for (const [t, lab] of yt) {
      const Y = py(t);
      if (Y < M.t - 1 || Y > M.t + H + 1) continue;
      s += `<line x1="${M.l}" y1="${Y}" x2="${M.l+W}" y2="${Y}" stroke="${T.grid}"/>`;
      s += `<text x="${M.l-7}" y="${Y+4}" fill="${T.muted}" text-anchor="end">${lab}</text>`;
    }
    s += `<rect x="${M.l}" y="${M.t}" width="${W}" height="${H}" fill="none" stroke="${T.axis}"/>`;
    s += `<text x="${M.l+W/2}" y="${M.t+H+30}" fill="${T.ink2}" text-anchor="middle">${S.labs.x}</text>`;
    s += `<text x="14" y="${M.t+H/2}" fill="${T.ink2}" text-anchor="middle" transform="rotate(-90 14 ${M.t+H/2})">${S.labs.y}</text>`;
    svg.innerHTML = s;
  }

  function layout() {
    W = Math.max(50, figEl.clientWidth - M.l - M.r);
    H = Math.max(50, figEl.clientHeight - M.t - M.b);
    host.style.left = M.l + 'px'; host.style.top = M.t + 'px';
    renderer.setSize(W, H);
    svg.setAttribute('width', figEl.clientWidth);
    svg.setAttribute('height', figEl.clientHeight);
    for (const lm of lineMats) lm.resolution.set(W, H);
    draw();
  }
  let rafPending = false;
  function draw() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      cam.updateProjectionMatrix();
      renderer.render(scene, cam);
      drawAxes();
    });
  }
  redraw = draw;

  // pan / zoom / hover
  const el = renderer.domElement;
  el.style.touchAction = 'none';
  function clampView() {
    const MAX = 2.4, LO = -0.7, HI = 1.7;
    for (const [a, b] of [['left', 'right'], ['bottom', 'top']]) {
      let span = cam[b] - cam[a];
      if (span > MAX) {
        const c = (cam[a] + cam[b]) / 2;
        cam[a] = c - MAX / 2; cam[b] = c + MAX / 2;
      }
      if (cam[a] < LO) { cam[b] += LO - cam[a]; cam[a] = LO; }
      if (cam[b] > HI) { cam[a] -= cam[b] - HI; cam[b] = HI; }
    }
  }
  let dragging = null;
  el.addEventListener('pointerdown', e => {
    dragging = { x: e.clientX, y: e.clientY,
                 l: cam.left, r: cam.right, t: cam.top, b: cam.bottom };
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener('pointerup', () => dragging = null);
  el.addEventListener('pointermove', e => {
    if (dragging) {
      const dx = (e.clientX - dragging.x) / W * (dragging.r - dragging.l);
      const dy = (e.clientY - dragging.y) / H * (dragging.t - dragging.b);
      cam.left = dragging.l - dx; cam.right = dragging.r - dx;
      cam.top = dragging.t + dy;  cam.bottom = dragging.b + dy;
      clampView();
      tip.style.display = 'none';
      draw();
    } else hover(e);
  });
  const hintEl = document.getElementById('hint');
  hintEl.style.background = T.surface + 'e6';
  hintEl.style.border = '1px solid ' + T.axis;
  hintEl.style.color = T.ink2;
  let hintT = 0, hintOff = 0;
  el.addEventListener('wheel', e => {
    if (!e.ctrlKey && !e.metaKey) {
      // let the page scroll; nudge toward the modifier
      const now = performance.now();
      if (now - hintT > 1500) {
        hintT = now;
        hintEl.textContent = (navigator.platform || '').includes('Mac')
          ? 'Use \\u2318 + scroll to zoom' : 'Use Ctrl + scroll to zoom';
        hintEl.style.opacity = '1';
        clearTimeout(hintOff);
        hintOff = setTimeout(() => hintEl.style.opacity = '0', 1200);
      }
      return;
    }
    e.preventDefault();
    const f = Math.exp(e.deltaY * 0.0015);
    const r = el.getBoundingClientRect();
    const cx = cam.left + (e.clientX - r.left) / W * (cam.right - cam.left);
    const cy = cam.top - (e.clientY - r.top) / H * (cam.top - cam.bottom);
    cam.left = cx + (cam.left - cx) * f;   cam.right = cx + (cam.right - cx) * f;
    cam.top = cy + (cam.top - cy) * f;     cam.bottom = cy + (cam.bottom - cy) * f;
    clampView();
    tip.style.display = 'none';
    draw();
  }, { passive: false });
  el.addEventListener('dblclick', () => {
    cam.left = -0.03; cam.right = 1.03; cam.bottom = -0.03; cam.top = 1.03;
    draw();
  });

  let hoverTick = 0;
  function hover(e) {
    const now = performance.now();
    if (now - hoverTick < 33) return;
    hoverTick = now;
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bestD = 12 * 12;
    for (const L of S.layers) {
      const catL = L.color && L.color.kind === 'cat';
      for (let i = 0; i < L.n; i++) {
        if (catL && hiddenCats.has(L.color.data[i] % S.color.cats.length))
          continue;
        const sx = (L.x.data[i] - cam.left) / (cam.right - cam.left) * W;
        const sy = (cam.top - L.y.data[i]) / (cam.top - cam.bottom) * H;
        const d = (sx-mx)*(sx-mx) + (sy-my)*(sy-my);
        if (d < bestD) { bestD = d; best = [L, i, sx, sy]; }
      }
    }
    if (!best) { tip.style.display = 'none'; return; }
    const [L, i, sx, sy] = best;
    const xv = dataLo('x') + L.x.data[i] * spanOf('x');
    const yv = dataLo('y') + L.y.data[i] * spanOf('y');
    let head = '';
    if (L.color && L.color.kind === 'cat')
      head = '<b>' + S.color.cats[L.color.data[i] % S.color.cats.length] + '</b><br>';
    else if (L.color && L.color.kind === 'num')
      head = '<b>' + fmt(cval(L.color.data[i] / 65535)) + '</b><br>';
    tip.innerHTML = head
      + fmtSpan('x', xv, (cam.right - cam.left) * spanOf('x')) + ', '
      + fmtSpan('y', yv, (cam.top - cam.bottom) * spanOf('y'));
    tip.style.left = (M.l + sx + 12) + 'px';
    tip.style.top = (M.t + sy - 10) + 'px';
    tip.style.display = 'block';
  }
  el.addEventListener('pointerleave', () => tip.style.display = 'none');

  new ResizeObserver(layout).observe(figEl);
  layout();

} else {
  // ═════════════════════════ 3D: orbit viewer ═════════════════════════════
  host.style.left = '0'; host.style.top = '0';
  const cam = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  cam.up.set(0, 0, 1);
  // proportional cube: preserve data aspect across axes
  const spans = axesList.map(a => spanOf(a));
  const maxSpan = Math.max(...spans);
  const ext = axesList.map((a, i) => spans[i] / maxSpan);
  const toCube = (a, i, v) => v * ext[i];

  for (const L of S.layers) {
    const n = L.n;
    const cols = layerColors(L, hex2rgb(T.cat[0]));
    const isCat = L.color && L.color.kind === 'cat';
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i*3]   = L.x.data[i] * ext[0];
      pos[i*3+1] = L.y.data[i] * ext[1];
      pos[i*3+2] = L.z.data[i] * ext[2];
    }
    if (L.kind === 'point') {
      if (isCat) {
        const k = S.color.cats.length;
        const buckets = Array.from({ length: k }, () => []);
        for (let i = 0; i < n; i++) buckets[L.color.data[i] % k].push(i);
        buckets.forEach((idx, ci) => {
          if (!idx.length) return;
          const sub = new Float32Array(idx.length * 3);
          for (let j = 0; j < idx.length; j++) {
            const i = idx[j];
            sub[j*3] = pos[i*3]; sub[j*3+1] = pos[i*3+1]; sub[j*3+2] = pos[i*3+2];
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(sub, 3));
          const pt = new THREE.Points(g, new THREE.PointsMaterial({
            color: S.color.palette[ci], size: L.size, sizeAttenuation: true,
            transparent: true, opacity: L.alpha }));
          scene.add(pt);
          regCat(ci, pt);
        });
      } else {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
        scene.add(new THREE.Points(g, new THREE.PointsMaterial({
          size: L.size, sizeAttenuation: true, vertexColors: true,
          transparent: true, opacity: L.alpha })));
      }
    } else {
      for (const [s0, cnt] of L.groups) {
        if (cnt < 2) continue;
        const lg = new LineGeometry();
        lg.setPositions(Array.from(pos.subarray(s0*3, (s0+cnt)*3)));
        const lm = new LineMaterial({
          color: new THREE.Color(cols[s0*3], cols[s0*3+1], cols[s0*3+2]).getHex(),
          linewidth: L.linewidth, worldUnits: false,
          transparent: true, opacity: L.alpha });
        lineMats.push(lm);
        const ln = new Line2(lg, lm);
        scene.add(ln);
        if (isCat) regCat(L.color.data[s0] % S.color.cats.length, ln);
      }
    }
  }

  // axes box + static ticks (Python-computed) + labels as sprites
  const boxMat = new THREE.LineBasicMaterial({ color: T.axis });
  const bg = new THREE.BufferGeometry().setFromPoints([
    [0,0,0],[1,0,0],[1,0,0],[1,1,0],[1,1,0],[0,1,0],[0,1,0],[0,0,0],
    [0,0,1],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,0,1],
    [0,0,0],[0,0,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,1,0],[0,1,1],
  ].map(p => new THREE.Vector3(p[0]*ext[0], p[1]*ext[1], p[2]*ext[2])));
  scene.add(new THREE.LineSegments(bg, boxMat));

  function sprite(text, small) {
    const c = document.createElement('canvas');
    const ctx = c.getContext('2d');
    const fs = small ? 22 : 26;
    ctx.font = fs + 'px system-ui';
    c.width = Math.max(2, Math.ceil(ctx.measureText(text).width) + 8);
    c.height = fs + 10;
    const ctx2 = c.getContext('2d');
    ctx2.font = fs + 'px system-ui';
    ctx2.fillStyle = small ? T.muted : T.ink2;
    ctx2.textBaseline = 'middle';
    ctx2.fillText(text, 4, c.height / 2);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, depthTest: false, transparent: true }));
    const k = small ? 0.0016 : 0.0019;
    sp.scale.set(c.width * k, c.height * k, 1);
    return sp;
  }
  function tickPos(ax, t) {                    // data -> cube coords on min edges
    return (t - dataLo(ax)) / spanOf(ax);
  }
  const off = 0.055;
  axesList.forEach((ax, ai) => {
    const sc = S.scales[ax];
    const ticks = sc.kind === 'cat'
      ? sc.cats.map((c, i) => [i, c])
      : (sc.ticks || (sc.ladder ? sc.ladder[0] : []));
    for (const [t, lab] of ticks) {
      const u = tickPos(ax, t);
      if (u < -0.001 || u > 1.001) continue;
      const p = [[u*ext[0], -off*ext[1], 0], [-off*ext[0], u*ext[1], 0],
                 [-off*ext[0], 0, u*ext[2]]][ai];
      const sp = sprite(String(lab), true);
      sp.position.set(p[0], p[1], p[2]);
      scene.add(sp);
    }
    const lp = [[0.5*ext[0], -2.6*off*ext[1], 0],
                [-2.6*off*ext[0], 0.5*ext[1], 0],
                [-2.6*off*ext[0], 0, 0.55*ext[2]]][ai];
    const tl = sprite(S.labs[ax], false);
    tl.position.set(lp[0], lp[1], lp[2]);
    scene.add(tl);
  });

  const ctr = new THREE.Vector3(ext[0]/2, ext[1]/2, ext[2]/2);
  // fit: place the camera so the cube's bounding sphere fills the frustum
  const rad = Math.sqrt(ext[0]*ext[0] + ext[1]*ext[1] + ext[2]*ext[2]) / 2;
  const dist = rad / Math.tan((cam.fov * Math.PI / 180) / 2) * 1.45;
  const dir = new THREE.Vector3(0.55, -0.85, 0.5).normalize();
  cam.position.copy(ctr.clone().add(dir.multiplyScalar(dist)));
  cam.near = dist / 100;
  cam.far = dist * 20;
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.copy(ctr);
  controls.enableDamping = true;

  function layout() {
    const w = Math.max(figEl.clientWidth, 1), h = Math.max(figEl.clientHeight, 1);
    renderer.setSize(w, h);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    for (const lm of lineMats) lm.resolution.set(w, h);
  }
  new ResizeObserver(layout).observe(figEl);
  layout();
  (function loop() {
    controls.update();
    renderer.render(scene, cam);
    requestAnimationFrame(loop);
  })();
}
</script>
</body></html>"""
