"""Embedded three.js viewer HTML template."""

from __future__ import annotations

_DOC_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}</script>
<style>
html,body{margin:0;height:100%;overflow:hidden;
  font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
body{display:flex;flex-direction:column}
#fig{position:relative;width:100%;flex:1;min-height:0;z-index:1}
#note{display:none;flex:none;padding:2px 14px 8px;font-size:11px;line-height:1.4;white-space:pre-line}
#title{position:absolute;left:14px;top:8px;max-width:calc(100% - 120px);
  font-size:14px;font-weight:600;z-index:4}
#canvas-host{position:absolute;z-index:1}
#year{display:none;position:absolute;left:50%;top:46%;transform:translate(-50%,-50%);
  z-index:0;font-weight:700;line-height:1;letter-spacing:-0.04em;opacity:0.13;
  pointer-events:none;user-select:none}
#axes{position:absolute;inset:0;pointer-events:none;z-index:2}
#grid{position:absolute;inset:0;pointer-events:none;z-index:0}
#legend{position:absolute;right:10px;top:36px;z-index:4;padding:6px 9px;
  border-radius:6px;font-size:11px;line-height:1.7;max-width:min(46%,280px);
  box-sizing:border-box}
#legend .lg-formula{display:block}
#legend .lg-vals{display:flex;flex-wrap:wrap;column-gap:8px}
#legend .lg-kv{white-space:nowrap}
#modebar{position:absolute;top:6px;right:6px;z-index:6;
  opacity:0;transition:opacity .12s ease;user-select:none}
#fig:hover #modebar,#modebar:focus-within,#modebar.open{opacity:1}
@media (hover:none){#modebar{opacity:1}}
#save-btn{font:600 12px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:4px 8px;border-radius:4px;cursor:pointer;color:inherit;
  box-shadow:0 1px 2px rgba(0,0,0,.25)}
#save-menu{position:absolute;right:0;top:calc(100% + 4px);min-width:232px;
  padding:4px;border-radius:6px;box-shadow:0 8px 24px rgba(0,0,0,.35)}
#modebar.open-up #save-menu{top:auto;bottom:calc(100% + 4px)}
#save-menu[hidden]{display:none}
#save-menu button{display:flex;align-items:baseline;justify-content:space-between;
  gap:12px;width:100%;text-align:left;padding:6px 8px;border:0;border-radius:4px;
  background:transparent;color:inherit;cursor:pointer;white-space:nowrap;
  font:12px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif}
#save-menu button[hidden]{display:none}
#save-menu .save-k{font-weight:650;min-width:4.2em}
#save-menu .save-d{opacity:.72;font-weight:400}
#save-menu .save-sep{height:1px;margin:4px 6px;background:currentColor;opacity:.25}
#save-menu button.save-status{display:block;white-space:normal;font-weight:500;text-align:left}
#save-btn:hover,#save-menu button:hover,#save-menu button:focus{background:rgba(128,128,128,.18)}
#save-btn:focus-visible,#save-menu button:focus-visible{outline:2px solid currentColor;outline-offset:-2px}
#legend .sw{display:inline-block;width:9px;height:9px;border-radius:5px;
  margin-right:6px;vertical-align:-1px}
#legend .lg-e{cursor:pointer;user-select:none}
#legend .sz{display:flex;align-items:center;gap:8px;line-height:1.2;margin:3px 0}
#legend .sz-block{clear:both;padding-top:6px}
#legend .bub{display:inline-block;border-radius:50%;box-sizing:border-box;flex:none;
  border:1px solid rgba(0,0,0,.35)}
#player{display:none;flex:none;align-items:center;flex-wrap:wrap;gap:8px;padding:4px 10px 8px;max-width:100%;box-sizing:border-box}
#player button{font:inherit;padding:3px 10px;border-radius:5px;cursor:pointer}
#player #play-range{flex:1;min-width:80px}
#player .plot3-slider{display:flex;align-items:center;gap:6px;flex:1 1 240px;min-width:0;max-width:100%}
#player .plot3-slider input[type=range]{flex:1;min-width:0}
#player .plot3-slider-readout{font-variant-numeric:tabular-nums;white-space:nowrap}
.plot3-math{white-space:nowrap}
#tip{position:absolute;display:none;z-index:5;pointer-events:none;
  padding:4px 8px;border-radius:5px;font-size:11px;white-space:nowrap;
  line-height:1.45}
#hint{position:absolute;left:50%;bottom:46px;transform:translateX(-50%);
  z-index:5;pointer-events:none;padding:5px 10px;border-radius:5px;
  font-size:11px;opacity:0;transition:opacity .25s}
#ramp{height:8px;width:110px;border-radius:4px;margin-top:3px}
</style></head><body>
<div id="fig">
  <div id="title"></div>
  <div id="year"></div>
  <svg id="grid"></svg>
  <div id="canvas-host"></div>
  <svg id="axes"></svg>
  <div id="legend" style="display:none"></div>
  <div id="tip"></div>
  <div id="hint"></div>
  <div id="modebar">
    <button type="button" id="save-btn" aria-haspopup="menu" aria-expanded="false" aria-controls="save-menu">⤓ Save</button>
    <div id="save-menu" role="menu" aria-label="Save" hidden>
      <button type="button" role="menuitem" data-act="html"><span class="save-k">HTML</span><span class="save-d">Interactive figure</span></button>
      <button type="button" role="menuitem" data-act="svg"><span class="save-k">SVG</span><span class="save-d">Vector, for papers</span></button>
      <button type="button" role="menuitem" data-act="png"><span class="save-k">PNG</span><span class="save-d">Image (2× sharp)</span></button>
      <div class="save-sep" role="separator"></div>
      <button type="button" role="menuitem" data-act="video" hidden><span class="save-k">Video</span><span class="save-d">WebM (animations)</span></button>
      <button type="button" role="menuitem" data-act="copy">Copy PNG to clipboard</button>
    </div>
  </div>
</div>
<div id="player">
  <button type="button" id="play-btn">Play</button>
  <input id="play-range" type="range" min="0" max="1000" value="1000" aria-label="Frame">
  <span id="play-readout"></span>
  <label>Speed <input id="play-speed" type="range" min="0.25" max="4" step="0.25" value="1" aria-label="Speed"></label>
</div>
<div id="note"></div>
__PAYLOADS__
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';

// Captured before the viewer touches the DOM. Saving the live tree would
// duplicate the canvas and the legend when the file is opened again.
const PRISTINE = '<!doctype html>' + document.documentElement.outerHTML;
function readSavedState() {
  const node = document.getElementById('plot3-state');
  if (!node) return null;
  try {
    const data = JSON.parse(node.textContent || '');
    if (!data || typeof data !== 'object') return null;
    return data;
  } catch (err) {
    return null;
  }
}
const SAVED = readSavedState();
const S = __SPEC__;
const T = S.theme;
document.body.style.background = T.surface;
document.body.style.color = T.ink;
const noteEl = document.getElementById('note');
if (S.notes && S.notes.length) {
  noteEl.style.display = 'block';
  noteEl.style.color = T.muted;
  noteEl.textContent = S.notes.join('\\n');
}

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
  if (dtype === 'u32') return new Uint32Array(a.buffer);
  if (dtype === 'u8') return a;
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
const extraChans = ['ymin','lower','middle','upper','ymax','ox','oy'];
for (const L of S.layers) {
  for (const a of axesList) {
    if (L[a] && L[a].id) L[a].data = toNorm(await decode(L[a].id, L[a].dtype));
  }
  for (const a of extraChans) {
    if (L[a] && L[a].id) L[a].data = toNorm(await decode(L[a].id, L[a].dtype));
  }
  if (L.color) L.color.data = await decode(L.color.id, 'u16');
  if (L.ocolor) L.ocolor.data = await decode(L.ocolor.id, 'u16');
  if (L.indices && L.indices.id) {
    L.indices.data = await decode(L.indices.id, 'u32');
  }
  if (L.size && L.size.id) L.size.data = toNorm(await decode(L.size.id, L.size.dtype));
  if (L.shape && L.shape.id) L.shape.data = await decode(L.shape.id, 'u16');
  if (L.frames) {
    const F = L.frames;
    for (const key of ['x', 'y', 'z', 'size']) {
      if (!F[key] || !F[key].id) continue;
      F[key].data = toNorm(await decode(F[key].id, F[key].dtype));
      if (F[key].mask) F[key].maskData = await decode(F[key].mask, 'u8');
    }
    if (F.color && F.color.id) {
      const raw = await decode(F.color.id, 'u16');
      F.color.data = F.color.kind === 'num' ? toNorm(raw) : raw;
      if (F.color.mask) F.color.maskData = await decode(F.color.mask, 'u8');
    }
  }
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
function plot3Esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);
}
function plot3MathHTML(segs, plain) {
  if (!segs || !segs.length) return plot3Esc(plain);
  return segs.map(p => p.latex
    ? '<span class="plot3-math" data-latex="' + plot3Esc(p.latex) + '">' + plot3Esc(p.text) + '</span>'
    : plot3Esc(p.text)).join('');
}
function plot3Typeset(root) {
  const katex = window.katex;
  if (!katex) return;
  (root || document).querySelectorAll('.plot3-math').forEach(el => {
    if (el.dataset.katexDone) return;
    const latex = el.getAttribute('data-latex');
    if (!latex) return;
    katex.render(latex, el, {throwOnError: false, trust: false, displayMode: false});
    el.dataset.katexDone = '1';
  });
}
function formulaHead(L) {
  if (!L || !L.tip) return '';
  if (!L.tip.latex) return plot3Esc(L.tip.pretty) + '<br>';
  return '<span class="plot3-math" data-latex="' + plot3Esc(L.tip.latex) + '">'
    + plot3Esc(L.tip.pretty) + '</span><br>';
}
function axisPair(ax, text) {
  const name = (S.labs && S.labs[ax]) || ax;
  return plot3Esc(name) + ' = ' + plot3Esc(String(text).replace(/-/g, '\u2212'));
}
function sizeLegendHTML() {
  const sl = S.sizeLegend;
  if (!sl || !sl.breaks || !sl.breaks.length) return '';
  const rows = sl.breaks.map(b => {
    const d = Math.max(6, Math.round((b.t || 0) * 26));
    return '<div class="sz"><span class="bub" style="width:' + d + 'px;height:' + d
      + 'px;background:' + T.ink2 + '"></span><span>' + plot3Esc(b.label) + '</span></div>';
  }).join('');
  const title = sl.label
    ? '<b style="color:' + T.ink + '">' + plot3Esc(sl.label) + '</b>' : '';
  return '<div class="sz-block">' + title + rows + '</div>';
}
function legendLabel(e) {
  if (e.math) return plot3MathHTML(e.math, e.label);
  const plain = String(e.label || '');
  const cut = plain.indexOf('  (');
  // A parameter tail stays in whole "a = 2" pieces so a narrow key cannot
  // split 0.03333. The formula above it wraps on spaces.
  if (cut >= 0) {
    const formula = plain.slice(0, cut);
    let body = plain.slice(cut + 3);
    if (body.endsWith(')')) body = body.slice(0, -1);
    const vals = body.split(', ').filter(Boolean).map(part =>
      '<span class="lg-kv">' + plot3Esc(part) + '</span>').join('');
    return '<span class="lg-formula">' + plot3Esc(formula) + '</span>'
      + '<span class="lg-vals">' + vals + '</span>';
  }
  if (e.latex) return '<span class="plot3-math" data-latex="' + plot3Esc(e.latex) + '">'
    + plot3Esc(plain) + '</span>';
  return plot3Esc(plain);
}
function richLabel(key, plain) {
  if (S.labsMath && S.labsMath[key]) return plot3MathHTML(S.labsMath[key], plain || '');
  return plot3Esc(plain || '');
}
const figEl = document.getElementById('fig');
const titleEl = document.getElementById('title');
if (S.labs.title) {
  if (S.labsMath && S.labsMath.title) titleEl.innerHTML = plot3MathHTML(S.labsMath.title, S.labs.title);
  else titleEl.textContent = S.labs.title;
}
const THEME_OPTS = S.themeOpts || {};
// labs(tag=, subtitle=, caption=) and theme(plot_title_hjust=).
if (S.labs.tag) {
  const tagEl = document.createElement('b');
  tagEl.style.marginRight = '8px';
  tagEl.textContent = S.labs.tag;
  titleEl.insertBefore(tagEl, titleEl.firstChild);
}
if (S.labs.subtitle) {
  const subEl = document.createElement('div');
  subEl.style.cssText = 'font-weight:400;font-size:12px;margin-top:2px;color:' + T.ink2;
  subEl.innerHTML = richLabel('subtitle', S.labs.subtitle);
  titleEl.appendChild(subEl);
}
if (THEME_OPTS.titleHjust != null && THEME_OPTS.titleHjust > 0.25) {
  titleEl.style.left = '14px';
  titleEl.style.right = '14px';
  titleEl.style.maxWidth = 'none';
  titleEl.style.textAlign = THEME_OPTS.titleHjust > 0.75 ? 'right' : 'center';
}
if (S.labs.caption) {
  const capEl = document.createElement('div');
  capEl.style.cssText = 'flex:none;padding:0 14px 6px;text-align:right;font-size:11px;color:' + T.muted;
  capEl.innerHTML = richLabel('caption', S.labs.caption);
  figEl.insertAdjacentElement('afterend', capEl);
}
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
function showLegendBox() {
  // theme(legend_position="none"), and all but one panel of a facet_grid.
  if (S.legendPosition === 'none') return;
  legEl.style.display = 'block';
  legEl.style.background = T.surface + 'e6';
  legEl.style.border = '1px solid ' + T.grid;
  legEl.style.color = T.ink2;
}
function placeLegend() {
  if (!legEl || legEl.style.display === 'none' || legEl.dataset.docked === '1') return;
  const figH = figEl.clientHeight || 0;
  if (figH < 40) return;
  // A key taller than the panel covers the curve. Sit it under the axes.
  if (legEl.offsetHeight > figH * 0.40) {
    legEl.dataset.docked = '1';
    legEl.style.top = 'auto';
    legEl.style.bottom = '8px';
    legEl.style.left = '8px';
    legEl.style.right = '8px';
    legEl.style.maxWidth = 'none';
  }
}
const LEGEND_GLYPH = { circle: '●', triangle: '▲', square: '■', diamond: '◆', plus: '+', cross: '×' };
// Legend keys: a square, the point's symbol, or a short (dashed) line.
function keyHTML(e, color) {
  if (e.shape) {
    return '<span class="sw" style="background:none;border-radius:0;width:auto;color:' + color +
      ';font-size:11px;line-height:9px">' + (LEGEND_GLYPH[e.shape] || '●') + '</span>';
  }
  if (e.dash !== undefined && e.dash !== null) {
    const da = (e.dash && e.dash.length) ? ' stroke-dasharray="' + e.dash.map(d => d * 1.6).join(' ') + '"' : '';
    return '<svg width="14" height="9" style="margin-right:4px;vertical-align:middle"><line x1="0" y1="4.5" x2="14" y2="4.5" stroke="' +
      color + '" stroke-width="1.6"' + da + '/></svg>';
  }
  return '<span class="sw" style="background:' + color + '"></span>';
}
// aes(shape=) / aes(linetype=) on a column other than colour: own block.
function keyLegendsHTML() {
  let h = '';
  for (const lg of [S.shapeLegend, S.linetypeLegend]) {
    if (!lg) continue;
    h += '<div style="margin-top:4px"><b style="color:' + T.ink + '">' + plot3Esc(lg.label) + '</b></div>' +
      lg.entries.map(e => '<div>' + keyHTML(e, T.ink) + plot3Esc(e.label) + '</div>').join('');
  }
  return h;
}
const szHTML = sizeLegendHTML() + keyLegendsHTML();
// Class entries and a height colour bar together (geom_box3d on a cloud).
function barHTML() {
  if (S.labs.colorBar == null || !S.color || S.color.kind !== 'num' || S.color.guide === false) return '';
  return '<div class="sz-block"><b style="color:' + T.ink + '">' + plot3Esc(S.labs.colorBar) +
    '</b><div id="ramp" style="background:linear-gradient(90deg,' + S.color.ramp.join(',') +
    ')"></div><span style="float:left">' + (+S.color.lo.toPrecision(3)) +
    '</span><span style="float:right">' + (+S.color.hi.toPrecision(3)) + '</span></div>';
}
if (S.legend) {
  showLegendBox();
  legEl.innerHTML = (S.labs.color ? '<b style="color:'+T.ink+'">' +
      richLabel('color', S.labs.color) + '</b>' : '') +
    S.legend.map((e, i) => '<div class="lg-e" data-ci="' + i +
      '">' + keyHTML(e, e.color) +
      legendLabel(e) + '</div>').join('') + barHTML() + szHTML;
  legEl.addEventListener('click', ev => {
    const row = ev.target.closest('.lg-e');
    if (!row) return;
    const ci = +row.dataset.ci;
    if (hiddenCats.has(ci)) hiddenCats.delete(ci); else hiddenCats.add(ci);
    row.style.opacity = hiddenCats.has(ci) ? 0.35 : 1;
    for (const o of (catObjs.get(ci) || [])) o.visible = !hiddenCats.has(ci);
    if (window.__plot3.afterLegend) window.__plot3.afterLegend();
    tip.style.display = 'none';
    redraw();
  });
} else if (S.color.kind === 'num' && S.color.guide !== false) {
  showLegendBox();
  legEl.innerHTML = '<b style="color:'+T.ink+'">' + richLabel('color', S.labs.color||'') +
    '</b><div id="ramp" style="background:linear-gradient(90deg,' +
    S.color.ramp.join(',') + ')"></div>' +
    '<span style="float:left">' + (+S.color.lo.toPrecision(3)) + '</span>' +
    '<span style="float:right">' + (+S.color.hi.toPrecision(3)) + '</span>' + szHTML;
} else if (szHTML) {
  showLegendBox();
  legEl.innerHTML = szHTML;
}
__KATEX__

function fmt(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toPrecision(3);
  return String(+v.toFixed(6));
}
// Tick labels that stay apart: 1000000000001, not five "1.00e+12".
function fmtTicks(vals) {
  let labels = vals.map(fmt);
  if (new Set(labels).size === labels.length) return labels;
  if (vals.every(v => Number.isInteger(v) && Math.abs(v) < 1e15)) return vals.map(v => String(v));
  for (let d = 4; d <= 15; d++) {
    labels = vals.map(v => String(+v.toPrecision(d)));
    if (new Set(labels).size === labels.length) return labels;
  }
  return labels;
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

// nice numeric ticks (JS side for pan/zoom), the same rule as scales.nice_ticks.
function niceTicks(lo, hi, n) {
  // About n ticks on a 1-2-2.5-5 grid inside the range, as ggplot2 (and
  // plot3's Python side) choose them: 0, 2, 4, 6, 8 for 0 to 9.2.
  if (!(hi > lo) || !isFinite(lo) || !isFinite(hi)) return [lo];
  const span = hi - lo, target = Math.max(3, n || 5), pad = 0.05 * span;
  const exp = Math.floor(Math.log10(Math.max(span / Math.max(1, target - 1), 1e-300)));
  let best = null, bestScore = Infinity, bestStep = -1;
  for (let shift = -1; shift <= 1; shift++) {
    const base = Math.pow(10, exp + shift);
    for (const mult of [1, 2, 2.5, 5]) {
      const step = mult * base;
      const first = Math.ceil((lo - pad) / step - 1e-9);
      const last = Math.floor((hi + pad) / step + 1e-9);
      const count = last - first + 1;
      if (count < 2 || count > 40) continue;
      let score = Math.abs(count - target);
      if (count < 4) score += 3;
      if (count > target + 2) score += 2 * (count - target - 2);
      score += 2 * (Math.max(0, first * step - lo) + Math.max(0, hi - last * step)) / span;
      if (mult === 2.5) score += 0.25;
      if (score < bestScore - 1e-9 || (Math.abs(score - bestScore) <= 1e-9 && step > bestStep)) {
        const digits = Math.max(0, 3 - Math.floor(Math.log10(step)));
        best = [];
        for (let k = first; k <= last; k++) {
          const tick = k * step;
          best.push(Math.abs(tick) < step * 1e-8 ? 0 : +tick.toFixed(Math.min(digits, 20)));
        }
        bestScore = score; bestStep = step;
      }
    }
  }
  return best || [lo, hi];
}
// ticks for any scale over a visible data range -> [[pos_data, label], ...]
function thin(vis, maxN) {
  // Every k-th tick, evenly spaced, on round values (0, 10, 20) when the
  // ticks allow: never a lone last tick at an odd distance.
  if (vis.length <= maxN) return vis;
  const numeric = vis.length > 1 && typeof vis[0][0] === 'number' && typeof vis[1][0] === 'number';
  for (let k = 2; k <= vis.length; k++) {
    if (Math.ceil(vis.length / k) > Math.max(2, maxN)) continue;
    let offset = 0;
    if (numeric) {
      const step = Math.abs(vis[1][0] - vis[0][0]) * k;
      for (let o = 0; o < k; o++) {
        const r = vis[o][0] / step;
        if (step > 0 && Math.abs(r - Math.round(r)) < 1e-6) { offset = o; break; }
      }
    }
    return vis.filter((_, i) => i >= offset && (i - offset) % k === 0);
  }
  return vis.slice(0, 1);
}
function ticksFor(ax, lo, hi) {
  const sc = S.scales[ax];
  if (lo > hi) { const t = lo; lo = hi; hi = t; }  // scale_*_reverse()
  // scale_*_continuous(breaks=, labels=): keep the chosen ticks when zooming.
  if (sc.fixed && sc.ticks) return sc.ticks.filter(t => t[0] >= lo && t[0] <= hi);
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
  if (sc.trans === 'log10') return logTicks(lo, hi);
  const vals = niceTicks(lo, hi, 5);
  return vals.map((v, i) => [v, fmtTicks(vals)[i]]);
}
function logTicks(lo, hi) {
  if (!(hi > lo)) return [[lo, fmt(Math.pow(10, lo))]];
  const span = hi - lo;
  const mults = span <= 3 ? [1, 2, 5] : [1];
  const out = [];
  const e0 = Math.floor(lo), e1 = Math.ceil(hi);
  for (let e = e0; e <= e1; e++) {
    for (const m of mults) {
      const val = m * Math.pow(10, e);
      const lv = Math.log10(val);
      if (lv < lo - 1e-9 || lv > hi + 1e-9) continue;
      out.push([lv, fmt(val)]);
    }
  }
  if (!out.length) out.push([lo, fmt(Math.pow(10, lo))]);
  return out;
}
// hover value: enough digits to resolve ~1/300 of the visible span
function fmtSpan(ax, v, spanData) {
  const sc = S.scales[ax];
  if (sc.kind !== 'num') return fmtAxis(ax, v);
  if (sc.trans === 'log10') return fmt(v);
  const d = Math.max(0, Math.min(6,
    Math.ceil(-Math.log10(Math.max(1e-12, spanData / 300)))));
  return v.toFixed(d);
}
const dataLo = ax => S.scales[ax].lo, dataHi = ax => S.scales[ax].hi;
const spanOf = ax => (dataHi(ax) - dataLo(ax)) || 1;
function fromScale(ax, norm) {
  const sc = S.scales[ax];
  const v = sc.lo + norm * ((sc.hi - sc.lo) || 1);
  if (sc.trans === 'log10') return Math.pow(10, v);
  return v;
}

const host = document.getElementById('canvas-host');
const svg = document.getElementById('axes');
// The 2D grid sits behind the data, as in ggplot2; axes and labels above it.
const gridSvg = document.getElementById('grid');
const tip = document.getElementById('tip');
tip.style.background = T.surface;
tip.style.border = '1px solid ' + T.axis;
tip.style.color = T.ink;

// preserveDrawingBuffer keeps the frame readable for the Save menu.
// The spec's colours are exact sRGB values. Without this, three.js reads
// vertex colours as linear and brightens them on output (a washed viridis).
THREE.ColorManagement.enabled = false;
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const lineMats = [];

// Point symbols (aes(shape=) / geom_point(shape=)) as white sprites tinted
// by the point colour.
const shapeMaps = {};
function shapeTexture(name) {
  if (shapeMaps[name]) return shapeMaps[name];
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#ffffff'; g.strokeStyle = '#ffffff';
  g.beginPath();
  if (name === 'triangle') { g.moveTo(32, 6); g.lineTo(60, 54); g.lineTo(4, 54); g.closePath(); g.fill(); }
  else if (name === 'square') { g.fillRect(9, 9, 46, 46); }
  else if (name === 'diamond') { g.moveTo(32, 2); g.lineTo(62, 32); g.lineTo(32, 62); g.lineTo(2, 32); g.closePath(); g.fill(); }
  else if (name === 'plus') { g.lineWidth = 12; g.moveTo(32, 4); g.lineTo(32, 60); g.moveTo(4, 32); g.lineTo(60, 32); g.stroke(); }
  else if (name === 'cross') { g.lineWidth = 12; g.moveTo(10, 10); g.lineTo(54, 54); g.moveTo(54, 10); g.lineTo(10, 54); g.stroke(); }
  else { g.arc(32, 32, 28, 0, Math.PI * 2); g.fill(); }
  const t = new THREE.CanvasTexture(c);
  t.needsUpdate = true;
  shapeMaps[name] = t;
  return t;
}
function shapeOf(L, i) {
  if (!L.shape) return null;
  if (typeof L.shape === 'string') return L.shape;
  const names = L.shape.names || ['circle'];
  return names[L.shape.data[i] % names.length];
}

// Bubbles: per-point size (area) and, when a transition is set, a frame tween.
let circleMap = null;
function circleTexture() {
  if (circleMap) return circleMap;
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const g = c.getContext('2d');
  g.clearRect(0, 0, 64, 64);
  g.beginPath();
  g.arc(32, 32, 28, 0, Math.PI * 2);
  g.fillStyle = '#ffffff';
  g.fill();
  g.lineWidth = 4;
  g.strokeStyle = 'rgba(0,0,0,0.7)';
  g.stroke();
  circleMap = new THREE.CanvasTexture(c);
  circleMap.needsUpdate = true;
  return circleMap;
}
function bubbleMaterial(opacity, sizeAtten) {
  const m = new THREE.PointsMaterial({
    size: 1,
    map: circleTexture(),
    vertexColors: true,
    transparent: true,
    opacity: opacity == null ? 0.9 : opacity,
    depthWrite: false,
    depthTest: !!sizeAtten,
    sizeAttenuation: !!sizeAtten,
    alphaTest: 0.01,
  });
  // Cache key so this patched program is not reused for plain points.
  m.customProgramCacheKey = () => 'plot3-bubble';
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        'uniform float size;',
        'uniform float size;\\nattribute float aSize;\\nattribute float aAlpha;\\nvarying float vAlpha;'
      )
      .replace(
        '#include <color_vertex>',
        '#include <color_vertex>\\nvAlpha = aAlpha;'
      )
      .replace('gl_PointSize = size;', 'gl_PointSize = max(aSize, 0.0);');
    shader.fragmentShader = shader.fragmentShader
      .replace(
        'uniform float opacity;',
        'uniform float opacity;\\nvarying float vAlpha;'
      )
      .replace(
        'vec4 diffuseColor = vec4( diffuse, opacity );',
        'vec4 diffuseColor = vec4( diffuse, opacity * vAlpha );'
      );
  };
  return m;
}
const bubbleEntries = [];
const lineEntries = [];
const surfaceEntries = [];
const stepEntries = [];
let hoverHold = false;
let trail = null;
let trailId = null;
function framePresent(F, i) {
  if (!F.x || !F.x.maskData || !F.x.maskData[i]) return false;
  if (!F.y || !F.y.maskData || !F.y.maskData[i]) return false;
  if (F.z && F.z.maskData && !F.z.maskData[i]) return false;
  if (F.size && F.size.maskData && !F.size.maskData[i]) return false;
  return true;
}
function applyBubbleHide(entry) {
  const L = entry.L;
  const nCats = (S.color && S.color.cats) ? S.color.cats.length : 0;
  for (let i = 0; i < L.n; i++) {
    let a = entry.baseAlpha[i];
    if (L._ci && nCats && hiddenCats.has(L._ci[i] % nCats)) a = 0;
    entry.aAlpha[i] = a;
  }
  entry.geom.attributes.aAlpha.needsUpdate = true;
}
function addBubbleLayer(L, ext, sizeAtten) {
  const n = L.n;
  const cols = layerColors(L, hex2rgb(T.cat[0]));
  const pos = new Float32Array(n * 3);
  const aSize = new Float32Array(n);
  const aAlpha = new Float32Array(n);
  const baseAlpha = new Float32Array(n);
  L._x = new Float32Array(n);
  L._y = new Float32Array(n);
  L._z = L.z ? new Float32Array(n) : null;
  L._size = aSize;
  L._alpha = aAlpha;
  L._ci = (L.color && L.color.kind === 'cat' && L.color.data)
    ? new Uint16Array(L.color.data) : null;
  L._cnorm = (L.color && L.color.kind === 'num' && L.color.data)
    ? Float32Array.from(L.color.data, v => v / 65535) : null;
  for (let i = 0; i < n; i++) {
    const x = L.x.data[i], y = L.y.data[i];
    const z = L.z ? L.z.data[i] : 0;
    L._x[i] = x; L._y[i] = y;
    if (L._z) L._z[i] = z;
    pos[i * 3] = x * ext[0];
    pos[i * 3 + 1] = y * ext[1];
    pos[i * 3 + 2] = z * ext[2];
    let sz = (typeof L.size === 'number') ? L.size : 6;
    if (L.size && L.size.data) sz = L.size.data[i] * L.size.max;
    aSize[i] = sz;
    baseAlpha[i] = 1;
    aAlpha[i] = 1;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  g.setAttribute('aSize', new THREE.BufferAttribute(aSize, 1));
  g.setAttribute('aAlpha', new THREE.BufferAttribute(aAlpha, 1));
  const pt = new THREE.Points(g, bubbleMaterial(L.alpha, sizeAtten));
  scene.add(pt);
  const entry = { L, ext, pos, cols, aSize, aAlpha, baseAlpha, geom: g };
  bubbleEntries.push(entry);
  applyBubbleHide(entry);
  return entry;
}
// playT is 0 at the first keyframe and nFrames-1 at the last. For a time
// column that position is linear in the column value, not in the keyframe index.
function frameSample(t) {
  const tr = S.transition;
  const nF = tr.nFrames;
  const times = tr.times || [];
  if (tr.type === 'time' && times.length === nF && nF > 1) {
    const t0 = +times[0];
    const t1 = +times[nF - 1];
    const span = t1 - t0;
    const T = span === 0 ? t0 : t0 + (t / (nF - 1)) * span;
    let f0 = 0;
    for (let i = 0; i < nF - 1; i++) {
      if (+times[i + 1] <= T + 1e-9) f0 = i + 1;
      else break;
    }
    if (f0 >= nF - 1) return { f0: nF - 1, f1: nF - 1, u: 0, T: t1 };
    const a = +times[f0], b = +times[f0 + 1];
    const u = b === a ? 0 : Math.max(0, Math.min(1, (T - a) / (b - a)));
    return { f0: f0, f1: f0 + 1, u: u, T: T };
  }
  let f0 = Math.floor(t);
  if (f0 < 0) f0 = 0;
  if (f0 >= nF) f0 = nF - 1;
  const f1 = Math.min(nF - 1, f0 + 1);
  const u = f0 === f1 ? 0 : (t - f0);
  return { f0: f0, f1: f1, u: u, T: null };
}
function writeBubbleFrame(entry, t) {
  const L = entry.L;
  const F = L.frames;
  if (!F || !S.transition) return;
  const nF = F.nFrames;
  const n = L.n;
  const sample = frameSample(t);
  const f0 = sample.f0;
  const f1 = sample.f1;
  let u = sample.u;
  if (S.transition.ease === 'smooth') u = u * u * (3 - 2 * u);
  const ext = entry.ext;
  const pos = entry.pos;
  const col = entry.cols;
  let colorDirty = false;
  for (let obj = 0; obj < n; obj++) {
    const i0 = obj * nF + f0;
    const i1 = obj * nF + f1;
    const p0 = framePresent(F, i0);
    const p1 = framePresent(F, i1);
    let x = 0, y = 0, z = 0, alpha = 0;
    let sz = (typeof L.size === 'number') ? L.size : 6;
    if (p0 && p1) {
      x = F.x.data[i0] * (1 - u) + F.x.data[i1] * u;
      y = F.y.data[i0] * (1 - u) + F.y.data[i1] * u;
      if (F.z) z = F.z.data[i0] * (1 - u) + F.z.data[i1] * u;
      alpha = 1;
      if (F.size) sz = (F.size.data[i0] * (1 - u) + F.size.data[i1] * u) * F.size.max;
    } else if (p0 || p1) {
      const i = p0 ? i0 : i1;
      x = F.x.data[i];
      y = F.y.data[i];
      if (F.z) z = F.z.data[i];
      alpha = p0 ? (1 - u) : u;
      if (F.size) sz = F.size.data[i] * F.size.max;
    }
    L._x[obj] = x; L._y[obj] = y;
    if (L._z) L._z[obj] = z;
    pos[obj * 3] = x * ext[0];
    pos[obj * 3 + 1] = y * ext[1];
    pos[obj * 3 + 2] = z * ext[2];
    entry.aSize[obj] = alpha > 0 ? sz : 0;
    entry.baseAlpha[obj] = alpha;
    if (F.color && F.color.kind === 'cat' && F.color.data) {
      const miss = 65535;
      const c0 = F.color.data[i0], c1 = F.color.data[i1];
      let code = miss;
      if (c0 !== miss && c1 !== miss) code = u < 0.5 ? c0 : c1;
      else if (c0 !== miss) code = c0;
      else if (c1 !== miss) code = c1;
      if (code === miss) entry.baseAlpha[obj] = 0;
      else if (L._ci) L._ci[obj] = code;
      if (code !== miss && PAL.length) {
        const p = PAL[code % PAL.length];
        col[obj * 3] = p[0]; col[obj * 3 + 1] = p[1]; col[obj * 3 + 2] = p[2];
        colorDirty = true;
      }
    } else if (F.color && F.color.kind === 'num' && F.color.data) {
      const m0 = !F.color.maskData || F.color.maskData[i0];
      const m1 = !F.color.maskData || F.color.maskData[i1];
      let tcol = null;
      if (m0 && m1) tcol = F.color.data[i0] * (1 - u) + F.color.data[i1] * u;
      else if (m0) tcol = F.color.data[i0];
      else if (m1) tcol = F.color.data[i1];
      if (tcol != null && L._cnorm) L._cnorm[obj] = tcol;
      if (tcol != null && RAMP.length) {
        const p = rampAt(tcol);
        col[obj * 3] = p[0]; col[obj * 3 + 1] = p[1]; col[obj * 3 + 2] = p[2];
        colorDirty = true;
      }
    }
  }
  entry.geom.attributes.position.needsUpdate = true;
  entry.geom.attributes.aSize.needsUpdate = true;
  if (colorDirty) entry.geom.attributes.color.needsUpdate = true;
  applyBubbleHide(entry);
}
function frameBlend(t) {
  const sample = frameSample(t);
  let u = sample.u;
  if (S.transition && S.transition.ease === 'smooth') u = u * u * (3 - 2 * u);
  return { f0: sample.f0, f1: sample.f1, u: u };
}
function writeLineFrame(entry, t) {
  const L = entry.L;
  const F = L.frames;
  if (!F || !F.x || !S.transition) return;
  const nF = F.nFrames;
  const n = L.n;
  const blend = frameBlend(t);
  const f0 = blend.f0, f1 = blend.f1, u = blend.u;
  const ext = entry.ext;
  const hasZ = !!entry.hasZ;
  for (let i = 0; i < n; i++) {
    const i0 = i * nF + f0;
    const i1 = i * nF + f1;
    L._x[i] = F.x.data[i0] * (1 - u) + F.x.data[i1] * u;
    L._y[i] = F.y ? (F.y.data[i0] * (1 - u) + F.y.data[i1] * u) : L.y.data[i];
    if (hasZ && F.z) L._z[i] = F.z.data[i0] * (1 - u) + F.z.data[i1] * u;
  }
  for (let s = 0; s < entry.segs.length; s++) {
    const seg = entry.segs[s];
    const flat = new Float32Array(seg.cnt * 3);
    for (let k = 0; k < seg.cnt; k++) {
      const i = seg.s0 + k;
      flat[k * 3] = L._x[i] * ext[0];
      flat[k * 3 + 1] = L._y[i] * ext[1];
      flat[k * 3 + 2] = hasZ ? L._z[i] * ext[2] : 0;
    }
    seg.geom.setPositions(Array.from(flat));
  }
}
function writeSurfaceFrame(entry, t) {
  const L = entry.L;
  const F = L.frames;
  if (!F || !S.transition) return;
  const nF = F.nFrames;
  const n = L.n;
  const blend = frameBlend(t);
  const f0 = blend.f0, f1 = blend.f1, u = blend.u;
  const ext = entry.ext;
  const pos = entry.pos;
  for (let i = 0; i < n; i++) {
    const i0 = i * nF + f0;
    const i1 = i * nF + f1;
    const x = F.x ? F.x.data[i0] * (1 - u) + F.x.data[i1] * u : L.x.data[i];
    const y = F.y ? F.y.data[i0] * (1 - u) + F.y.data[i1] * u : L.y.data[i];
    const z = F.z ? F.z.data[i0] * (1 - u) + F.z.data[i1] * u : L.z.data[i];
    L._x[i] = x;
    L._y[i] = y;
    L._z[i] = z;
    pos[i * 3] = x * ext[0];
    pos[i * 3 + 1] = y * ext[1];
    pos[i * 3 + 2] = z * ext[2];
  }
  entry.geom.attributes.position.needsUpdate = true;
  entry.geom.computeVertexNormals();
  if (entry.wire) {
    const next = new THREE.WireframeGeometry(entry.geom);
    entry.wire.geometry.dispose();
    entry.wire.geometry = next;
  }
}
function writeStepIndex(entry, f) {
  const bucket = entry.bucket;
  for (let i = 0; i < bucket.length; i++) {
    const on = i === f;
    for (let s = 0; s < bucket[i].length; s++) bucket[i][s].visible = on;
  }
  const L = entry.L;
  const F = L.frames;
  const span = F.spans[f];
  const start = span[0], count = span[1];
  L._stepN = count;
  if (!L._x || L._x.length !== count) {
    L._x = new Float32Array(count);
    L._y = new Float32Array(count);
  }
  for (let i = 0; i < count; i++) {
    L._x[i] = F.x.data[start + i];
    L._y[i] = F.y.data[start + i];
  }
}
function writeStepFrame(entry, t) {
  const sample = frameSample(t);
  let f = sample.u >= 0.5 ? sample.f1 : sample.f0;
  if (sample.f0 === sample.f1) f = sample.f0;
  writeStepIndex(entry, f);
}
function sliderAxes() {
  const params = (S.slider && S.slider.params) || [];
  const axes = [];
  for (let p = 0; p < params.length; p++) {
    const spec = params[p];
    const node = document.getElementById('slider-' + spec.name);
    const u = node ? (+node.value) / 1000 : 0;
    const n = spec.n;
    const span = Math.max(0, n - 1);
    const t = u * span;
    let i0 = Math.floor(t);
    if (i0 >= n - 1) i0 = n - 1;
    if (i0 < 0) i0 = 0;
    const i1 = Math.min(n - 1, i0 + 1);
    const f = (i0 === i1) ? 0 : (t - i0);
    const value = (+spec.lo) + u * ((+spec.hi) - (+spec.lo));
    axes.push({ spec: spec, i0: i0, i1: i1, f: f, value: value });
  }
  return axes;
}
function sliderStepIndex(axes) {
  let index = 0;
  for (let b = 0; b < axes.length; b++) {
    const ax = axes[b];
    const i = ax.f >= 0.5 ? ax.i1 : ax.i0;
    index += i * ax.spec.stride;
  }
  return index;
}
function sliderCorners(axes) {
  const n = axes.length;
  const corners = [];
  const limit = 1 << n;
  for (let mask = 0; mask < limit; mask++) {
    let index = 0;
    let w = 1;
    for (let b = 0; b < n; b++) {
      const bit = (mask >> b) & 1;
      const ax = axes[b];
      index += (bit ? ax.i1 : ax.i0) * ax.spec.stride;
      w *= bit ? ax.f : (1 - ax.f);
    }
    if (w < 1e-8) continue;
    corners.push({ index: index, w: w });
  }
  if (!corners.length) corners.push({ index: sliderStepIndex(axes), w: 1 });
  return corners;
}
function writeLineCorners(entry, corners) {
  const L = entry.L;
  const F = L.frames;
  if (!F || !F.x) return;
  const nF = F.nFrames;
  const n = L.n;
  const ext = entry.ext;
  const hasZ = !!entry.hasZ;
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0, z = 0;
    const base = i * nF;
    for (let c = 0; c < corners.length; c++) {
      const w = corners[c].w;
      const idx = base + corners[c].index;
      x += w * F.x.data[idx];
      if (F.y) y += w * F.y.data[idx];
      if (hasZ && F.z) z += w * F.z.data[idx];
    }
    L._x[i] = x;
    L._y[i] = F.y ? y : L.y.data[i];
    if (hasZ && F.z) L._z[i] = z;
  }
  for (let s = 0; s < entry.segs.length; s++) {
    const seg = entry.segs[s];
    const flat = new Float32Array(seg.cnt * 3);
    for (let k = 0; k < seg.cnt; k++) {
      const i = seg.s0 + k;
      flat[k * 3] = L._x[i] * ext[0];
      flat[k * 3 + 1] = L._y[i] * ext[1];
      flat[k * 3 + 2] = hasZ ? L._z[i] * ext[2] : 0;
    }
    seg.geom.setPositions(Array.from(flat));
  }
}
function writeSurfaceCorners(entry, corners) {
  const L = entry.L;
  const F = L.frames;
  if (!F) return;
  const nF = F.nFrames;
  const n = L.n;
  const ext = entry.ext;
  const pos = entry.pos;
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0, z = 0;
    let wx = 0, wy = 0, wz = 0;
    const base = i * nF;
    for (let c = 0; c < corners.length; c++) {
      const w = corners[c].w;
      const idx = base + corners[c].index;
      if (F.x) { x += w * F.x.data[idx]; wx += w; }
      if (F.y) { y += w * F.y.data[idx]; wy += w; }
      if (F.z) { z += w * F.z.data[idx]; wz += w; }
    }
    if (wx) L._x[i] = x; else L._x[i] = L.x.data[i];
    if (wy) L._y[i] = y; else L._y[i] = L.y.data[i];
    if (wz) L._z[i] = z; else L._z[i] = L.z.data[i];
    pos[i * 3] = L._x[i] * ext[0];
    pos[i * 3 + 1] = L._y[i] * ext[1];
    pos[i * 3 + 2] = L._z[i] * ext[2];
  }
  entry.geom.attributes.position.needsUpdate = true;
  entry.geom.computeVertexNormals();
  if (entry.wire) {
    const next = new THREE.WireframeGeometry(entry.geom);
    entry.wire.geometry.dispose();
    entry.wire.geometry = next;
  }
}
function addTweenLines(L, ext, cols) {
  const n = L.n;
  const hasZ = !!(L.z && L.z.data);
  L._x = new Float32Array(L.x.data);
  L._y = new Float32Array(L.y.data);
  if (hasZ) L._z = new Float32Array(L.z.data);
  const segs = [];
  const groups = L.groups || [[0, n]];
  for (let g = 0; g < groups.length; g++) {
    const s0 = groups[g][0], cnt = groups[g][1];
    if (cnt < 2) continue;
    const flat = new Float32Array(cnt * 3);
    for (let i = 0; i < cnt; i++) {
      const j = s0 + i;
      flat[i * 3] = L.x.data[j] * ext[0];
      flat[i * 3 + 1] = L.y.data[j] * ext[1];
      flat[i * 3 + 2] = hasZ ? L.z.data[j] * ext[2] : 0;
    }
    const lg = new LineGeometry();
    lg.setPositions(Array.from(flat));
    const lm = new LineMaterial({
      color: new THREE.Color(cols[s0 * 3], cols[s0 * 3 + 1], cols[s0 * 3 + 2]).getHex(),
      linewidth: L.linewidth || 2,
      worldUnits: false,
      transparent: true,
      opacity: L.alpha == null ? 1 : L.alpha
    });
    lineMats.push(lm);
    scene.add(new Line2(lg, lm));
    segs.push({ geom: lg, s0: s0, cnt: cnt });
  }
  lineEntries.push({ L: L, ext: ext, segs: segs, hasZ: hasZ });
}
function addStepLayer(L, ext, cols) {
  const F = L.frames;
  const rgb = [cols[0], cols[1], cols[2]];
  const bucket = [];
  for (let f = 0; f < F.nFrames; f++) {
    const span = F.spans[f];
    const start = span[0], count = span[1];
    const groups = (F.groups && F.groups[f]) || [[0, count]];
    const segs = [];
    for (let g = 0; g < groups.length; g++) {
      const gs = groups[g][0], gc = groups[g][1];
      if (gc < 2) continue;
      const flat = new Float32Array(gc * 3);
      for (let i = 0; i < gc; i++) {
        const j = start + gs + i;
        flat[i * 3] = F.x.data[j] * ext[0];
        flat[i * 3 + 1] = F.y.data[j] * ext[1];
        flat[i * 3 + 2] = 0;
      }
      const lg = new LineGeometry();
      lg.setPositions(Array.from(flat));
      const lm = new LineMaterial({
        color: new THREE.Color(rgb[0], rgb[1], rgb[2]).getHex(),
        linewidth: L.linewidth || 2,
        worldUnits: false,
        transparent: true,
        opacity: L.alpha == null ? 1 : L.alpha
      });
      lineMats.push(lm);
      const ln = new Line2(lg, lm);
      ln.visible = false;
      scene.add(ln);
      segs.push(ln);
    }
    bucket.push(segs);
  }
  stepEntries.push({ L: L, bucket: bucket, ext: ext });
}
function registerSurface(L, ext, geom, pos, wire) {
  L._x = new Float32Array(L.x.data);
  L._y = new Float32Array(L.y.data);
  L._z = new Float32Array(L.z.data);
  surfaceEntries.push({ L: L, ext: ext, geom: geom, pos: pos, wire: wire });
}
function clearTrail() {
  if (trail) {
    scene.remove(trail);
    const mat = trail.userData ? trail.userData.mat : null;
    if (mat) {
      const idx = lineMats.indexOf(mat);
      if (idx >= 0) lineMats.splice(idx, 1);
    }
    trail = null;
  }
  trailId = null;
}
function setTrail(L, obj) {
  clearTrail();
  if (!L || obj == null || !L.frames) return;
  const F = L.frames;
  const nF = F.nFrames;
  const entry = bubbleEntries.find(e => e.L === L);
  const ext = entry ? entry.ext : [1, 1, 1];
  const runs = [];
  let run = [];
  for (let f = 0; f < nF; f++) {
    const i = obj * nF + f;
    if (!framePresent(F, i)) {
      if (run.length >= 6) runs.push(run);
      run = [];
      continue;
    }
    const z = F.z ? F.z.data[i] : 0;
    run.push(F.x.data[i] * ext[0], F.y.data[i] * ext[1], z * ext[2]);
  }
  if (run.length >= 6) runs.push(run);
  if (!runs.length) return;
  let rgb = hex2rgb(T.ink);
  if (entry) rgb = [entry.cols[obj * 3], entry.cols[obj * 3 + 1], entry.cols[obj * 3 + 2]];
  const lm = new LineMaterial({
    color: new THREE.Color(rgb[0], rgb[1], rgb[2]).getHex(),
    linewidth: 2, worldUnits: false, transparent: true, opacity: 0.85,
  });
  const w = renderer.domElement.width || 800;
  const h = renderer.domElement.height || 600;
  lm.resolution.set(w, h);
  lineMats.push(lm);
  const group = new THREE.Group();
  group.userData.mat = lm;
  for (let r = 0; r < runs.length; r++) {
    const lg = new LineGeometry();
    lg.setPositions(runs[r]);
    group.add(new Line2(lg, lm));
  }
  scene.add(group);
  trail = group;
  trailId = String(obj) + '|' + (L.ids ? L.ids[obj] : '');
}
let frameLabel = '';
const axisTitleSprites = [];
let refreshFrameLabels = () => {};
function withFrame(text) {
  const s = text == null ? '' : String(text);
  if (s.indexOf('{frame_time}') < 0) return s;
  return s.split('{frame_time}').join(frameLabel);
}
function fmtParam(v) {
  if (!isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 1e6 || (a > 0 && a < 1e-3)) return v.toExponential(2);
  return v.toFixed(2);
}
function frameText(t) {
  const tr = S.transition;
  if (tr.params && tr.params.length) {
    const nF = tr.nFrames;
    const u = nF <= 1 ? 0 : t / (nF - 1);
    const parts = [];
    for (let i = 0; i < tr.params.length; i++) {
      const p = tr.params[i];
      const v = (+p.lo) + u * ((+p.hi) - (+p.lo));
      parts.push(p.name + ' = ' + fmtParam(v));
    }
    return parts.join(', ');
  }
  const sample = frameSample(t);
  if (tr.type === 'states') {
    return String(sample.u < 0.5 ? tr.times[sample.f0] : tr.times[sample.f1]);
  }
  if (sample.T != null) {
    if (tr.integer) return String(Math.round(sample.T));
    return fmt(sample.T);
  }
  const a = +tr.times[sample.f0], b = +tr.times[sample.f1];
  const v = a + (b - a) * sample.u;
  if (tr.integer) return String(Math.round(v));
  return fmt(v);
}
let playT = 0;
let playing = false;
let playSpeed = 1;
let recording = false;
let mediaRec = null;
function installSliders(renderFrame) {
  const spec = S.slider;
  const params = (spec && spec.params) || [];
  const player = document.getElementById('player');
  const yearEl = document.getElementById('year');
  player.style.display = 'flex';
  player.style.color = T.ink2;
  yearEl.style.display = 'none';
  document.getElementById('play-btn').style.display = 'none';
  document.getElementById('play-range').style.display = 'none';
  document.getElementById('play-readout').style.display = 'none';
  const speed = document.getElementById('play-speed');
  if (speed && speed.parentElement) speed.parentElement.style.display = 'none';
  const readouts = [];
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    const lab = document.createElement('label');
    lab.className = 'plot3-slider';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '1000';
    const savedU = SAVED && SAVED.sliders ? SAVED.sliders[p.name] : null;
    input.value = (savedU == null || !isFinite(+savedU)) ? '0' : String(savedU);
    input.id = 'slider-' + p.name;
    input.setAttribute('aria-label', p.name);
    const out = document.createElement('span');
    out.className = 'plot3-slider-readout';
    lab.appendChild(input);
    lab.appendChild(out);
    player.appendChild(lab);
    readouts.push(out);
    input.addEventListener('input', () => paint());
  }
  let shownLab = null;
  function paint() {
    const axes = sliderAxes();
    const corners = sliderCorners(axes);
    for (const e of lineEntries) writeLineCorners(e, corners);
    for (const e of surfaceEntries) writeSurfaceCorners(e, corners);
    const stepIndex = sliderStepIndex(axes);
    for (const e of stepEntries) writeStepIndex(e, stepIndex);
    const parts = [];
    for (let i = 0; i < axes.length; i++) {
      const text = axes[i].spec.name + ' = ' + fmtParam(axes[i].value);
      parts.push(text);
      if (readouts[i]) readouts[i].textContent = text;
    }
    const lab = parts.join(', ');
    frameLabel = lab;
    if (lab !== shownLab) {
      shownLab = lab;
      refreshFrameLabels();
      const titleTemplate = (S.labs && S.labs.title) || '';
      if (titleTemplate.indexOf('{frame_time}') >= 0) {
        const next = titleTemplate.split('{frame_time}').join(lab);
        if (S.labsMath && S.labsMath.title) {
          const segs = S.labsMath.title.map(p => ({
            latex: p.latex,
            text: String(p.text == null ? '' : p.text).split('{frame_time}').join(lab),
          }));
          titleEl.innerHTML = plot3MathHTML(segs, next);
          plot3Typeset(titleEl);
        } else {
          titleEl.textContent = next;
        }
      }
    }
    renderFrame();
  }
  paint();
}
function installPlayer(renderFrame) {
  window.__plot3.afterLegend = () => {
    for (const e of bubbleEntries) applyBubbleHide(e);
  };
  if (S.slider) { installSliders(renderFrame); return; }
  if (!S.transition) return;
  const tr = S.transition;
  const nF = tr.nFrames;
  playT = Math.max(0, nF - 1);
  const reduced = window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  playing = !reduced && nF > 1;
  // A paused chart shows the last keyframe. Playback starts at the first one,
  // so the opening frame is not immediately wrapped away.
  if (playing) playT = 0;
  if (SAVED && typeof SAVED.playT === 'number' && isFinite(SAVED.playT)) {
    playT = Math.max(0, Math.min(nF - 1, SAVED.playT));
    playing = false;
  }
  const player = document.getElementById('player');
  const btn = document.getElementById('play-btn');
  const range = document.getElementById('play-range');
  const readout = document.getElementById('play-readout');
  const speed = document.getElementById('play-speed');
  const yearEl = document.getElementById('year');
  player.style.display = 'flex';
  player.style.color = T.ink2;
  btn.style.background = T.surface;
  btn.style.color = T.ink;
  btn.style.border = '1px solid ' + T.axis;
  yearEl.style.display = 'block';
  yearEl.style.color = T.ink;
  const titleTemplate = (S.labs && S.labs.title) || '';
  let shownLab = null;
  function syncChrome() {
    const lab = frameText(playT);
    frameLabel = lab;
    readout.textContent = lab;
    yearEl.textContent = lab;
    yearEl.style.fontSize = lab.length > 8 ? '56px' : 'min(22vw, 148px)';
    range.value = (nF <= 1) ? '1000' : String(Math.round(playT / (nF - 1) * 1000));
    btn.textContent = playing ? 'Pause' : 'Play';
    btn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    if (lab === shownLab) return;
    shownLab = lab;
    refreshFrameLabels();
    if (titleTemplate.indexOf('{frame_time}') < 0) return;
    const next = titleTemplate.split('{frame_time}').join(lab);
    if (S.labsMath && S.labsMath.title) {
      const segs = S.labsMath.title.map(p => ({
        latex: p.latex,
        text: String(p.text == null ? '' : p.text).split('{frame_time}').join(lab),
      }));
      titleEl.innerHTML = plot3MathHTML(segs, next);
      plot3Typeset(titleEl);
    } else {
      titleEl.textContent = next;
    }
  }
  function paint() {
    for (const e of bubbleEntries) if (e.L.frames) writeBubbleFrame(e, playT);
    for (const e of lineEntries) writeLineFrame(e, playT);
    for (const e of surfaceEntries) writeSurfaceFrame(e, playT);
    for (const e of stepEntries) writeStepFrame(e, playT);
    syncChrome();
    renderFrame();
  }
  paint();
  btn.addEventListener('click', () => {
    playing = !playing;
    if (playing && nF > 1 && playT >= nF - 1) playT = 0;
    syncChrome();
  });
  range.addEventListener('input', () => {
    playing = false;
    const u = (+range.value) / 1000;
    playT = (nF <= 1) ? 0 : u * (nF - 1);
    paint();
  });
  speed.addEventListener('input', () => { playSpeed = +speed.value || 1; });
  window.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'BUTTON') return;
    if (e.code === 'Space') {
      e.preventDefault();
      playing = !playing;
      syncChrome();
    } else if (e.code === 'ArrowRight' || e.code === 'ArrowLeft') {
      e.preventDefault();
      playing = false;
      const dir = e.code === 'ArrowRight' ? 1 : -1;
      const trTimes = tr.times || [];
      if (tr.type === 'time' && trTimes.length === nF && nF > 1) {
        const s = frameSample(playT);
        let idx = s.f0;
        if (dir > 0) idx = s.u > 1e-4 ? s.f1 : Math.min(nF - 1, s.f0 + 1);
        else idx = s.u > 1e-4 ? s.f0 : Math.max(0, s.f0 - 1);
        const t0 = +trTimes[0], t1 = +trTimes[nF - 1];
        const span = t1 - t0;
        playT = span === 0 ? 0 : ((+trTimes[idx] - t0) / span) * (nF - 1);
      } else {
        playT = Math.max(0, Math.min(nF - 1, Math.round(playT) + dir));
      }
      paint();
    }
  });
  window.__plot3.startRecording = () => new Promise((resolve, reject) => {
    if (recording) { reject(new Error('Already recording')); return; }
    if (typeof MediaRecorder === 'undefined' || !renderer.domElement.captureStream) {
      reject(new Error('Video recording is not available in this browser'));
      return;
    }
    let mime = 'video/webm';
    if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) mime = 'video/webm;codecs=vp9';
    else if (!MediaRecorder.isTypeSupported('video/webm')) {
      reject(new Error('WebM recording is not available in this browser'));
      return;
    }
    const stream = renderer.domElement.captureStream(30);
    const rec = new MediaRecorder(stream, { mimeType: mime });
    const chunks = [];
    rec.ondataavailable = (ev) => { if (ev.data && ev.data.size) chunks.push(ev.data); };
    rec.onstop = () => {
      recording = false;
      resolve(new Blob(chunks, { type: 'video/webm' }));
    };
    rec.onerror = () => {
      recording = false;
      reject(new Error('Video recording failed'));
    };
    mediaRec = rec;
    recording = true;
    playing = true;
    playT = 0;
    rec.start();
    paint();
  });
  let last = 0;
  function loop(now) {
    if (!last) last = now;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (playing && !hoverHold && nF > 1) {
      const dur = (tr.duration || 12) / (playSpeed || 1);
      playT += dt * (nF - 1) / dur;
      if (playT >= nF - 1) {
        if (recording) {
          playT = nF - 1;
          playing = false;
          if (mediaRec && mediaRec.state === 'recording') mediaRec.stop();
        } else {
          playT = playT % (nF - 1);
        }
      }
      paint();
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}
window.__plot3.afterLegend = () => {
  for (const e of bubbleEntries) applyBubbleHide(e);
};

let renderNow = () => {};

if (!S.is3d) {
  // ═════════════════════════ 2D: ortho + pan/zoom ═════════════════════════
  // theme_void: no tick labels or axis titles to make room for.
  const VOID2 = !!T.void;
  const M = VOID2 ? { l: 12, r: 12, t: 30, b: 12 } : { l: 58, r: 12, t: 30, b: 40 };
  if (S.labs.subtitle) M.t += 16;
  const X_ANGLE = THEME_OPTS.xAngle || 0;
  if (X_ANGLE > 0 && S.scales.x) {
    // Room for turned x labels (about 6.5 px per character at 11 px).
    const labels = S.scales.x.kind === 'cat' ? S.scales.x.cats : (S.scales.x.ticks || []).map(t => t[1]);
    const longest = Math.max(0, ...labels.map(l => String(l).length)) * 6.5;
    M.b += Math.max(0, longest * Math.sin(X_ANGLE * Math.PI / 180) - 8);
  }
  let W = 100, H = 100;
  // coord_cartesian(expand=False): the limits are the panel's edges.
  const PAD2 = (S.coord && S.coord.expand === false) ? 0 : 0.03;
  const cam = new THREE.OrthographicCamera(-PAD2, 1 + PAD2, 1 + PAD2, -PAD2, -10, 10);
  // Equal aspect: one data unit has the same length on x and y. The panel
  // stays the cell's shape; the camera shows extra range on the looser axis.
  const coord2d = S.coord || { aspect: 'data', ratio: 1 };
  const equalAspect = coord2d.aspect === 'equal';
  const equalRatio = coord2d.ratio > 0 ? coord2d.ratio : 1;
  let equalZoom = 1;
  let equalCx = 0.5;
  let equalCy = 0.5;

  function equalBase() {
    // Nx / Ny so pixels per x unit = ratio * pixels per y unit.
    const target = (W / Math.max(H, 1)) * (spanOf('y') / spanOf('x')) / equalRatio;
    const pad = PAD2;
    const need = 1 + 2 * pad;
    let Nx = need, Ny = need;
    if (Nx / Ny < target) Nx = Ny * target;
    else Ny = Nx / target;
    return { Nx: Nx, Ny: Ny };
  }
  function applyEqual() {
    const base = equalBase();
    const Nx = base.Nx * equalZoom;
    const Ny = base.Ny * equalZoom;
    cam.left = equalCx - Nx / 2;
    cam.right = equalCx + Nx / 2;
    cam.bottom = equalCy - Ny / 2;
    cam.top = equalCy + Ny / 2;
  }
  function readEqualFromCam() {
    const base = equalBase();
    equalCx = (cam.left + cam.right) / 2;
    equalCy = (cam.bottom + cam.top) / 2;
    equalZoom = base.Nx > 0 ? (cam.right - cam.left) / base.Nx : 1;
  }

  // Paint layers in the order they were added, like ggplot2. three.js draws
  // opaque objects before transparent ones, which would put an error bar
  // (opaque line) underneath the translucent bar it belongs on.
  let paintOrder = 0;
  const addToScene = scene.add.bind(scene);
  scene.add = (...objs) => {
    for (const o of objs) {
      o.traverse(c => {
        c.renderOrder = paintOrder;
        const mats = c.material ? (Array.isArray(c.material) ? c.material : [c.material]) : [];
        for (const m of mats) m.transparent = true;
      });
    }
    return addToScene(...objs);
  };
  for (const L of S.layers) {
    paintOrder += 1;
    const n = L.n;
    const cols = layerColors(L, hex2rgb(T.cat[0]));
    const isCat = L.color && L.color.kind === 'cat';
    if (L.kind === 'point' && (L.frames || (L.size && L.size.id))) {
      addBubbleLayer(L, [1, 1, 1], false);
    } else if (L.kind === 'point') {
      if (L.shape) {
        // One Points object per (category, symbol), each with its sprite.
        const k = isCat ? S.color.cats.length : 1;
        const buckets = new Map();
        for (let i = 0; i < n; i++) {
          const ci = isCat ? L.color.data[i] % k : 0;
          const key = ci + '|' + shapeOf(L, i);
          if (!buckets.has(key)) buckets.set(key, []);
          buckets.get(key).push(i);
        }
        for (const [key, idx] of buckets) {
          const [ciText, shape] = key.split('|');
          const ci = Number(ciText);
          const pos = new Float32Array(idx.length * 3);
          const col = new Float32Array(idx.length * 3);
          for (let j = 0; j < idx.length; j++) {
            const i = idx[j];
            pos[j*3] = L.x.data[i]; pos[j*3+1] = L.y.data[i]; pos[j*3+2] = 0;
            col[j*3] = cols[i*3]; col[j*3+1] = cols[i*3+1]; col[j*3+2] = cols[i*3+2];
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          g.setAttribute('color', new THREE.BufferAttribute(col, 3));
          const pt = new THREE.Points(g, new THREE.PointsMaterial({
            size: (L.size || 6) * 1.25, sizeAttenuation: false, vertexColors: true,
            map: shapeTexture(shape), alphaTest: 0.3,
            transparent: true, opacity: L.alpha }));
          scene.add(pt);
          if (isCat) regCat(ci, pt);
        }
      } else if (isCat) {
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
    } else if (L.kind === 'box') {
      // Tukey boxplot: body [lower,upper], median, whiskers [ymin,ymax], outliers.
      const hw = (L.width || 0.08) * 0.5;
      const cap = hw * 0.55;
      const pos = new Float32Array(n * 6 * 3);
      const col = new Float32Array(n * 6 * 3);
      // 9 segments × 2 endpoints × 3 = 54 floats per box
      const linePos = new Float32Array(n * 54);
      const lineCol = new Float32Array(n * 54);
      // aes(fill=): a filled box with a dark outline, whiskers, and median.
      const filled = !!L.fillMapped;
      const inkRGB = hex2rgb(T.ink2 || '#333333');
      let p = 0, c = 0, lp = 0, lc = 0;
      function pushSeg(x0,y0,x1,y1,r,g,b) {
        linePos[lp++]=x0; linePos[lp++]=y0; linePos[lp++]=0;
        linePos[lp++]=x1; linePos[lp++]=y1; linePos[lp++]=0;
        for (let k=0;k<2;k++){ lineCol[lc++]=r; lineCol[lc++]=g; lineCol[lc++]=b; }
      }
      for (let i = 0; i < n; i++) {
        const x = L.x.data[i];
        const ymin = L.ymin.data[i], lower = L.lower.data[i];
        const middle = L.middle.data[i], upper = L.upper.data[i];
        const ymax = L.ymax.data[i];
        const r = cols[i*3], gch = cols[i*3+1], b = cols[i*3+2];
        const x0 = x - hw, x1 = x + hw;
        // box body (white inside when the box has no colour of its own)
        const tri = [x0,lower,0, x1,lower,0, x1,upper,0, x0,lower,0, x1,upper,0, x0,upper,0];
        for (let k = 0; k < 18; k++) pos[p++] = tri[k];
        const [fr, fg, fb] = L.plainFill ? hex2rgb(T.surface) : [r, gch, b];
        for (let k = 0; k < 6; k++) { col[c++]=fr; col[c++]=fg; col[c++]=fb; }
        // whisker stem + caps + median, and the box outline
        const [lr, lg2, lb] = filled ? inkRGB : [r, gch, b];
        pushSeg(x, ymin, x, lower, lr, lg2, lb);
        pushSeg(x, upper, x, ymax, lr, lg2, lb);
        pushSeg(x - cap, ymin, x + cap, ymin, lr, lg2, lb);
        pushSeg(x - cap, ymax, x + cap, ymax, lr, lg2, lb);
        pushSeg(x0, middle, x1, middle, lr, lg2, lb);
        pushSeg(x0, lower, x0, upper, lr, lg2, lb);
        pushSeg(x1, lower, x1, upper, lr, lg2, lb);
        pushSeg(x0, lower, x1, lower, lr, lg2, lb);
        pushSeg(x0, upper, x1, upper, lr, lg2, lb);
      }
      const bg = new THREE.BufferGeometry();
      bg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      bg.setAttribute('color', new THREE.BufferAttribute(col, 3));
      scene.add(new THREE.Mesh(bg, new THREE.MeshBasicMaterial({
        vertexColors: true, transparent: true,
        opacity: L.plainFill ? 1 : filled ? Math.min(1, (L.alpha || 0.9) * 0.9) : Math.min(1, (L.alpha||0.9)*0.35),
        side: THREE.DoubleSide, depthWrite: false })));
      // Fat lines: WebGL draws plain lines 1 px wide whatever is asked.
      const lg = new LineSegmentsGeometry();
      lg.setPositions(linePos);
      lg.setColors(lineCol);
      const boxLine = new LineMaterial({
        vertexColors: true, linewidth: 1.4, worldUnits: false,
        transparent: true, opacity: L.alpha || 0.95 });
      lineMats.push(boxLine);
      scene.add(new LineSegments2(lg, boxLine));
      // outliers
      const nOut = L.nOut || 0;
      if (nOut > 0 && L.ox && L.oy) {
        const opos = new Float32Array(nOut * 3);
        const ocol = new Float32Array(nOut * 3);
        for (let i = 0; i < nOut; i++) {
          opos[i*3] = L.ox.data[i]; opos[i*3+1] = L.oy.data[i]; opos[i*3+2] = 0;
          if (L.ocolor) {
            if (L.ocolor.kind === 'cat') {
              const p = PAL[L.ocolor.data[i] % PAL.length];
              ocol[i*3]=p[0]; ocol[i*3+1]=p[1]; ocol[i*3+2]=p[2];
            } else {
              const p = rampAt(L.ocolor.data[i] / 65535);
              ocol[i*3]=p[0]; ocol[i*3+1]=p[1]; ocol[i*3+2]=p[2];
            }
          } else if (L.constColor) {
            const p = hex2rgb(L.constColor);
            ocol[i*3]=p[0]; ocol[i*3+1]=p[1]; ocol[i*3+2]=p[2];
          } else {
            ocol[i*3]=cols[0]; ocol[i*3+1]=cols[1]; ocol[i*3+2]=cols[2];
          }
        }
        const og = new THREE.BufferGeometry();
        og.setAttribute('position', new THREE.BufferAttribute(opos, 3));
        og.setAttribute('color', new THREE.BufferAttribute(ocol, 3));
        scene.add(new THREE.Points(og, new THREE.PointsMaterial({
          size: L.outlierSize || 3, sizeAttenuation: false, vertexColors: true,
          transparent: true, opacity: L.alpha || 0.9 })));
      }
    } else if (L.kind === 'area' || L.kind === 'poly') {
      // Filled ribbons (density) or closed polygons (violin).
      const y0 = (L.y0 != null) ? L.y0 : 0;
      for (const [s0, cnt] of (L.groups || [[0, n]])) {
        if (cnt < 2) continue;
        const r = cols[s0*3], gch = cols[s0*3+1], b = cols[s0*3+2];
        if (L.kind === 'area') {
          // Triangle strip under the curve down to baseline y0.
          const pos = new Float32Array((cnt - 1) * 6 * 3);
          const col = new Float32Array((cnt - 1) * 6 * 3);
          let p = 0, c = 0;
          for (let i = 0; i < cnt - 1; i++) {
            const i0 = s0 + i, i1 = s0 + i + 1;
            const x0 = L.x.data[i0], yA = L.y.data[i0];
            const x1 = L.x.data[i1], yB = L.y.data[i1];
            const tri = [x0,y0,0, x1,y0,0, x1,yB,0, x0,y0,0, x1,yB,0, x0,yA,0];
            for (let k = 0; k < 18; k++) pos[p++] = tri[k];
            for (let k = 0; k < 6; k++) { col[c++]=r; col[c++]=gch; col[c++]=b; }
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          g.setAttribute('color', new THREE.BufferAttribute(col, 3));
          const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
            vertexColors: true, transparent: true, opacity: L.alpha || 0.35,
            side: THREE.DoubleSide, depthWrite: false }));
          scene.add(mesh);
          if (isCat) regCat(L.color.data[s0] % S.color.cats.length, mesh);
        } else {
          // Violin (and closed polys): authoring order is left bottom→top then
          // right top→bottom. Fan-from-first folds that contour into leaf-like
          // fragments; pair left/right halves into a triangle strip instead.
          const half = cnt >> 1;
          const paired = half >= 2 && half * 2 === cnt;
          const triCount = paired
            ? Math.max(0, half - 1) * 2
            : Math.max(0, cnt - 2);
          const pos = new Float32Array(triCount * 9);
          const col = new Float32Array(triCount * 9);
          // A violin with no colour of its own is white inside (ggplot2).
          const [pr, pg, pb] = L.plainFill ? hex2rgb(T.surface) : [r, gch, b];
          let p = 0, c = 0;
          function pushTri(iA, iB, iC) {
            const tri = [
              L.x.data[iA], L.y.data[iA], 0,
              L.x.data[iB], L.y.data[iB], 0,
              L.x.data[iC], L.y.data[iC], 0,
            ];
            for (let k = 0; k < 9; k++) pos[p++] = tri[k];
            for (let k = 0; k < 3; k++) { col[c++]=pr; col[c++]=pg; col[c++]=pb; }
          }
          if (paired) {
            for (let i = 0; i < half - 1; i++) {
              const l0 = s0 + i, l1 = s0 + i + 1;
              // right side is stored top→bottom, so same-y pair is mirrored
              const r0 = s0 + cnt - 1 - i, r1 = s0 + cnt - 2 - i;
              pushTri(l0, r0, l1);
              pushTri(r0, r1, l1);
            }
          } else {
            for (let i = 1; i < cnt - 1; i++) pushTri(s0, s0 + i, s0 + i + 1);
          }
          const g = new THREE.BufferGeometry();
          g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
          g.setAttribute('color', new THREE.BufferAttribute(col, 3));
          const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
            vertexColors: true, transparent: true, opacity: L.plainFill ? 1 : (L.alpha || 0.45),
            side: THREE.DoubleSide, depthWrite: false }));
          scene.add(mesh);
          if (isCat) regCat(L.color.data[s0] % S.color.cats.length, mesh);
        }
        // Outline stroke
        const flat = new Float32Array(cnt * 3);
        for (let i = 0; i < cnt; i++) {
          flat[i*3] = L.x.data[s0+i];
          flat[i*3+1] = L.y.data[s0+i];
          flat[i*3+2] = 0;
        }
        const lg = new LineGeometry();
        lg.setPositions(Array.from(flat));
        const lm = new LineMaterial({
          color: new THREE.Color(r, gch, b).getHex(),
          linewidth: L.linewidth || 1.5, worldUnits: false,
          transparent: true, opacity: Math.min(1, (L.alpha || 0.9) + 0.3) });
        lineMats.push(lm);
        const ln = new Line2(lg, lm);
        scene.add(ln);
        if (isCat) regCat(L.color.data[s0] % S.color.cats.length, ln);
      }
    } else if (L.frames && L.frames.mode === 'step') {
      addStepLayer(L, [1, 1, 1], cols);
    } else if (L.frames && L.frames.x) {
      addTweenLines(L, [1, 1, 1], cols);
    } else {
      L.groups.forEach(([s0, cnt], gi) => {
        if (cnt < 2) return;
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
        const dash = (L.dashes && L.dashes[gi]) || L.dash;
        if (dash && dash.length >= 2) {
          // Dash lengths are in world units: size them from the pixels the
          // panel shows now (they scale with zoom).
          const wpp = (cam.right - cam.left) / Math.max(1, renderer.domElement.clientWidth || 800);
          const unit = Math.max(L.linewidth || 1, 1) * wpp;
          lm.dashed = true;
          lm.dashSize = dash[0] * unit;
          lm.gapSize = dash[1] * unit;
          lm.dashScale = 1;
          ln.computeLineDistances();
        }
        scene.add(ln);
        if (isCat) regCat(L.color.data[s0] % S.color.cats.length, ln);
      });
    }
  }
  scene.add = addToScene;

  function drawAxes() {
    const x0 = dataLo('x') + cam.left  * spanOf('x');
    const x1 = dataLo('x') + cam.right * spanOf('x');
    const y0 = dataLo('y') + cam.bottom * spanOf('y');
    const y1 = dataLo('y') + cam.top    * spanOf('y');
    const px = v => M.l + (v - x0) / (x1 - x0) * W;
    const py = v => M.t + H - (v - y0) / (y1 - y0) * H;
    let s = '', grid = '';
    // cap tick count by panel size so labels never collide
    const xt = VOID2 ? [] : thin(ticksFor('x', x0, x1), Math.max(5, Math.floor(W / 80)));
    const yt = VOID2 ? [] : thin(ticksFor('y', y0, y1), Math.max(5, Math.floor(H / 40)));
    const showGrid = THEME_OPTS.panelGrid !== false;
    for (const [t, lab] of xt) {
      const X = px(t);
      if (X < M.l - 1 || X > M.l + W + 1) continue;
      if (showGrid) grid += `<line x1="${X}" y1="${M.t}" x2="${X}" y2="${M.t+H}" stroke="${T.grid}"/>`;
      if (X_ANGLE > 0) {
        const ty = M.t + H + 8;
        s += `<text x="${X}" y="${ty}" fill="${T.muted}" text-anchor="end" dominant-baseline="middle" transform="rotate(${-X_ANGLE} ${X} ${ty})">${lab}</text>`;
      } else {
        s += `<text x="${X}" y="${M.t+H+14}" fill="${T.muted}" text-anchor="middle">${lab}</text>`;
      }
    }
    for (const [t, lab] of yt) {
      const Y = py(t);
      if (Y < M.t - 1 || Y > M.t + H + 1) continue;
      if (showGrid) grid += `<line x1="${M.l}" y1="${Y}" x2="${M.l+W}" y2="${Y}" stroke="${T.grid}"/>`;
      s += `<text x="${M.l-7}" y="${Y+4}" fill="${T.muted}" text-anchor="end">${lab}</text>`;
    }
    if (!VOID2) s += `<rect x="${M.l}" y="${M.t}" width="${W}" height="${H}" fill="none" stroke="${T.axis}"/>`;
    // geom_rug: a short tick at the panel edge for every value.
    const rugs = S.rugs || [];
    if (rugs.length) {
      s += `<clipPath id="p3rugclip"><rect x="${M.l}" y="${M.t}" width="${W}" height="${H}"/></clipPath><g clip-path="url(#p3rugclip)">`;
      for (const r of rugs) {
        const onX = r.side === 'b' || r.side === 't';
        const reach = (r.length || 0.03) * (onX ? H : W);
        const w = r.width || 0.75, a = r.alpha == null ? 1 : r.alpha;
        (r.values || []).forEach((v, i) => {
          const c = r.colors ? r.colors[i] : r.color;
          if (onX) {
            const X = px(v);
            if (X < M.l || X > M.l + W) return;
            const Y0 = r.side === 'b' ? M.t + H : M.t, Y1 = r.side === 'b' ? Y0 - reach : Y0 + reach;
            s += `<line x1="${X}" y1="${Y0}" x2="${X}" y2="${Y1}" stroke="${c}" stroke-width="${w}" stroke-opacity="${a}"/>`;
          } else {
            const Y = py(v);
            if (Y < M.t || Y > M.t + H) return;
            const X0 = r.side === 'l' ? M.l : M.l + W, X1 = r.side === 'l' ? X0 + reach : X0 - reach;
            s += `<line x1="${X0}" y1="${Y}" x2="${X1}" y2="${Y}" stroke="${c}" stroke-width="${w}" stroke-opacity="${a}"/>`;
          }
        });
      }
      s += '</g>';
    }
    // geom_hline / geom_vline / geom_abline, clipped to the panel so they
    // follow pan and zoom without spilling into the margins.
    const refs = S.refs || [];
    if (refs.length) {
      s += `<clipPath id="p3refclip"><rect x="${M.l}" y="${M.t}" width="${W}" height="${H}"/></clipPath><g clip-path="url(#p3refclip)">`;
      for (const r of refs) {
        let X1, Y1, X2, Y2;
        if (r.kind === 'hline') { Y1 = Y2 = py(r.value); X1 = M.l; X2 = M.l + W; }
        else if (r.kind === 'vline') { X1 = X2 = px(r.value); Y1 = M.t; Y2 = M.t + H; }
        else { X1 = px(x0); X2 = px(x1); Y1 = py(r.intercept + r.slope * x0); Y2 = py(r.intercept + r.slope * x1); }
        const w = r.width || 1;
        const dash = (r.dash && r.dash.length) ? ` stroke-dasharray="${r.dash.map(d => d * Math.max(w, 1)).join(' ')}"` : '';
        s += `<line x1="${X1}" y1="${Y1}" x2="${X2}" y2="${Y2}" stroke="${r.color}" stroke-width="${w}" stroke-opacity="${r.alpha == null ? 1 : r.alpha}"${dash}/>`;
      }
      s += '</g>';
    }
    // arrow() heads, in screen space so they keep their shape when zooming.
    for (const r of (S.arrows || [])) {
      const tx = px(r.x1), ty = py(r.y1), bx = px(r.x0), by = py(r.y0);
      const d = Math.hypot(tx - bx, ty - by);
      if (!(d > 1e-6)) continue;
      const ux = (tx - bx) / d, uy = (ty - by) / d, L = (r.length || 24);
      const pts = [1, -1].map(sg => {
        const a = (r.angle || 30) * Math.PI / 180, c = Math.cos(a), sn = sg * Math.sin(a);
        return [tx - (ux * c - uy * sn) * L, ty - (ux * sn + uy * c) * L];
      });
      const w = r.width || 1;
      if (r.type === 'closed') {
        s += `<polygon points="${pts[0].join(',')} ${tx},${ty} ${pts[1].join(',')}" fill="${r.color}" stroke="${r.color}" stroke-width="${w}"/>`;
      } else {
        for (const p of pts) s += `<line x1="${tx}" y1="${ty}" x2="${p[0]}" y2="${p[1]}" stroke="${r.color}" stroke-width="${w}" stroke-linecap="round"/>`;
      }
    }
    if (!S.facetChild && !VOID2) {
      // A facet panel leaves the shared axis titles to the figure around it.
      s += `<text x="${M.l+W/2}" y="${M.t+H+M.b-10}" fill="${T.ink2}" text-anchor="middle">${plot3Esc(withFrame(S.labs.x))}</text>`;
      s += `<text x="14" y="${M.t+H/2}" fill="${T.ink2}" text-anchor="middle" transform="rotate(-90 14 ${M.t+H/2})">${plot3Esc(withFrame(S.labs.y))}</text>`;
    }
    const anns = S.ann || [];
    const textBoxes = [];
    for (let i = 0; i < anns.length; i++) {
      const ann = anns[i];
      const X = px(ann.x), Y = py(ann.y);
      if (X < M.l - 2 || X > M.l + W + 2 || Y < M.t - 2 || Y > M.t + H + 2) continue;
      if (ann.style === 'text' || ann.style === 'label') {
        // geom_text / geom_label, anchored by hjust/vjust like ggplot2.
        // 3.88 mm (ggplot2's default) matches this viewer's base font.
        const size = (ann.size || 14.7) * (13.75 / 14.67);
        const w = 0.55 * size * String(ann.text).length, h = 1.2 * size;
        const hj = ann.hjust == null ? 0.5 : ann.hjust, vj = ann.vjust == null ? 0.5 : ann.vjust;
        const left = X - hj * w, top = Y - (1 - vj) * h;
        if (ann.overlap === false && textBoxes.some(b =>
            left < b[0] + b[2] && b[0] < left + w && top < b[1] + b[3] && b[1] < top + h)) continue;
        textBoxes.push([left, top, w, h]);
        const col = ann.color || T.ink;
        if (ann.style === 'label') {
          const pad = 0.25 * size;
          s += `<rect x="${left - pad}" y="${top - pad * 0.6}" width="${w + 2 * pad}" height="${h + pad * 1.2}" rx="3" fill="${T.surface}" stroke="${col}" stroke-width="0.8"/>`;
        }
        s += `<text x="${left + w / 2}" y="${top + h / 2}" fill="${col}" fill-opacity="${ann.alpha == null ? 1 : ann.alpha}" text-anchor="middle" dominant-baseline="middle" font-size="${size}" font-weight="${ann.weight || 400}"${ann.italic ? ' font-style="italic"' : ''}>${plot3Esc(ann.text)}</text>`;
        continue;
      }
      // Inside the panel, with a halo in the page colour so the label reads
      // where it crosses the curve (P(X >= 1.96) sits on a thin tail).
      const halfW = 0.29 * 12 * String(ann.text).length + 3;
      const AX = Math.min(Math.max(X, M.l + halfW + 2), M.l + W - halfW - 2);
      const AY = Math.min(Math.max(Y, M.t + 12), M.t + H - 10);
      s += '<text x="' + AX + '" y="' + AY + '" fill="' + T.ink + '" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="600" paint-order="stroke" stroke="' + T.surface + '" stroke-width="4" stroke-linejoin="round">' + plot3Esc(ann.text) + '</text>';
    }
    svg.innerHTML = s;
    if (gridSvg) gridSvg.innerHTML = grid;
  }

  function layout() {
    placeLegend();
    W = Math.max(50, figEl.clientWidth - M.l - M.r);
    H = Math.max(50, figEl.clientHeight - M.t - M.b);
    if (equalAspect) applyEqual();
    host.style.left = M.l + 'px'; host.style.top = M.t + 'px';
    renderer.setSize(W, H);
    for (const layer of [svg, gridSvg]) {
      if (!layer) continue;
      layer.setAttribute('width', figEl.clientWidth);
      layer.setAttribute('height', figEl.clientHeight);
    }
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
    if (equalAspect) {
      readEqualFromCam();
      if (equalZoom > 2.4) equalZoom = 2.4;
      if (equalZoom < 0.01) equalZoom = 0.01;
      const base = equalBase();
      const Nx = base.Nx * equalZoom;
      const Ny = base.Ny * equalZoom;
      // Keep a slice of the data range on screen. Spans stay locked together.
      const slack = 0.15;
      equalCx = Math.min((1 - slack) + Nx / 2, Math.max(slack - Nx / 2, equalCx));
      equalCy = Math.min((1 - slack) + Ny / 2, Math.max(slack - Ny / 2, equalCy));
      applyEqual();
      return;
    }
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
  let pointerMoved = false;
  el.addEventListener('pointerdown', e => {
    pointerMoved = false;
    dragging = { x: e.clientX, y: e.clientY,
                 l: cam.left, r: cam.right, t: cam.top, b: cam.bottom };
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener('pointerup', () => dragging = null);
  el.addEventListener('pointermove', e => {
    if (dragging) {
      if (Math.abs(e.clientX - dragging.x) + Math.abs(e.clientY - dragging.y) > 4)
        pointerMoved = true;
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
    if (equalAspect) {
      equalZoom = 1; equalCx = 0.5; equalCy = 0.5;
      applyEqual();
    } else {
      cam.left = -PAD2; cam.right = 1 + PAD2; cam.bottom = -PAD2; cam.top = 1 + PAD2;
    }
    draw();
  });

  let hoverTick = 0;
  function dataY(norm) { return fromScale('y', norm); }
  function dataX(norm) { return fromScale('x', norm); }
  function screenX(norm) {
    return (norm - cam.left) / (cam.right - cam.left) * W;
  }
  function screenY(norm) {
    return (cam.top - norm) / (cam.top - cam.bottom) * H;
  }
  function catHidden(L, i) {
    return L.color && L.color.kind === 'cat' &&
      hiddenCats.has(L.color.data[i] % S.color.cats.length);
  }
  function colorHead(L, i) {
    if (L.color && L.color.kind === 'cat') {
      const codes = L._ci || L.color.data;
      return '<b>' + S.color.cats[codes[i] % S.color.cats.length] + '</b><br>';
    }
    if (L.color && L.color.kind === 'num') {
      const t = L._cnorm ? L._cnorm[i] : (L.color.data[i] / 65535);
      return '<b>' + fmt(cval(t)) + '</b><br>';
    }
    return '';
  }
  function hover(e) {
    const now = performance.now();
    if (!S.transition && now - hoverTick < 33) return;
    hoverTick = now;
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    // best: {score, sx, sy, html} — lower score wins; 0 = solid hit
    let best = null;
    function consider(score, sx, sy, html, hold) {
      if (!best || score < best.score) best = { score, sx, sy, html, hold: !!hold };
    }
    for (const L of S.layers) {
      if (L.kind === 'col') {
        // Full bar rectangle (not just the top-center point).
        const hw = (L.width || 0.08) * 0.5;
        const y0 = (L.y0 != null) ? L.y0 : 0;
        for (let i = 0; i < L.n; i++) {
          if (catHidden(L, i)) continue;
          const x = L.x.data[i], y = L.y.data[i];
          const sx0 = screenX(x - hw), sx1 = screenX(x + hw);
          const yLo = Math.min(y, y0), yHi = Math.max(y, y0);
          const sy0 = screenY(yHi), sy1 = screenY(yLo);
          const left = Math.min(sx0, sx1), right = Math.max(sx0, sx1);
          const top = Math.min(sy0, sy1), bot = Math.max(sy0, sy1);
          if (mx < left || mx > right || my < top || my > bot) continue;
          const sx = (left + right) * 0.5, sy = (top + bot) * 0.5;
          const xv = dataX(x), yv = dataY(y);
          consider(0, sx, sy, colorHead(L, i)
            + fmtSpan('x', xv, (cam.right - cam.left) * spanOf('x')) + ', '
            + fmtSpan('y', yv, (cam.top - cam.bottom) * spanOf('y')));
        }
      } else if (L.kind === 'box') {
        // Hit full whisker band; tooltip shows all Tukey stats.
        const hw = (L.width || 0.08) * 0.5;
        for (let i = 0; i < L.n; i++) {
          if (catHidden(L, i)) continue;
          const x = L.x.data[i];
          const ymin = L.ymin.data[i], lower = L.lower.data[i];
          const middle = L.middle.data[i], upper = L.upper.data[i];
          const ymax = L.ymax.data[i];
          const sx0 = screenX(x - hw), sx1 = screenX(x + hw);
          const sy0 = screenY(Math.max(ymin, ymax));
          const sy1 = screenY(Math.min(ymin, ymax));
          const left = Math.min(sx0, sx1), right = Math.max(sx0, sx1);
          const top = Math.min(sy0, sy1), bot = Math.max(sy0, sy1);
          // slight pad so whisker caps are easy to hit
          if (mx < left - 4 || mx > right + 4 || my < top - 4 || my > bot + 4)
            continue;
          const sx = (left + right) * 0.5, sy = (top + bot) * 0.5;
          const ySpan = (cam.top - cam.bottom) * spanOf('y');
          const xv = dataX(x);
          const html = colorHead(L, i)
            + fmtSpan('x', xv, (cam.right - cam.left) * spanOf('x')) + '<br>'
            + 'min ' + fmtSpan('y', dataY(ymin), ySpan) + '<br>'
            + 'q1 ' + fmtSpan('y', dataY(lower), ySpan) + '<br>'
            + 'median ' + fmtSpan('y', dataY(middle), ySpan) + '<br>'
            + 'q3 ' + fmtSpan('y', dataY(upper), ySpan) + '<br>'
            + 'max ' + fmtSpan('y', dataY(ymax), ySpan);
          consider(0, sx, sy, html);
        }
      } else {
        // Points / line samples / poly vertices — nearest point within its radius.
        const xs = L._x || (L.x && L.x.data);
        const ys = L._y || (L.y && L.y.data);
        if (!xs || !ys) continue;
        let bestD = Infinity;
        const nHover = L._stepN != null ? L._stepN : L.n;
        for (let i = 0; i < nHover; i++) {
          if (L._alpha && L._alpha[i] <= 0.04) continue;
          if (L._ci && S.color.cats && hiddenCats.has(L._ci[i] % S.color.cats.length)) continue;
          if (!L._ci && catHidden(L, i)) continue;
          const sx = screenX(xs[i]);
          const sy = screenY(ys[i]);
          const rad = L._size ? Math.max(10, L._size[i] * 0.55) : 12;
          const limit = rad * rad;
          const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
          if (d <= limit && d < bestD) {
            bestD = d;
            const xv = dataX(xs[i]), yv = dataY(ys[i]);
            const xSpan = (cam.right - cam.left) * spanOf('x');
            const ySpan = (cam.top - cam.bottom) * spanOf('y');
            let head = '';
            if (L.ids && L.ids[i]) head += '<b>' + plot3Esc(L.ids[i]) + '</b><br>';
            head += colorHead(L, i);
            let sizeBit = '';
            if (L._size && L.size && L.size.vmax && L.size.max) {
              const frac = L._size[i] / L.size.max;
              const value = frac * frac * L.size.vmax;
              const name = (S.sizeLegend && S.sizeLegend.label) || 'size';
              sizeBit = '<br>' + plot3Esc(name) + ' = ' + fmt(value);
            }
            const html = L.tip
              ? head + formulaHead(L)
                + axisPair('x', fmtSpan('x', xv, xSpan)) + ', '
                + axisPair('y', fmtSpan('y', yv, ySpan)) + sizeBit
              : head
                + fmtSpan('x', xv, xSpan) + ', '
                + fmtSpan('y', yv, ySpan) + sizeBit;
            consider(d, sx, sy, html, !!(L.kind === 'point' && L.frames && S.transition));
          }
        }
      }
    }
    hoverHold = !!(best && best.hold);
    if (!best) { tip.style.display = 'none'; return; }
    tip.innerHTML = best.html;
    plot3Typeset(tip);
    tip.style.left = (M.l + best.sx + 12) + 'px';
    tip.style.top = (M.t + best.sy - 10) + 'px';
    tip.style.display = 'block';
  }
  el.addEventListener('pointerleave', () => {
    tip.style.display = 'none';
    hoverHold = false;
  });
  el.addEventListener('click', (e) => {
    if (!S.transition) return;
    if (pointerMoved) return;
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let hit = null, hitD = Infinity;
    for (const L of S.layers) {
      if (L.kind !== 'point' || !L.frames || !L._x) continue;
      for (let i = 0; i < L.n; i++) {
        if (L._alpha && L._alpha[i] <= 0.04) continue;
        const sx = screenX(L._x[i]), sy = screenY(L._y[i]);
        const rad = L._size ? Math.max(10, L._size[i] * 0.55) : 12;
        const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
        if (d <= rad * rad && d < hitD) { hitD = d; hit = { L, i }; }
      }
    }
    if (!hit) { clearTrail(); draw(); return; }
    const key = String(hit.i) + '|' + (hit.L.ids ? hit.L.ids[hit.i] : '');
    if (trailId === key) { clearTrail(); draw(); return; }
    setTrail(hit.L, hit.i);
    draw();
  });

  installPlayer(() => draw());
  new ResizeObserver(layout).observe(figEl);
  layout();
  renderNow = () => {
    cam.updateProjectionMatrix();
    renderer.render(scene, cam);
    drawAxes();
  };

} else {
  // ═════════════════════════ 3D: orbit viewer ═════════════════════════════
  host.style.left = '0'; host.style.top = '0';
  // FOV 60 matches pcviz; near/far refit after we know the scene radius.
  const cam = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  cam.up.set(0, 0, 1);
  const coord = S.coord || { aspect: 'data', sizeMode: 'scene' };
  // proportional cube: preserve data aspect, or equal axes
  const spans = axesList.map(a => spanOf(a));
  const maxSpan = Math.max(...spans, 1e-12);
  // aspect "auto" sends its box sides (a tall cloud is shortened).
  const ext = (coord.ext && coord.ext.length === 3)
    ? coord.ext.map(Number)
    : (coord.aspect === 'equal')
    ? axesList.map(() => 1)
    : axesList.map((a, i) => spans[i] / maxSpan);
  const sizeAtten = coord.sizeMode !== 'screen';
  // Soft lighting for surface meshes (added once).
  // Physical light units divide by pi; colours are exact sRGB, so a face
  // toward the light shows its own colour and a face away about 80% of it.
  scene.add(new THREE.AmbientLight(0xffffff, 0.8 * Math.PI));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.3 * Math.PI);
  dirLight.position.set(1.2, 0.8, 1.5);
  scene.add(dirLight);

  function pointMaterial(opts) {
    // depthWrite on opaque marks keeps dense lidar crisp; transparent when alpha<1.
    const a = opts.opacity != null ? opts.opacity : 1;
    return new THREE.PointsMaterial({
      ...opts,
      sizeAttenuation: sizeAtten,
      transparent: a < 0.999,
      depthWrite: a >= 0.999,
    });
  }

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
    if (L.kind === 'point' && (L.frames || (L.size && L.size.id))) {
      addBubbleLayer(L, ext, sizeAtten);
    } else if (L.kind === 'point') {
      // Default size is set in Python for unit-cube scene (pcviz-relative).
      const psz = (typeof L.size === 'number') ? L.size : 0.001;
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
          const pt = new THREE.Points(g, pointMaterial({
            color: S.color.palette[ci], size: psz, opacity: L.alpha }));
          scene.add(pt);
          regCat(ci, pt);
        });
      } else {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
        scene.add(new THREE.Points(g, pointMaterial({
          size: psz, vertexColors: true, opacity: L.alpha })));
      }
    } else if (L.kind === 'surface' || L.kind === 'isosurface') {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      if (L.indices && L.indices.data) {
        g.setIndex(new THREE.BufferAttribute(L.indices.data, 1));
      }
      g.computeVertexNormals();
      const baseAlpha = L.alpha != null ? L.alpha : (L.kind === 'isosurface' ? 0.55 : 0.95);
      if (L.wireframe) {
        const rgb = L.constColor ? hex2rgb(L.constColor) : hex2rgb(T.ink2 || '#c3c2b7');
        const wf = new THREE.WireframeGeometry(g);
        const wire = new THREE.LineSegments(wf, new THREE.LineBasicMaterial({
          color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
          transparent: true, opacity: baseAlpha }));
        scene.add(wire);
        if (L.frames && L.kind === 'surface') registerSurface(L, ext, g, pos, wire);
      } else {
        scene.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({
          vertexColors: true, transparent: true, opacity: baseAlpha,
          side: THREE.DoubleSide, depthWrite: L.kind !== 'isosurface' })));
        if (L.frames && L.kind === 'surface') registerSurface(L, ext, g, pos, null);
      }
    } else if (L.frames && L.frames.mode === 'step') {
      addStepLayer(L, ext, cols);
    } else if (L.frames && L.frames.x) {
      addTweenLines(L, ext, cols);
    } else {
      for (const [s0, cnt] of (L.groups || [[0, n]])) {
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

  // Axes box as matplotlib and plotly draw it: the three far walls with
  // grid lines, refreshed as the camera orbits. The edges at the near
  // corner are hidden so they never cross the data.
  const boxMat = new THREE.LineBasicMaterial({ color: T.axis });
  const gridMat = new THREE.LineBasicMaterial({ color: T.grid });
  const cornerAt = (i) => new THREE.Vector3(
    (i & 1) * ext[0], ((i >> 1) & 1) * ext[1], ((i >> 2) & 1) * ext[2]);
  // theme_lidar (void): the data alone, no box, grid, or labels.
  const VOID = !!T.void;
  const boxEdges = (VOID ? [] : [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],
                    [0,4],[1,5],[2,6],[3,7]]).map(([a, b]) => {
    const o = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([cornerAt(a), cornerAt(b)]), boxMat);
    scene.add(o);
    return { a, b, axis: Math.round(Math.log2(a ^ b)), o };
  });
  function axisTicks(ax) {
    const sc = S.scales[ax];
    return sc.kind === 'cat'
      ? sc.cats.map((c, i) => [i, c])
      : (sc.ticks || (sc.ladder ? sc.ladder[0] : []));
  }
  const showGrid = !VOID && !(S.themeOpts && S.themeOpts.panelGrid === false) &&
    String(T.grid).toLowerCase() !== String(T.surface).toLowerCase();
  const walls = [0, 1, 2].map(wall => [0, 1].map(side => {
    const pts = [];
    if (showGrid) {
      for (let k = 0; k < 3; k++) {
        if (k === wall) continue;
        const other = 3 - wall - k;
        for (const [t] of axisTicks(axesList[k])) {
          const u = (t - dataLo(axesList[k])) / spanOf(axesList[k]);
          if (!(u > 0.001 && u < 0.999)) continue;
          const a = [0, 0, 0];
          a[wall] = side * ext[wall];
          a[k] = u * ext[k];
          const b = a.slice();
          b[other] = ext[other];
          pts.push(new THREE.Vector3(a[0], a[1], a[2]), new THREE.Vector3(b[0], b[1], b[2]));
        }
      }
    }
    const o = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), gridMat);
    scene.add(o);
    return o;
  }));
  function updateBox() {
    // A wall is a far wall when the camera is on the box side of its plane.
    const back = [0, 1, 2].map(k => cam.position.getComponent(k) > ext[k] / 2 ? 0 : 1);
    walls.forEach((pair, k) => pair.forEach((o, side) => { o.visible = side === back[k]; }));
    for (const e of boxEdges) {
      let far = false;
      for (let k = 0; k < 3; k++) {
        if (k !== e.axis && ((e.a >> k) & 1) === back[k]) far = true;
      }
      e.o.visible = far;
    }
  }
  const tickSprites = [[], [], []];

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
    tex.colorSpace = THREE.NoColorSpace;
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
  (VOID ? [] : axesList).forEach((ax, ai) => {
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
      tickSprites[ai].push({ sp, t: +t });
    }
    const lp = [[0.5*ext[0], -2.6*off*ext[1], 0],
                [-2.6*off*ext[0], 0.5*ext[1], 0],
                [-2.6*off*ext[0], 0, 0.55*ext[2]]][ai];
    const rawLab = S.labs[ax] == null ? '' : String(S.labs[ax]);
    const tl = sprite(withFrame(rawLab), false);
    tl.position.set(lp[0], lp[1], lp[2]);
    scene.add(tl);
    if (rawLab.indexOf('{frame_time}') >= 0) {
      axisTitleSprites.push({ sprite: tl, raw: rawLab, shown: withFrame(rawLab) });
    }
  });
  refreshFrameLabels = () => {
    for (const item of axisTitleSprites) {
      const text = withFrame(item.raw);
      if (text === item.shown) continue;
      item.shown = text;
      const next = sprite(text, false);
      item.sprite.material.map.dispose();
      item.sprite.material.map = next.material.map;
      item.sprite.scale.copy(next.scale);
      next.material.map = null;
      next.material.dispose();
    }
  };

  // Fit like pcviz: bounding sphere of the scene cube, camera on a soft orbit.
  const ctr = new THREE.Vector3(ext[0]/2, ext[1]/2, ext[2]/2);
  const rad = Math.max(
    Math.sqrt(ext[0]*ext[0] + ext[1]*ext[1] + ext[2]*ext[2]) / 2, 1e-3);
  // pcviz: position at ~1.4R from centre on a diagonal; far = 20R
  // coord_3d(elev=, azim=, zoom=) sends the opening direction and zoom.
  const camSpec = coord.camera || {};
  const dist = rad * 1.25 / Math.tan((cam.fov * Math.PI / 180) / 2) / (camSpec.zoom > 0 ? camSpec.zoom : 1);
  const dir = (camSpec.dir && camSpec.dir.length === 3)
    ? new THREE.Vector3(camSpec.dir[0], camSpec.dir[1], camSpec.dir[2]).normalize()
    : new THREE.Vector3(0.55, -0.85, 0.5).normalize();
  cam.position.copy(ctr.clone().add(dir.multiplyScalar(dist)));
  cam.near = Math.max(rad / 200, 1e-4);
  cam.far = rad * 40;
  cam.updateProjectionMatrix();
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.copy(ctr);
  controls.enableDamping = true;
  controls.update();
  window.__plot3.camera = cam;
  window.__plot3.controls = controls;
  if (SAVED && SAVED.camera) {
    const shot = SAVED.camera;
    if (shot.position && shot.position.length === 3)
      cam.position.set(+shot.position[0], +shot.position[1], +shot.position[2]);
    if (shot.target && shot.target.length === 3)
      controls.target.set(+shot.target[0], +shot.target[1], +shot.target[2]);
    cam.updateProjectionMatrix();
    controls.update();
  }

  // ── 3D hover: nearest point (screen-space) for point layers ────────────
  const tip3 = document.getElementById('tip');
  tip3.style.background = T.surface;
  tip3.style.border = '1px solid ' + T.axis;
  tip3.style.color = T.ink;
  const pickLayers = S.layers
    .map((L, li) => ({ L, li }))
    .filter(({ L }) => L.n > 0 && L.x && L.y && L.z &&
      (L.kind === 'point' || L.kind === 'surface' || L.tip));
  let hover3Tick = 0;
  function hover3d(e) {
    const now = performance.now();
    if (!S.transition && now - hover3Tick < 40) return;
    hover3Tick = now;
    if (!pickLayers.length) { tip3.style.display = 'none'; return; }
    const rect = renderer.domElement.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const w = Math.max(rect.width, 1), h = Math.max(rect.height, 1);
    // Project each point; find nearest in screen px (cap search for big clouds).
    let best = null, bestD = Infinity;
    const maxScan = 80000;
    for (const { L } of pickLayers) {
      const n = L._stepN != null ? L._stepN : L.n;
      const step = n > maxScan ? Math.ceil(n / maxScan) : 1;
      const xs = L._x || L.x.data;
      const ys = L._y || L.y.data;
      const zs = L._z || L.z.data;
      for (let i = 0; i < n; i += step) {
        if (L._alpha && L._alpha[i] <= 0.04) continue;
        const codes = L._ci || (L.color && L.color.kind === 'cat' ? L.color.data : null);
        if (codes && S.color.cats && hiddenCats.has(codes[i] % S.color.cats.length)) continue;
        const v = new THREE.Vector3(
          xs[i] * ext[0],
          ys[i] * ext[1],
          zs[i] * ext[2]
        );
        v.project(cam);
        if (v.z < -1 || v.z > 1) continue;
        const sx = (v.x * 0.5 + 0.5) * w;
        const sy = (-v.y * 0.5 + 0.5) * h;
        const rad = (L._size && !sizeAtten) ? Math.max(14, L._size[i] * 0.55) : 14;
        const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
        if (d <= rad * rad && d < bestD) {
          bestD = d; best = [L, i, sx, sy, xs[i], ys[i], zs[i]];
        }
      }
    }
    hoverHold = !!(best && best[0].kind === 'point' && best[0].frames && S.transition);
    if (!best) { tip3.style.display = 'none'; return; }
    const [L, i, sx, sy, xn, yn, zn] = best;
    const xv = fromScale('x', xn);
    const yv = fromScale('y', yn);
    const zv = fromScale('z', zn);
    let head = '';
    if (L.ids && L.ids[i]) head += '<b>' + plot3Esc(L.ids[i]) + '</b><br>';
    const codes = L._ci || (L.color && L.color.kind === 'cat' ? L.color.data : null);
    if (codes && S.color.cats)
      head += '<b>' + S.color.cats[codes[i] % S.color.cats.length] + '</b><br>';
    else if (L.color && L.color.kind === 'num') {
      const t = L._cnorm ? L._cnorm[i] : (L.color.data[i] / 65535);
      head += '<b>' + fmt(cval(t)) + '</b><br>';
    }
    let sizeBit = '';
    if (L._size && L.size && L.size.vmax && L.size.max) {
      const frac = L._size[i] / L.size.max;
      const value = frac * frac * L.size.vmax;
      const name = (S.sizeLegend && S.sizeLegend.label) || 'size';
      sizeBit = '<br>' + plot3Esc(name) + ' = ' + fmt(value);
    }
    tip3.innerHTML = L.tip
      ? head + formulaHead(L)
        + axisPair('x', fmt(xv)) + ', '
        + axisPair('y', fmt(yv)) + ', '
        + axisPair('z', fmt(zv)) + sizeBit
      : head + fmt(xv) + ', ' + fmt(yv) + ', ' + fmt(zv) + sizeBit;
    plot3Typeset(tip3);
    tip3.style.left = Math.min(w - 8, sx + 12) + 'px';
    tip3.style.top = Math.max(8, sy - 10) + 'px';
    tip3.style.display = 'block';
  }
  function fmt(v) {
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a >= 1e6 || (a > 0 && a < 1e-3)) return v.toExponential(2);
    return (Math.round(v * 1e4) / 1e4).toString();
  }
  // colour value for continuous ramp (same helper as 2D tip when present)
  function cval(t) {
    if (S.color && S.color.kind === 'num') {
      const lo = S.color.lo, hi = S.color.hi;
      return lo + t * (hi - lo);
    }
    return t;
  }
  renderer.domElement.addEventListener('pointermove', hover3d);
  renderer.domElement.addEventListener('pointerleave', () => {
    tip3.style.display = 'none';
    hoverHold = false;
  });
  let pointerMoved3 = false;
  let down3 = null;
  renderer.domElement.addEventListener('pointerdown', (e) => {
    pointerMoved3 = false;
    down3 = { x: e.clientX, y: e.clientY };
  });
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (!down3) return;
    if (Math.abs(e.clientX - down3.x) + Math.abs(e.clientY - down3.y) > 4)
      pointerMoved3 = true;
  });
  renderer.domElement.addEventListener('pointerup', () => { down3 = null; });
  renderer.domElement.addEventListener('click', (e) => {
    if (!S.transition || pointerMoved3) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const w = Math.max(rect.width, 1), h = Math.max(rect.height, 1);
    let hit = null, hitD = Infinity;
    for (const L of S.layers) {
      if (L.kind !== 'point' || !L.frames || !L._x || !L._z) continue;
      for (let i = 0; i < L.n; i++) {
        if (L._alpha && L._alpha[i] <= 0.04) continue;
        const v = new THREE.Vector3(L._x[i] * ext[0], L._y[i] * ext[1], L._z[i] * ext[2]);
        v.project(cam);
        const sx = (v.x * 0.5 + 0.5) * w;
        const sy = (-v.y * 0.5 + 0.5) * h;
        const rad = (L._size && !sizeAtten) ? Math.max(16, L._size[i] * 0.55) : 16;
        const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
        if (d <= rad * rad && d < hitD) { hitD = d; hit = { L, i }; }
      }
    }
    if (!hit) { clearTrail(); return; }
    const key = String(hit.i) + '|' + (hit.L.ids ? hit.L.ids[hit.i] : '');
    if (trailId === key) { clearTrail(); return; }
    setTrail(hit.L, hit.i);
  });

  function layout() {
    placeLegend();
    const w = Math.max(figEl.clientWidth, 1), h = Math.max(figEl.clientHeight, 1);
    renderer.setSize(w, h);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    for (const lm of lineMats) lm.resolution.set(w, h);
  }
  // Tick labels on a foreshortened edge crowd together: keep every k-th,
  // the smallest k whose labels do not touch, on round values when the
  // ticks allow, and drop any that land on another axis's labels.
  const tickTmp = new THREE.Vector3();
  function tickRect(sp, w, h) {
    tickTmp.copy(sp.position).project(cam);
    const sx = (tickTmp.x * 0.5 + 0.5) * w, sy = (-tickTmp.y * 0.5 + 0.5) * h;
    tickTmp.copy(sp.position).applyMatrix4(cam.matrixWorldInverse);
    const depth = Math.max(-tickTmp.z, 1e-6);
    const perUnit = h / (2 * Math.tan(cam.fov * Math.PI / 360) * depth);
    const pw = sp.scale.x * perUnit * 0.8, ph = sp.scale.y * perUnit * 0.75;
    return [sx - pw / 2, sy - ph / 2, pw, ph];
  }
  const touches = (a, b, air) => a[0] - air < b[0] + b[2] && b[0] - air < a[0] + a[2] &&
    a[1] - air < b[1] + b[3] && b[1] - air < a[1] + a[3];
  function declutterTicks() {
    const w = Math.max(figEl.clientWidth, 1), h = Math.max(figEl.clientHeight, 1);
    const placed = [];
    for (const marks of tickSprites) {
      if (!marks.length) continue;
      const rects = marks.map(m => tickRect(m.sp, w, h));
      const air = Math.max(3, 0.5 * (rects[0] ? rects[0][3] : 0));
      let keep = null;
      for (let k = 1; k <= marks.length && !keep; k++) {
        const offsets = [];
        const step = marks.length > 1 ? Math.abs(marks[1].t - marks[0].t) * k : 0;
        for (let o = 0; o < k; o++) {
          const r = step > 0 ? Math.abs(marks[o].t / step - Math.round(marks[o].t / step)) : 1;
          if (r < 1e-6) offsets.unshift(o); else offsets.push(o);
        }
        for (const o of offsets) {
          let ok = true;
          for (let i = o; i + k < marks.length; i += k) {
            if (touches(rects[i], rects[i + k], air)) { ok = false; break; }
          }
          if (ok) { keep = [o, k]; break; }
        }
      }
      if (!keep) keep = [0, marks.length];
      marks.forEach((m, i) => {
        let on = i >= keep[0] && (i - keep[0]) % keep[1] === 0;
        if (on && placed.some(r => touches(rects[i], r, 0))) on = false;
        if (on) placed.push(rects[i]);
        m.sp.visible = on;
      });
    }
  }
  function frame3d() {
    controls.update();
    updateBox();
    declutterTicks();
    renderer.render(scene, cam);
  }
  installPlayer(() => {});
  new ResizeObserver(layout).observe(figEl);
  layout();
  (function loop() {
    frame3d();
    requestAnimationFrame(loop);
  })();
  renderNow = frame3d;
}

// Save menu. PNG is the picture on screen. SVG keeps the axis overlay as
// vectors and embeds the WebGL layer, which is already a raster.
function installSave() {
  const bar = document.getElementById('modebar');
  const btn = document.getElementById('save-btn');
  const menu = document.getElementById('save-menu');
  if (!bar || !btn || !menu) return;
  btn.style.background = T.surface;
  btn.style.color = T.ink;
  btn.style.border = '1px solid ' + T.axis;
  menu.style.background = T.surface;
  menu.style.color = T.ink;
  menu.style.border = '1px solid ' + T.axis;
  const svgHint = menu.querySelector('[data-act="svg"] .save-d');
  if (svgHint && S.is3d) svgHint.textContent = 'Vector axes';
  const videoItem = menu.querySelector('[data-act="video"]');
  const canRecord = !!(S.transition && S.transition.nFrames > 1
    && typeof MediaRecorder !== 'undefined' && renderer.domElement.captureStream);
  if (videoItem) videoItem.hidden = !canRecord;
  bar.addEventListener('pointerdown', (ev) => ev.stopPropagation());

  function slug(raw, limit) {
    let stem = '';
    const cap = limit || 60;
    for (let i = 0; i < raw.length && stem.length < cap; i++) {
      const c = raw.charAt(i);
      const ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') || c === '.' || c === '_' || c === '-';
      if (ok) stem += c;
      else if (stem && stem.charAt(stem.length - 1) !== '_') stem += '_';
    }
    while (stem.charAt(0) === '_') stem = stem.slice(1);
    while (stem && stem.charAt(stem.length - 1) === '_') stem = stem.slice(0, -1);
    return stem;
  }
  function fileStem() {
    let raw = (S.labs && S.labs.title) ? String(S.labs.title) : '';
    const hasToken = raw.indexOf('{frame_time}') >= 0;
    if (frameLabel && hasToken) raw = raw.split('{frame_time}').join(frameLabel);
    let stem = slug(raw, 60) || 'plot3';
    if (frameLabel && !hasToken && (S.transition || S.slider)) {
      const extra = slug(String(frameLabel), 40);
      if (extra) stem = slug(stem + '_' + extra, 80) || stem;
    }
    return stem || 'plot3';
  }
  function num(v) { return String(Math.round(Number(v) * 100) / 100); }
  function visibleColor(c) {
    if (!c || c === 'transparent' || c === 'rgba(0, 0, 0, 0)') return '';
    return c;
  }
  function traceRound(ctx, x, y, w, h, rad) {
    const r = Math.max(0, Math.min(rad || 0, w / 2, h / 2));
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
  }
  function paintBox(ctx, el, st, ox, oy, alpha) {
    const r = el.getBoundingClientRect();
    const w = r.width, h = r.height;
    if (w < 0.5 || h < 0.5) return;
    const bg = visibleColor(st.backgroundColor);
    const ramp = el.id === 'ramp';
    const bw = parseFloat(st.borderTopWidth) || 0;
    const bc = bw > 0 ? visibleColor(st.borderTopColor) : '';
    if (!bg && !ramp && !bc) return;
    const x = r.left - ox, y = r.top - oy;
    const rad = parseFloat(st.borderRadius) || 0;
    ctx.save();
    ctx.globalAlpha = alpha;
    traceRound(ctx, x, y, w, h, rad);
    if (ramp && S.color && S.color.ramp && S.color.ramp.length) {
      const g = ctx.createLinearGradient(x, y, x + w, y);
      const stops = S.color.ramp;
      for (let i = 0; i < stops.length; i++)
        g.addColorStop(stops.length === 1 ? 0 : i / (stops.length - 1), stops[i]);
      ctx.fillStyle = g;
      ctx.fill();
    } else if (bg) {
      ctx.fillStyle = bg;
      ctx.fill();
    }
    if (bc) {
      ctx.lineWidth = bw;
      ctx.strokeStyle = bc;
      ctx.stroke();
    }
    ctx.restore();
  }
  function fillWrapped(ctx, text, x, y, maxW, lineH) {
    const words = text.split(' ');
    let line = '', cy = y;
    for (let i = 0; i < words.length; i++) {
      const trial = line ? line + ' ' + words[i] : words[i];
      if (line && ctx.measureText(trial).width > maxW) {
        ctx.fillText(line, x, cy);
        line = words[i];
        cy += lineH;
      } else line = trial;
    }
    if (line) ctx.fillText(line, x, cy);
  }
  function paintText(ctx, node, ox, oy, alpha) {
    const text = node.textContent;
    if (!text || !text.trim()) return;
    const parent = node.parentElement;
    if (!parent) return;
    const st = getComputedStyle(parent);
    const range = document.createRange();
    range.selectNodeContents(node);
    const rects = range.getClientRects();
    range.detach();
    if (!rects.length) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = st.color;
    ctx.font = (st.fontStyle || 'normal') + ' ' + (st.fontWeight || '400') + ' '
      + (st.fontSize || '12px') + ' ' + (st.fontFamily || 'sans-serif');
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    if ('letterSpacing' in ctx && st.letterSpacing && st.letterSpacing !== 'normal')
      ctx.letterSpacing = st.letterSpacing;
    if (rects.length === 1) {
      ctx.fillText(text, rects[0].left - ox, rects[0].top - oy);
    } else {
      const pr = parent.getBoundingClientRect();
      const padL = parseFloat(st.paddingLeft) || 0;
      const padR = parseFloat(st.paddingRight) || 0;
      const maxW = Math.max(1, pr.width - padL - padR);
      let lineH = parseFloat(st.lineHeight);
      if (!isFinite(lineH)) lineH = rects[0].height || 14;
      fillWrapped(ctx, text.trim(), pr.left + padL - ox, rects[0].top - oy, maxW, lineH);
    }
    ctx.restore();
  }
  function paintNode(ctx, node, ox, oy, alpha) {
    if (!node) return;
    if (node.nodeType === 3) { paintText(ctx, node, ox, oy, alpha); return; }
    if (node.nodeType !== 1) return;
    const el = node;
    if (el.id === 'modebar' || el.id === 'tip' || el.id === 'hint' || el.id === 'canvas-host' || el.id === 'axes')
      return;
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return;
    const own = parseFloat(st.opacity);
    const next = alpha * (isFinite(own) ? own : 1);
    if (next <= 0.001) return;
    paintBox(ctx, el, st, ox, oy, next);
    for (let i = 0; i < el.childNodes.length; i++)
      paintNode(ctx, el.childNodes[i], ox, oy, next);
  }
  function svgRotate(transform) {
    if (!transform) return null;
    const i = transform.indexOf('rotate(');
    if (i < 0) return null;
    const inner = transform.slice(i + 7, transform.indexOf(')', i));
    const parts = inner.trim().split(/[ ,]+/).map(Number);
    if (parts.length < 3 || parts.some(n => !isFinite(n))) return null;
    return parts;
  }
  function paintAxes(ctx, from) {
    const root = from || svg;
    if (!root) return;
    const font = '12px ' + (getComputedStyle(document.body).fontFamily || 'sans-serif');
    const nodes = root.querySelectorAll('line, rect, text');
    for (let i = 0; i < nodes.length; i++) {
      const el = nodes[i];
      const tag = el.tagName.toLowerCase();
      ctx.save();
      if (tag === 'line') {
        ctx.beginPath();
        ctx.strokeStyle = el.getAttribute('stroke') || T.grid;
        ctx.lineWidth = 1;
        ctx.moveTo(+el.getAttribute('x1'), +el.getAttribute('y1'));
        ctx.lineTo(+el.getAttribute('x2'), +el.getAttribute('y2'));
        ctx.stroke();
      } else if (tag === 'rect') {
        ctx.strokeStyle = el.getAttribute('stroke') || T.axis;
        ctx.lineWidth = 1;
        ctx.strokeRect(+el.getAttribute('x'), +el.getAttribute('y'),
          +el.getAttribute('width'), +el.getAttribute('height'));
      } else if (tag === 'text') {
        const x = +el.getAttribute('x'), y = +el.getAttribute('y');
        const anchor = el.getAttribute('text-anchor') || 'start';
        ctx.fillStyle = el.getAttribute('fill') || T.ink2;
        ctx.font = font;
        ctx.textAlign = anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left';
        ctx.textBaseline = 'alphabetic';
        const rot = svgRotate(el.getAttribute('transform'));
        if (rot) {
          ctx.translate(rot[1], rot[2]);
          ctx.rotate(rot[0] * Math.PI / 180);
          ctx.translate(-rot[1], -rot[2]);
        }
        ctx.fillText(el.textContent || '', x, y);
      }
      ctx.restore();
    }
  }
  function pickDpr(w, h, prefer) {
    let dpr = prefer == null ? Math.min(window.devicePixelRatio || 1, 3) : prefer;
    const maxSide = 4096;
    if (w * dpr > maxSide) dpr = maxSide / w;
    if (h * dpr > maxSide) dpr = Math.min(dpr, maxSide / h);
    return Math.max(0.5, dpr);
  }
  function layerCanvas(w, h, dpr) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w * dpr));
    c.height = Math.max(1, Math.round(h * dpr));
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { canvas: c, ctx: ctx };
  }
  function compose(kind) {
    const gl = renderer.domElement;
    const bufW = Math.max(1, gl.clientWidth);
    const bufH = Math.max(1, gl.clientHeight);
    const prevRatio = renderer.getPixelRatio();
    const boost = kind === 'png' && bufW > 1 && bufH > 1;
    if (boost) {
      renderer.setPixelRatio(2);
      renderer.setSize(bufW, bufH, false);
    }
    try {
      return composeView(kind);
    } finally {
      if (boost) {
        renderer.setPixelRatio(prevRatio);
        renderer.setSize(bufW, bufH, false);
        renderNow();
      }
    }
  }
  function composeView(kind) {
    renderNow();
    const figR = figEl.getBoundingClientRect();
    const cssW = Math.max(1, figR.width);
    const cssH = Math.max(1, figR.height);
    const noteEl = document.getElementById('note');
    const yearEl = document.getElementById('year');
    let noteH = 0;
    if (noteEl && getComputedStyle(noteEl).display !== 'none' && (noteEl.textContent || '').trim())
      noteH = noteEl.offsetHeight || 0;
    const totalH = cssH + noteH;
    const dpr = pickDpr(cssW, totalH, kind === 'png' ? 2 : null);
    const hostR = host.getBoundingClientRect();
    const hx = hostR.left - figR.left, hy = hostR.top - figR.top;
    const shot = layerCanvas(cssW, totalH, dpr);
    const ctx = shot.ctx;
    ctx.fillStyle = T.surface;
    ctx.fillRect(0, 0, cssW, totalH);
    const yearOn = yearEl && getComputedStyle(yearEl).display !== 'none' && (yearEl.textContent || '').trim();
    let yearURL = '', yearBox = null;
    if (yearOn) {
      const yr = yearEl.getBoundingClientRect();
      const pad = 8;
      const yw = yr.width + pad * 2, yh = yr.height + pad * 2;
      const layer = layerCanvas(yw, yh, dpr);
      paintNode(layer.ctx, yearEl, yr.left - pad, yr.top - pad, 1);
      yearBox = { x: yr.left - figR.left - pad, y: yr.top - figR.top - pad, w: yw, h: yh };
      ctx.drawImage(layer.canvas, yearBox.x, yearBox.y, yw, yh);
      if (kind === 'svg') yearURL = layer.canvas.toDataURL('image/png');
    }
    paintAxes(ctx, gridSvg);
    if (hostR.width > 1 && hostR.height > 1)
      ctx.drawImage(renderer.domElement, hx, hy, hostR.width, hostR.height);
    paintAxes(ctx);
    const chrome = layerCanvas(cssW, totalH, dpr);
    paintNode(chrome.ctx, titleEl, figR.left, figR.top, 1);
    if (legEl && legEl.style.display !== 'none')
      paintNode(chrome.ctx, legEl, figR.left, figR.top, 1);
    if (noteH > 0) {
      const nr = noteEl.getBoundingClientRect();
      paintNode(chrome.ctx, noteEl, figR.left, nr.top - cssH, 1);
    }
    ctx.drawImage(chrome.canvas, 0, 0, cssW, totalH);
    if (kind === 'svg') {
      const W = Math.max(1, Math.round(cssW)), H = Math.max(1, Math.round(totalH));
      const glURL = renderer.domElement.toDataURL('image/png');
      const chromeURL = chrome.canvas.toDataURL('image/png');
      const axesInner = (svg && svg.innerHTML.trim()) ? svg.innerHTML : '';
      const titleText = plot3Esc(withFrame((S.labs && S.labs.title) || 'plot3'));
      let body = '<rect width="100%" height="100%" fill="' + T.surface + '"/>';
      if (yearURL && yearBox) {
        body += '<image x="' + num(yearBox.x) + '" y="' + num(yearBox.y)
          + '" width="' + num(yearBox.w) + '" height="' + num(yearBox.h)
          + '" href="' + yearURL + '"/>';
      }
      if (gridSvg && gridSvg.innerHTML.trim()) body += '<g>' + gridSvg.innerHTML + '</g>';
      body += '<image x="' + num(hx) + '" y="' + num(hy)
        + '" width="' + num(hostR.width) + '" height="' + num(hostR.height)
        + '" href="' + glURL + '"/>';
      if (axesInner) {
        body += '<g font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">'
          + axesInner + '</g>';
      }
      body += '<image x="0" y="0" width="' + W + '" height="' + H + '" href="' + chromeURL + '"/>';
      return '<?xml version="1.0" encoding="UTF-8"?>'
        + '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H
        + '" viewBox="0 0 ' + W + ' ' + H + '"><title>' + titleText + '</title>'
        + body + '</svg>';
    }
    return shot.canvas.toDataURL('image/png');
  }
  function dataURLToBlob(url) {
    const comma = url.indexOf(',');
    const head = url.slice(0, comma);
    const body = url.slice(comma + 1);
    const mimeEnd = head.indexOf(';');
    const mime = head.slice(head.indexOf(':') + 1, mimeEnd < 0 ? head.length : mimeEnd)
      || 'application/octet-stream';
    const bin = atob(body);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  }
  function downloadBlob(blob, name) {
    const a = document.createElement('a');
    const url = URL.createObjectURL(blob);
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  function downloadsAllowed() {
    try {
      const sb = window.frameElement && window.frameElement.sandbox;
      return !sb || sb.length === 0 || sb.contains('allow-downloads');
    } catch (err) { return true; }
  }
  async function saveFile(blob, name, mime, ext) {
    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: name,
          types: [{ description: ext.toUpperCase() + ' file', accept: { [mime]: ['.' + ext] } }],
        });
        const out = await handle.createWritable();
        await out.write(blob);
        await out.close();
        return 'saved';
      } catch (err) {
        if (err.name === 'AbortError') return 'cancelled';
      }
    }
    if (downloadsAllowed()) { downloadBlob(blob, name); return 'downloaded'; }
    throw new Error('downloads blocked');
  }
  function stateJSON() {
    const state = {};
    if (S.transition) state.playT = playT;
    if (S.slider && S.slider.params) {
      const sliders = {};
      for (let i = 0; i < S.slider.params.length; i++) {
        const name = S.slider.params[i].name;
        const node = document.getElementById('slider-' + name);
        if (node) sliders[name] = +node.value;
      }
      state.sliders = sliders;
    }
    const cam = window.__plot3.camera;
    const controls = window.__plot3.controls;
    if (cam && controls) {
      state.camera = {
        position: [cam.position.x, cam.position.y, cam.position.z],
        target: [controls.target.x, controls.target.y, controls.target.z],
      };
    }
    return JSON.stringify(state).split('<').join('\\\\' + 'u003c');
  }
  function htmlDocument() {
    // Split the tags so the source itself does not contain them. PRISTINE
    // includes this script, and a contiguous match would rewrite the module.
    const open = '<script type="application/json" id="' + 'plot3-state">';
    const close = '<' + '/script>';
    const tag = open + stateJSON() + close;
    let doc = PRISTINE;
    const at = doc.indexOf(open);
    if (at >= 0) {
      const end = doc.indexOf(close, at);
      if (end >= 0) return doc.slice(0, at) + tag + doc.slice(end + close.length);
    }
    const closeBody = '<' + '/body>';
    const bodyAt = doc.lastIndexOf(closeBody);
    if (bodyAt >= 0) return doc.slice(0, bodyAt) + tag + doc.slice(bodyAt);
    return doc + tag;
  }
  const BLOCKED = "Downloads are blocked here. Use Copy PNG, or in Python: ggsave('plot.png', fig).";
  function remember(item) {
    if (!item.dataset.labelHtml) item.dataset.labelHtml = item.innerHTML;
  }
  function restoreItem(item) {
    if (item._saveTimer) { clearTimeout(item._saveTimer); item._saveTimer = 0; }
    if (item.dataset.labelHtml) item.innerHTML = item.dataset.labelHtml;
    item.classList.remove('save-status');
    item.disabled = false;
  }
  function showStatus(item, text) {
    if (item._saveTimer) { clearTimeout(item._saveTimer); item._saveTimer = 0; }
    remember(item);
    item.textContent = text;
    item.classList.add('save-status');
  }
  function flashSaved(item) {
    showStatus(item, 'Saved ✓');
    item._saveTimer = setTimeout(() => restoreItem(item), 1600);
  }
  function menuItems() {
    return Array.prototype.filter.call(
      menu.querySelectorAll('[role="menuitem"]'),
      (el) => !el.hidden);
  }
  function placeMenu() {
    // The button is at the top of the figure. Opening upward when the plot
    // is short clips the menu off the iframe. Pick the side that shows more
    // of it, which is upward only when the button sits low in the window.
    const btnR = btn.getBoundingClientRect();
    const menuH = menu.offsetHeight || 220;
    const spaceBelow = window.innerHeight - btnR.bottom;
    const spaceAbove = btnR.top;
    const visibleBelow = Math.min(menuH, Math.max(0, spaceBelow));
    const visibleAbove = Math.min(menuH, Math.max(0, spaceAbove));
    bar.classList.toggle('open-up', visibleAbove > visibleBelow);
  }
  function closeMenu(focusBtn) {
    menu.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
    bar.classList.remove('open');
    menuItems().forEach(restoreItem);
    if (focusBtn) btn.focus();
  }
  function openMenu(focusIndex) {
    menu.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    bar.classList.add('open');
    placeMenu();
    const items = menuItems();
    if (!items.length) return;
    const idx = focusIndex == null ? 0 : Math.max(0, Math.min(items.length - 1, focusIndex));
    items[idx].focus();
  }
  async function runSave(item) {
    if (!item || item.dataset.busy === '1') return;
    const act = item.getAttribute('data-act');
    item.dataset.busy = '1';
    item.disabled = true;
    showStatus(item, act === 'video' ? 'Recording…' : 'Saving…');
    try {
      if (act === 'video') {
        const blob = await window.__plot3.startRecording();
        showStatus(item, 'Saving…');
        const how = await saveFile(blob, fileStem() + '.webm', 'video/webm', 'webm');
        window.__plot3.saveError = '';
        if (how === 'cancelled') restoreItem(item);
        else flashSaved(item);
        return;
      }
      const stem = fileStem();
      let blob, name, mime, ext;
      if (act === 'png' || act === 'copy') {
        blob = dataURLToBlob(compose('png'));
        name = stem + '.png'; mime = 'image/png'; ext = 'png';
      } else if (act === 'svg') {
        blob = new Blob([compose('svg')], { type: 'image/svg+xml;charset=utf-8' });
        name = stem + '.svg'; mime = 'image/svg+xml'; ext = 'svg';
      } else {
        blob = new Blob([htmlDocument()], { type: 'text/html;charset=utf-8' });
        name = stem + '.html'; mime = 'text/html'; ext = 'html';
      }
      if (act === 'copy') {
        if (!navigator.clipboard || !navigator.clipboard.write || typeof ClipboardItem === 'undefined')
          throw new Error('Clipboard image copy is not available in this browser');
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
        window.__plot3.saveError = '';
        flashSaved(item);
        return;
      }
      const how = await saveFile(blob, name, mime, ext);
      window.__plot3.saveError = '';
      if (how === 'cancelled') restoreItem(item);
      else flashSaved(item);
    } catch (err) {
      const blocked = err && err.message === 'downloads blocked';
      const msg = blocked ? BLOCKED : String(err && err.message ? err.message : err);
      window.__plot3.saveError = msg;
      showStatus(item, msg);
      item.disabled = false;
      console.error(err);
    } finally {
      item.dataset.busy = '0';
    }
  }
  btn.addEventListener('click', () => {
    if (menu.hidden) openMenu(0);
    else closeMenu(false);
  });
  menu.querySelectorAll('[role="menuitem"]').forEach((item) => {
    item.addEventListener('click', () => runSave(item));
  });
  document.addEventListener('pointerdown', (ev) => {
    if (menu.hidden || bar.contains(ev.target)) return;
    closeMenu(false);
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && !menu.hidden) {
      ev.preventDefault();
      ev.stopPropagation();
      closeMenu(true);
      return;
    }
    const items = menuItems();
    if (menu.hidden) {
      if (!bar.contains(ev.target)) return;
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
        ev.preventDefault();
        openMenu(ev.key === 'ArrowUp' ? items.length - 1 : 0);
      }
      return;
    }
    if (!bar.contains(ev.target)) return;
    const cur = items.indexOf(document.activeElement);
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      ev.stopPropagation();
      if (!items.length) return;
      const dir = ev.key === 'ArrowDown' ? 1 : -1;
      const next = items[(cur + dir + items.length) % items.length];
      next.focus();
    } else if (ev.key === 'Home' && items.length) {
      ev.preventDefault();
      items[0].focus();
    } else if (ev.key === 'End' && items.length) {
      ev.preventDefault();
      items[items.length - 1].focus();
    }
  });
  window.__plot3.saveError = '';
  window.__plot3.snapshot = (kind) => compose(kind || 'png');
}
installSave();
</script>
</body></html>"""

# Inserted only when the figure has a formula or a ``$...$`` label. Data
# plots keep the portable HTML free of the KaTeX download.
_KATEX_BOOT = """if (S.math) {
  const katexCss = document.createElement('link');
  katexCss.rel = 'stylesheet';
  katexCss.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css';
  document.head.appendChild(katexCss);
  const katexJs = document.createElement('script');
  katexJs.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js';
  katexJs.onload = () => plot3Typeset(document);
  document.head.appendChild(katexJs);
}
"""
