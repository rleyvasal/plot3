"""Save menu: iframe download permission, and the viewer menu in a browser."""

from __future__ import annotations

import re
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from plot3 import aes, geom_point, ggplot, transition_time

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

_PROBE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:#0b1020}
#view{width:100%;height:100%;border:0}
#boxed{position:absolute;width:320px;height:240px;left:-4000px;top:0;border:0}
#short{position:absolute;width:640px;height:160px;left:0;top:120vh;border:0}
#log{position:absolute;left:8px;bottom:8px;margin:0;color:#e2e8f0;
  font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap}
</style></head><body>
<iframe id="view" src="plot.html"></iframe>
<iframe id="boxed" sandbox="allow-scripts allow-same-origin allow-pointer-lock" src="plot.html"></iframe>
<iframe id="short" src="anim.html"></iframe>
<pre id="log">waiting</pre>
<script>
function log(line) {
  const el = document.getElementById('log');
  el.textContent = (el.textContent === 'waiting' ? '' : el.textContent + '\\n') + line;
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
async function waitPlot(frame) {
  const w = frame.contentWindow;
  for (let i = 0; i < 80; i++) {
    try {
      if (w.__plot3 && typeof w.__plot3.snapshot === 'function') return w;
    } catch (err) {}
    await sleep(50);
  }
  throw new Error('snapshot never arrived');
}
async function pixelsVary(url) {
  const img = new Image();
  const loaded = new Promise((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('png decode failed'));
  });
  img.src = url;
  await loaded;
  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  let varied = false;
  const r0 = d[0], g0 = d[1], b0 = d[2];
  for (let i = 0; i < d.length; i += 16) {
    if (d[i] !== r0 || d[i + 1] !== g0 || d[i + 2] !== b0) { varied = true; break; }
  }
  return { varied: varied, w: img.width, h: img.height };
}
(async () => {
  try {
    const w = await waitPlot(document.getElementById('view'));
    const doc = w.document;
    const url = w.__plot3.snapshot('png');
    const info = await pixelsVary(url);
    const cssW = doc.getElementById('fig').clientWidth;
    log('png ' + info.w + 'x' + info.h + ' vary=' + info.varied + ' css=' + cssW);
    if (String(url).indexOf('data:image/png') === 0 && info.varied && info.w >= cssW * 1.5)
      log('PNG_OK');
    else log('PNG_BAD');
    const btn = doc.getElementById('save-btn');
    const menu = doc.getElementById('save-menu');
    const svgHint = menu.querySelector('[data-act="svg"] .save-d').textContent;
    log('svgHint ' + svgHint);
    btn.click();
    const first = menu.querySelector('[data-act="html"]');
    const second = menu.querySelector('[data-act="svg"]');
    log('open hidden=' + menu.hidden + ' exp=' + btn.getAttribute('aria-expanded')
      + ' focus1=' + (doc.activeElement === first));
    first.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    const arrows = doc.activeElement === second;
    log('focus2=' + arrows);
    btn.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    const escOk = menu.hidden && btn.getAttribute('aria-expanded') === 'false'
      && doc.activeElement === btn;
    log('esc hidden=' + menu.hidden + ' exp=' + btn.getAttribute('aria-expanded')
      + ' focusBtn=' + (doc.activeElement === btn));
    btn.click();
    doc.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    const outsideOk = menu.hidden && btn.getAttribute('aria-expanded') === 'false';
    log('outside hidden=' + menu.hidden);
    if (arrows && escOk && outsideOk) log('MENU_OK');
    else log('MENU_BAD');
    if (svgHint === 'Vector, for papers') log('SVG_LABEL_OK');

    const bw = await waitPlot(document.getElementById('boxed'));
    try {
      Object.defineProperty(bw, 'showSaveFilePicker', {
        configurable: true, writable: true, value: undefined,
      });
    } catch (err) {
      try { bw.showSaveFilePicker = undefined; } catch (err2) {}
    }
    const bdoc = bw.document;
    bdoc.getElementById('save-btn').click();
    const item = bdoc.querySelector('[data-act="png"]');
    item.click();
    let seen = item.textContent || '';
    for (let i = 0; i < 40 && seen.indexOf('Downloads are blocked') < 0; i++) {
      await sleep(50);
      seen = item.textContent || '';
    }
    log('blocked ' + seen);
    log('saveError ' + bw.__plot3.saveError);
    if (seen.indexOf("Downloads are blocked here. Use Copy PNG, or in Python: ggsave('plot.png', fig).") >= 0
        && bw.__plot3.saveError.indexOf('Downloads are blocked') >= 0)
      log('BLOCK_OK');
    else log('BLOCK_BAD');

    const sw = await waitPlot(document.getElementById('short'));
    const sdoc = sw.document;
    const sbtn = sdoc.getElementById('save-btn');
    const sbar = sdoc.getElementById('modebar');
    const vid = sdoc.querySelector('[data-act="video"]');
    sbtn.click();
    const canvas = sdoc.querySelector('canvas');
    const canRec = typeof sw.MediaRecorder !== 'undefined' && canvas && canvas.captureStream;
    const sMenu = sdoc.getElementById('save-menu');
    const sBtnR = sbtn.getBoundingClientRect();
    const menuH = sMenu.offsetHeight || 0;
    const spaceBelow = sw.innerHeight - sBtnR.bottom;
    const spaceAbove = sBtnR.top;
    const visibleBelow = Math.min(menuH, Math.max(0, spaceBelow));
    const visibleAbove = Math.min(menuH, Math.max(0, spaceAbove));
    const up = sbar.classList.contains('open-up');
    const expectUp = visibleAbove > visibleBelow;
    log('shortH ' + sw.innerHeight + ' menuH=' + menuH
      + ' above=' + Math.round(spaceAbove) + ' below=' + Math.round(spaceBelow)
      + ' up=' + up + ' videoHidden=' + vid.hidden + ' canRec=' + !!canRec);
    if (sw.innerHeight > 100 && menuH > 40 && up === expectUp) log('UP_OK');
    else log('UP_BAD');
    if (!canRec) log('VIDEO_SKIP');
    else if (!vid.hidden) log('VIDEO_OK');
    else log('VIDEO_BAD');

    btn.click();
    log('DONE');
  } catch (err) {
    log('EXC ' + (err && err.stack ? err.stack : err));
  }
})();
</script>
</body></html>
"""


def test_iframe_allows_downloads_and_clipboard():
    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    iframe = (ggplot(df, aes(x="x", y="y")) + geom_point())._iframe()
    sandbox = re.search(r'sandbox="([^"]*)"', iframe)
    allow = re.search(r'\sallow="([^"]*)"', iframe)
    assert sandbox is not None
    tokens = set(sandbox.group(1).split())
    assert tokens >= {
        "allow-scripts",
        "allow-same-origin",
        "allow-pointer-lock",
        "allow-downloads",
    }
    assert allow is not None
    assert "clipboard-write" in allow.group(1)
    assert "fullscreen" in allow.group(1)


@pytest.mark.skipif(not CHROME.exists(), reason="Chrome is not installed")
def test_save_menu_in_headless_chrome(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 2.0]})
    (tmp_path / "plot.html").write_text(
        (ggplot(df, aes(x="x", y="y")) + geom_point()).html(),
        encoding="utf-8",
    )
    anim = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 3.0, 2.0],
        "year": [2000, 2001, 2002],
        "id": ["a", "a", "a"],
    })
    (tmp_path / "anim.html").write_text(
        (
            ggplot(anim, aes(x="x", y="y", group="id"))
            + geom_point()
            + transition_time("year")
        ).html(),
        encoding="utf-8",
    )
    (tmp_path / "probe.html").write_text(_PROBE, encoding="utf-8")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    shot = tmp_path / "menu.png"
    try:
        proc = subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--enable-webgl",
                "--force-device-scale-factor=1",
                "--virtual-time-budget=10000",
                "--window-size=980,720",
                f"--screenshot={shot}",
                "--dump-dom",
                f"http://127.0.0.1:{port}/probe.html",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        httpd.shutdown()
    text = proc.stdout or ""
    assert proc.returncode == 0, (proc.stderr or "")[-2000:]
    found = re.search(r'<pre id="log">(.*?)</pre>', text, re.DOTALL)
    assert found, text
    log = found.group(1)
    for token in ("PNG_OK", "MENU_OK", "SVG_LABEL_OK", "BLOCK_OK", "UP_OK", "DONE"):
        assert token in log, log
    for token in ("PNG_BAD", "MENU_BAD", "BLOCK_BAD", "UP_BAD", "VIDEO_BAD", "EXC "):
        assert token not in log, log
    assert "VIDEO_OK" in log or "VIDEO_SKIP" in log, log
    assert shot.exists() and shot.stat().st_size > 1000
