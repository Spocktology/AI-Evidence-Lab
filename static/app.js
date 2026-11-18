/* AI Evidence Lab - front end (no build step). */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fileUrl = (caseId, rel) => `/api/cases/${caseId}/files/${rel}`;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ---------------------------------------------------------------------------
// Routing: "/" (home) and "#case/<id>/<step>"
// ---------------------------------------------------------------------------
const state = { caseData: null, step: "metadata", pollTimer: null };

function route() {
  clearInterval(state.pollTimer);
  const m = location.hash.match(/^#case\/([^/]+)(?:\/(\w+))?/);
  if (m) openCase(m[1], m[2] || null); else renderHome();
}
window.addEventListener("hashchange", route);
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-nav=home]");
  if (a) { e.preventDefault(); location.hash = ""; route(); }
});

async function checkHealth() {
  try {
    const h = await api("/api/health");
    const parts = [];
    parts.push(h.api_key_configured ? `<span class="ok">API key set</span>` : `<span class="bad">No ANTHROPIC_API_KEY - analysis disabled</span>`);
    if (!h.exiftool) parts.push(`<span class="bad">exiftool not installed (reduced metadata)</span>`);
    if (!h.ffmpeg) parts.push(`<span class="bad">ffmpeg not installed (no video)</span>`);
    parts.push(`<span class="muted">${esc(h.model)} · effort ${esc(h.effort)}</span>`);
    $("#health").innerHTML = parts.join(" · ");
    state.health = h;
  } catch (_) { $("#health").textContent = "Server not reachable"; }
}

// ---------------------------------------------------------------------------
// Home
// ---------------------------------------------------------------------------
async function renderHome() {
  const view = $("#view");
  view.replaceChildren($("#tpl-home").content.cloneNode(true));
  const drop = $("#drop"), input = $("#file");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
  input.addEventListener("change", () => { if (input.files[0]) upload(input.files[0]); });
  try {
    const list = await api("/api/cases");
    $("#case-list").innerHTML = list.length ? list.map((c) => `
      <a class="case-item" href="#case/${esc(c.id)}">
        <span><b>${esc(c.title || c.filename)}</b><br><span class="muted small">${esc(c.kind)} · ${esc(c.filename)} · ${esc((c.created_at || "").replace("T", " ").slice(0, 16))}</span></span>
        <span class="pill ${esc(c.status)}">${esc(c.status === "new" ? "not analysed" : c.status)}</span>
      </a>`).join("") : `<p class="muted">No cases yet.</p>`;
  } catch (e) { $("#case-list").textContent = e.message; }
}

function upload(file) {
  const status = $("#upload-status");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("title", $("#case-title").value || "");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/cases");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) status.textContent = `Uploading ${file.name}: ${Math.round((e.loaded / e.total) * 100)}%`; };
  xhr.upload.onload = () => { status.textContent = "Upload complete. Hashing the file and reading metadata..."; };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      const c = JSON.parse(xhr.responseText);
      location.hash = `#case/${c.id}/metadata`;
    } else {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail; } catch (_) { /* ignore */ }
      status.innerHTML = `<span style="color:var(--bad)">Upload failed: ${esc(msg)}</span>`;
    }
  };
  xhr.onerror = () => { status.innerHTML = `<span style="color:var(--bad)">Upload failed (network error).</span>`; };
  xhr.send(fd);
}

// ---------------------------------------------------------------------------
// Case
// ---------------------------------------------------------------------------
async function openCase(id, step) {
  let c;
  try { c = await api(`/api/cases/${id}`); } catch (e) { $("#view").innerHTML = `<div class="notice err">${esc(e.message)}</div>`; return; }
  state.caseData = c;
  state.step = step || (c.analysis?.status === "done" ? "analyze" : "metadata");
  const view = $("#view");
  view.replaceChildren($("#tpl-case").content.cloneNode(true));
  $("#case-id").textContent = `Case ${c.id} · ${c.kind} · ${c.filename}`;
  $("#case-title-h").textContent = c.title || c.filename;
  $("#steps").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-step]");
    if (b) location.hash = `#case/${c.id}/${b.dataset.step}`;
  });
  renderStep();
}

function markSteps() {
  const c = state.caseData;
  const done = {
    metadata: true,
    context: Object.values(c.context || {}).some(Boolean),
    mark: (c.selection?.items || []).some((i) => i.cells.length),
    analyze: c.analysis?.status === "done",
  };
  document.querySelectorAll("#steps button").forEach((b) => {
    b.classList.toggle("active", b.dataset.step === state.step);
    b.classList.toggle("done", !!done[b.dataset.step] && b.dataset.step !== state.step);
  });
}

function renderStep() {
  markSteps();
  ({ metadata: stepMetadata, context: stepContext, mark: stepMark, analyze: stepAnalyze }[state.step] || stepMetadata)();
}
const go = (step) => { location.hash = `#case/${state.caseData.id}/${step}`; };

// ---- Step 1: metadata -------------------------------------------------------
function stepMetadata() {
  const c = state.caseData;
  const body = $("#step-body");
  const media = c.kind === "image"
    ? `<img class="preview-media" src="${fileUrl(c.id, c.preview.path)}" alt="preview">`
    : `<video class="preview-media" controls preload="metadata" src="${fileUrl(c.id, c.playback === "original" ? "original/" + encodeURIComponent(c.stored_name) : c.playback)}"></video>`;
  const facts = Object.entries(c.facts || {}).map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td>${esc(Array.isArray(v) ? v.join(", ") : v)}</td></tr>`).join("");
  const flags = (c.flags || []).map((f) => `<div class="flag ${esc(f.level)}"><b>${esc(f.level)}</b> · ${esc(f.message)}</div>`).join("") || `<p class="muted">No issues flagged.</p>`;
  const missingCount = (c.questionnaire || []).filter((q) => q.required).length;
  body.innerHTML = `
    <div class="cols">
      <div class="panel">${media}
        <h3>File integrity</h3>
        <table class="kv">
          <tr><td>SHA-256</td><td style="font-family:monospace;font-size:12px">${esc(c.hashes.sha256)}</td></tr>
          <tr><td>MD5</td><td style="font-family:monospace;font-size:12px">${esc(c.hashes.md5)}</td></tr>
          <tr><td>Size</td><td>${(c.size_bytes / 1048576).toFixed(2)} MB</td></tr>
          <tr><td>Metadata tags found</td><td>${c.metadata_tag_count} (via ${esc(c.metadata_source)})</td></tr>
        </table>
      </div>
      <div class="panel">
        <h2>What the file tells us</h2>
        <table class="kv">${facts || `<tr><td colspan=2 class="muted">No technical metadata found.</td></tr>`}</table>
        <h3>Automatic checks</h3>${flags}
        <h3>Content Credentials (C2PA)</h3>
        <p class="small">${c.c2pa?.present ? `Manifest present - generator: <b>${esc(c.c2pa.claim_generator)}</b>, signer: ${esc(c.c2pa.signature_issuer)}, source types: ${esc((c.c2pa.digital_source_types || []).join(", ") || "none declared")}` : "No C2PA manifest."}</p>
        <details><summary>Show all ${c.metadata_tag_count} raw metadata tags</summary><pre class="raw" id="raw-meta">Loading...</pre></details>
      </div>
    </div>
    <div class="actions">
      ${missingCount ? `<div class="notice warn" style="margin:0">The metadata is missing ${missingCount} item(s) the AI needs. Please fill them in on the next step.</div>` : ""}
      <button class="btn" id="next">Next: capture details &rarr;</button>
    </div>`;
  $("#next").onclick = () => go("context");
  body.querySelector("details").addEventListener("toggle", async (e) => {
    if (!e.target.open) return;
    const m = await api(`/api/cases/${c.id}/metadata`);
    $("#raw-meta").textContent = JSON.stringify(m.tags, null, 1) + (m.ffprobe ? "\n\nffprobe:\n" + JSON.stringify(m.ffprobe, null, 1) : "");
  }, { once: true });
}

// ---- Step 2: capture context -----------------------------------------------
function stepContext() {
  const c = state.caseData;
  const ctx = c.context || {};
  const fields = (c.questionnaire || []).map((q) => {
    const val = ctx[q.key] ?? (q.known != null ? String(q.known) : "");
    const badge = q.known != null ? `<span class="badge known">from file metadata</span>` : q.required ? `<span class="badge req">${["provenance", "source_chain"].includes(q.key) ? "please answer" : "needed - not in metadata"}</span>` : "";
    let input;
    if (q.type === "select") {
      input = `<select name="${esc(q.key)}"><option value="">- choose -</option>${q.options.map((o) => `<option ${o === val ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
    } else if (q.type === "textarea") {
      input = `<textarea name="${esc(q.key)}">${esc(val)}</textarea>`;
    } else {
      input = `<input type="text" name="${esc(q.key)}" value="${esc(val)}">`;
    }
    return `<div class="field ${q.type === "textarea" || q.key === "source_chain" ? "wide" : ""}">
      <label>${esc(q.label)}${badge}</label>${q.help ? `<p class="help">${esc(q.help)}</p>` : ""}${input}</div>`;
  }).join("");
  $("#step-body").innerHTML = `
    <div class="panel">
      <h2>How was this captured?</h2>
      <p class="muted small">Answer what you know - "unknown" is a fine answer. Fields marked <span class="badge req">needed</span> could not be read from the file. Values marked <span class="badge known">from file metadata</span> were read automatically; correct them if they are wrong.</p>
      <form id="ctx" class="form-grid">${fields}</form>
      <div class="actions"><button class="btn" id="save">Save and continue &rarr;</button><span id="msg" class="small muted"></span></div>
    </div>`;
  $("#save").onclick = async () => {
    const answers = Object.fromEntries(new FormData($("#ctx")).entries());
    try {
      state.caseData = await postJSON(`/api/cases/${c.id}/context`, { answers });
      go("mark");
    } catch (e) { $("#msg").textContent = e.message; }
  };
}

// ---- Step 3: mark the object ---------------------------------------------
function colName(i) { let s = ""; i += 1; while (i) { const r = (i - 1) % 26; s = String.fromCharCode(65 + r) + s; i = Math.floor((i - 1) / 26); } return s; }
function gridRows(cols, w, h) { return Math.max(2, Math.round((cols * h) / w)); }

/** Interactive grid on a canvas: click or drag to toggle cells. */
class GridPicker {
  constructor(container, imgUrl, fullW, fullH, cols, selected, onChange) {
    this.fullW = fullW; this.fullH = fullH; this.cols = cols; this.rows = gridRows(cols, fullW, fullH);
    this.selected = new Set(selected || []); this.suggested = new Set(); this.onChange = onChange;
    this.canvas = document.createElement("canvas");
    container.replaceChildren(this.canvas);
    this.img = new Image();
    this.img.onload = () => this.draw();
    this.img.src = imgUrl;
    let dragging = false, mode = true;
    const cellAt = (e) => {
      const r = this.canvas.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * this.fullW, y = ((e.clientY - r.top) / r.height) * this.fullH;
      const c = Math.min(this.cols - 1, Math.max(0, Math.floor((x * this.cols) / this.fullW)));
      const rr = Math.min(this.rows - 1, Math.max(0, Math.floor((y * this.rows) / this.fullH)));
      return colName(c) + (rr + 1);
    };
    this.canvas.addEventListener("pointerdown", (e) => {
      dragging = true; this.canvas.setPointerCapture(e.pointerId);
      const cell = cellAt(e); mode = !this.selected.has(cell); this.set(cell, mode);
    });
    this.canvas.addEventListener("pointermove", (e) => { if (dragging) this.set(cellAt(e), mode); });
    this.canvas.addEventListener("pointerup", () => { dragging = false; });
  }
  set(cell, on) {
    const had = this.selected.has(cell);
    if (on && !had) this.selected.add(cell); else if (!on && had) this.selected.delete(cell); else return;
    this.draw(); this.onChange?.([...this.selected]);
  }
  setCols(cols) { this.cols = cols; this.rows = gridRows(cols, this.fullW, this.fullH); this.selected.clear(); this.draw(); this.onChange?.([]); }
  setSelected(cells) { this.selected = new Set(cells); this.draw(); this.onChange?.([...this.selected]); }
  draw() {
    if (!this.img.complete || !this.img.naturalWidth) return;
    const cv = this.canvas, W = this.img.naturalWidth, H = this.img.naturalHeight;
    cv.width = W; cv.height = H;
    const g = cv.getContext("2d");
    g.drawImage(this.img, 0, 0, W, H);
    const cw = W / this.cols, ch = H / this.rows;
    const fs = Math.max(10, Math.min(cw, ch) * 0.2);
    g.font = `bold ${fs}px sans-serif`; g.textBaseline = "top";
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const name = colName(c) + (r + 1);
        const x0 = Math.round((c * W) / this.cols), y0 = Math.round((r * H) / this.rows);
        const x1 = Math.round(((c + 1) * W) / this.cols), y1 = Math.round(((r + 1) * H) / this.rows);
        if (this.selected.has(name)) {
          g.fillStyle = "rgba(255,70,70,.30)"; g.fillRect(x0, y0, x1 - x0, y1 - y0);
          g.strokeStyle = "#ff4646"; g.lineWidth = Math.max(2, fs / 6); g.setLineDash([]); g.strokeRect(x0 + 1, y0 + 1, x1 - x0 - 2, y1 - y0 - 2);
        } else {
          g.strokeStyle = "rgba(255,255,255,.35)"; g.lineWidth = 1; g.setLineDash([]); g.strokeRect(x0 + .5, y0 + .5, x1 - x0, y1 - y0);
        }
        if (this.suggested.has(name) && !this.selected.has(name)) {
          g.strokeStyle = "#ffd84a"; g.lineWidth = Math.max(2, fs / 7); g.setLineDash([fs / 2, fs / 3]); g.strokeRect(x0 + 2, y0 + 2, x1 - x0 - 4, y1 - y0 - 4); g.setLineDash([]);
        }
        g.lineWidth = Math.max(2, fs / 5); g.strokeStyle = "rgba(0,0,0,.85)"; g.strokeText(name, x0 + 4, y0 + 3);
        g.fillStyle = "rgba(255,240,80,.95)"; g.fillText(name, x0 + 4, y0 + 3);
      }
    }
  }
}

function stepMark() { if (state.caseData.kind === "image") markImage(); else markVideo(); }

function densityOptions(current) {
  return [4, 6, 8, 10, 12, 16, 20, 26].map((n) => `<option value="${n}" ${n === current ? "selected" : ""}>${n} columns</option>`).join("");
}

function markImage() {
  const c = state.caseData;
  const sel = c.selection || { grid_cols: 8, items: [] };
  let cols = sel.grid_cols || 8;
  const cells = sel.items.find((i) => i.frame_id === "IMG")?.cells || [];
  $("#step-body").innerHTML = `
    <div class="panel">
      <h2>Where in the photo is the object or event?</h2>
      <p class="muted small">Click or drag across the grid squares that contain what you want analysed. Use a finer grid for small or distant objects. The AI will receive these squares at <b>full original resolution</b>, plus enhanced and pixel-level views. If you mark nothing, the whole photo is examined in full-resolution tiles.</p>
      <div class="toolbar">
        <div class="field"><label>Grid density</label><select id="density">${densityOptions(cols)}</select></div>
        <button class="btn ghost small" id="clear">Clear selection</button>
        <div class="legend"><span class="l-sel">Marked</span></div>
      </div>
      <div class="grid-wrap" id="gw"></div>
      <div class="chips" id="chips"></div>
      <div class="actions"><button class="btn" id="save">Save and continue &rarr;</button><span id="msg" class="small muted"></span></div>
    </div>`;
  const chips = (list) => { $("#chips").innerHTML = list.length ? list.sort().map((x) => `<span class="chip">${esc(x)}</span>`).join("") : `<span class="muted small">Nothing marked yet.</span>`; };
  const picker = new GridPicker($("#gw"), fileUrl(c.id, c.preview.path), c.facts.width, c.facts.height, cols, cells, chips);
  chips(cells.slice());
  $("#density").onchange = (e) => { cols = +e.target.value; picker.setCols(cols); };
  $("#clear").onclick = () => picker.setSelected([]);
  $("#save").onclick = async () => {
    try {
      state.caseData = await postJSON(`/api/cases/${c.id}/selection`, { grid_cols: cols, items: [{ frame_id: "IMG", cells: [...picker.selected] }] });
      go("analyze");
    } catch (e) { $("#msg").textContent = e.message; }
  };
}

function markVideo() {
  const c = state.caseData;
  const dur = c.facts.duration_s || 0;
  const sel = c.selection || { grid_cols: 8, items: [] };
  const vs = { cols: sel.grid_cols || c.grid_cols || 8, frames: c.frames || [], cells: {}, active: null, suggested: c.motion_suggestions || [], picker: null };
  sel.items.forEach((i) => { vs.cells[i.frame_id] = i.cells; });
  const rng = c.video_range || { start: 0, end: Math.min(dur, 5) };
  const src = fileUrl(c.id, c.playback === "original" ? "original/" + encodeURIComponent(c.stored_name) : c.playback);
  $("#step-body").innerHTML = `
    <div class="panel">
      <h2>When and where in the video?</h2>
      <p class="muted small">1) Play the video and set the <b>start</b> and <b>end</b> of the moment you care about. 2) Pull frames from that range: they are extracted from the original at full resolution. 3) On each frame, mark the grid squares containing the object. Squares where motion was detected are outlined in <span style="color:#ffd84a">yellow</span> as suggestions.</p>
      <video id="vid" class="preview-media" controls preload="metadata" src="${src}"></video>
      <div class="toolbar" style="margin-top:12px">
        <div class="field"><label>Start (s)</label><input type="number" id="t0" step="0.01" min="0" max="${dur}" value="${(+rng.start).toFixed(2)}"></div>
        <button class="btn ghost small" id="set0">Use current time</button>
        <div class="field"><label>End (s)</label><input type="number" id="t1" step="0.01" min="0" max="${dur}" value="${(+rng.end).toFixed(2)}"></div>
        <button class="btn ghost small" id="set1">Use current time</button>
        <div class="field"><label>Frames to pull</label><select id="count">${[4, 6, 8, 12, 16, 20, 24].map((n) => `<option ${n === (vs.frames.length || 12) ? "selected" : ""}>${n}</option>`).join("")}</select></div>
        <div class="field"><label>Grid density</label><select id="density">${densityOptions(vs.cols)}</select></div>
        <button class="btn" id="pull">Pull frames</button>
      </div>
      <p class="muted small">Duration ${dur ? dur.toFixed(2) + " s" : "unknown"}${c.facts.fps_average ? ` · ${c.facts.fps_average} fps` : ""}. If the range contains fewer frames than requested, every frame in the range is pulled.</p>
      <div id="pull-msg" class="small muted"></div>
      <div id="frames-area" style="display:none">
        <div class="frames-strip" id="strip"></div>
        <div class="toolbar">
          <button class="btn ghost small" id="copy-all">Copy this frame's marks to all frames</button>
          <button class="btn ghost small" id="use-sug">Use motion suggestions on this frame</button>
          <button class="btn ghost small" id="clear">Clear this frame</button>
          <div class="legend"><span class="l-sel">Marked</span><span class="l-sug">Motion detected</span></div>
        </div>
        <div class="grid-wrap" id="gw"></div>
        <div class="chips" id="chips"></div>
        <details id="heat-d" style="display:none"><summary>Motion heat-map for this range</summary><img id="heat" class="preview-media" style="margin-top:8px"></details>
      </div>
      <div class="actions"><button class="btn" id="save" ${vs.frames.length ? "" : "disabled"}>Save and continue &rarr;</button><span id="msg" class="small muted"></span></div>
    </div>`;
  const vid = $("#vid");
  $("#set0").onclick = () => { $("#t0").value = vid.currentTime.toFixed(2); };
  $("#set1").onclick = () => { $("#t1").value = vid.currentTime.toFixed(2); };
  $("#density").onchange = (e) => {
    vs.cols = +e.target.value; vs.cells = {};
    if (vs.picker) vs.picker.setCols(vs.cols);
    renderStrip();
  };

  const chips = (list) => { $("#chips").innerHTML = list.length ? list.sort().map((x) => `<span class="chip">${esc(x)}</span>`).join("") : `<span class="muted small">Nothing marked on this frame.</span>`; };
  function renderStrip() {
    $("#strip").innerHTML = vs.frames.map((f) => `
      <div class="frame-thumb ${f.id === vs.active ? "active" : ""} ${(vs.cells[f.id] || []).length ? "marked" : ""}" data-id="${esc(f.id)}">
        <img src="${fileUrl(c.id, f.preview)}" alt=""><div class="lbl"><span>${esc(f.id)} · ${f.t.toFixed(2)}s</span></div></div>`).join("");
  }
  function showFrame(id) {
    vs.active = id;
    const f = vs.frames.find((x) => x.id === id);
    vs.picker = new GridPicker($("#gw"), fileUrl(c.id, f.preview), f.width, f.height, vs.cols, vs.cells[id] || [], (list) => {
      vs.cells[id] = list; chips(list.slice());
      document.querySelector(`.frame-thumb[data-id="${id}"]`)?.classList.toggle("marked", list.length > 0);
    });
    vs.picker.suggested = new Set(vs.suggested);
    chips((vs.cells[id] || []).slice());
    renderStrip();
  }
  function showFrames() {
    $("#frames-area").style.display = vs.frames.length ? "" : "none";
    $("#save").disabled = !vs.frames.length;
    if (vs.frames.length) showFrame(vs.active && vs.frames.some((f) => f.id === vs.active) ? vs.active : vs.frames[0].id);
  }
  $("#strip").addEventListener("click", (e) => { const t = e.target.closest(".frame-thumb"); if (t) showFrame(t.dataset.id); });
  $("#copy-all").onclick = () => { const cur = vs.cells[vs.active] || []; vs.frames.forEach((f) => { vs.cells[f.id] = cur.slice(); }); renderStrip(); };
  $("#use-sug").onclick = () => { if (vs.picker) vs.picker.setSelected([...new Set([...(vs.cells[vs.active] || []), ...vs.suggested])]); };
  $("#clear").onclick = () => { if (vs.picker) vs.picker.setSelected([]); };
  $("#pull").onclick = async () => {
    const t0 = +$("#t0").value, t1 = +$("#t1").value;
    if (!(t1 >= t0)) { $("#pull-msg").textContent = "End must be after start."; return; }
    $("#pull").disabled = true;
    $("#pull-msg").textContent = "Extracting full-resolution frames and scanning for motion... (large 4K files can take a minute)";
    try {
      const r = await postJSON(`/api/cases/${c.id}/frames`, { start: t0, end: t1, count: +$("#count").value, grid_cols: vs.cols });
      vs.frames = r.frames; vs.cells = {}; vs.suggested = r.suggested_cells || []; vs.active = null;
      $("#pull-msg").textContent = `${r.frames.length} frames pulled.` + (vs.suggested.length ? ` Motion detected in ${vs.suggested.join(", ")}.` : " No distinct motion detected (camera-shake removed).");
      if (r.heatmap) { $("#heat").src = fileUrl(c.id, r.heatmap) + "?t=" + Date.now(); $("#heat-d").style.display = ""; }
      state.caseData = await api(`/api/cases/${c.id}`);
      showFrames();
    } catch (e) { $("#pull-msg").textContent = "Failed: " + e.message; }
    $("#pull").disabled = false;
  };
  $("#save").onclick = async () => {
    const items = vs.frames.map((f) => ({ frame_id: f.id, cells: vs.cells[f.id] || [] }));
    if (!items.some((i) => i.cells.length) && !confirm("No grid squares are marked. Continue anyway? The AI will examine the frames without a marked region.")) return;
    try {
      state.caseData = await postJSON(`/api/cases/${c.id}/selection`, { grid_cols: vs.cols, items });
      go("analyze");
    } catch (e) { $("#msg").textContent = e.message; }
  };
  if (vs.frames.length) showFrames();
}

// ---- Step 4: analysis -----------------------------------------------------
function stepAnalyze() {
  const c = state.caseData;
  const a = c.analysis || { status: "new" };
  const missing = (c.questionnaire || []).filter((q) => q.required && !(c.context || {})[q.key]);
  const marked = (c.selection?.items || []).filter((i) => i.cells.length);
  const noKey = state.health && !state.health.api_key_configured;
  $("#step-body").innerHTML = `
    <div class="panel">
      <h2>AI analysis</h2>
      ${missing.length ? `<div class="notice warn">Not yet answered: ${missing.map((q) => esc(q.label)).join("; ")}. The analysis will be less certain without them. <a href="#case/${c.id}/context">Add details</a></div>` : ""}
      ${c.kind === "video" && !(c.frames || []).length ? `<div class="notice err">Pull frames from the video first (step 3).</div>` : ""}
      ${!marked.length ? `<div class="notice warn">No region marked: the AI will examine the whole ${c.kind === "image" ? "photo" : "frames"} without a focus area.</div>` : ""}
      ${noKey ? `<div class="notice err">No ANTHROPIC_API_KEY is configured on the server. Add it to the .env file and restart.</div>` : ""}
      <p class="small muted">What happens: the AI first examines the media, metadata and forensic measurements <b>without</b> your account of what you saw (so it can't be influenced by it), then compares your account with its findings and writes the report. Expect 2-8 minutes.</p>
      <label class="checkbox"><input type="checkbox" id="consent"> <span>I understand the media, its metadata and my answers will be sent to Anthropic's API for analysis.</span></label>
      <div class="actions">
        <button class="btn" id="run" ${noKey ? "disabled" : ""}>${a.status === "done" ? "Re-run analysis" : "Run AI analysis"}</button>
        ${a.status === "done" ? `<a class="btn ghost" href="/cases/${c.id}/report" target="_blank">Open report in new tab</a>` : ""}
      </div>
      <div id="prog" style="margin-top:16px; ${["running", "queued"].includes(a.status) ? "" : "display:none"}">
        <div class="progress"><div id="bar"></div></div>
        <p class="small" id="stepmsg"></p>
      </div>
      <div id="err"></div>
    </div>
    <div id="report-area" style="margin-top:16px"></div>`;
  $("#run").onclick = async () => {
    if (!$("#consent").checked) { $("#err").innerHTML = `<div class="notice warn">Please tick the box to confirm.</div>`; return; }
    $("#err").innerHTML = "";
    try { await postJSON(`/api/cases/${c.id}/analyze`, {}); poll(); } catch (e) { $("#err").innerHTML = `<div class="notice err">${esc(e.message)}</div>`; }
  };
  if (["running", "queued"].includes(a.status)) poll();
  if (a.status === "error") $("#err").innerHTML = `<div class="notice err">Last run failed: ${esc(a.error)}</div>`;
  if (a.status === "done") showReport();
}

function poll() {
  const c = state.caseData;
  $("#prog").style.display = "";
  $("#run").disabled = true;
  clearInterval(state.pollTimer);
  const tick = async () => {
    try {
      const s = await api(`/api/cases/${c.id}/status`);
      $("#bar").style.width = (s.progress || 0) + "%";
      $("#stepmsg").textContent = s.step || s.status;
      if (s.status === "done" || s.status === "error") {
        clearInterval(state.pollTimer);
        state.caseData = await api(`/api/cases/${c.id}`);
        stepAnalyze();
      }
    } catch (_) { /* keep polling */ }
  };
  tick();
  state.pollTimer = setInterval(tick, 2500);
}

function showReport() {
  const c = state.caseData;
  $("#report-area").innerHTML = `<iframe class="report" src="/cases/${c.id}/report"></iframe>`;
}

checkHealth();
route();
