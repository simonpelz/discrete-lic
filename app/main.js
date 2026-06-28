"use strict";

import * as pipeline from "./gl/pipeline.js";

// ── Tab switching ───────────────────────────────────────────────

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.panel).classList.add("active");
  });
});

// ── Gallery ─────────────────────────────────────────────────────
//
// The Python web server is gone; we load a static gallery.json. Try the
// in-repo location first (web/gallery/), then a co-located gallery/ dir.
// Image filenames in the manifest are bare names, so we resolve them
// against the directory the manifest was found in.

const GALLERY_SOURCES = [
  { json: "../web/gallery/gallery.json", base: "../web/gallery/" },
  { json: "./gallery/gallery.json", base: "./gallery/" },
];

async function loadGallery() {
  const grid = document.getElementById("gallery-grid");

  let items = null;
  let base = "";
  for (const src of GALLERY_SOURCES) {
    try {
      const res = await fetch(src.json);
      if (!res.ok) continue;
      items = await res.json();
      base = src.base;
      break;
    } catch {
      /* try next source */
    }
  }

  if (items === null) {
    grid.innerHTML = '<p style="color:var(--muted);font-family:var(--font-mono)">No gallery manifest found. (gallery.json absent — this is fine.)</p>';
    return;
  }

  if (!Array.isArray(items) || !items.length) {
    grid.innerHTML = '<p style="color:var(--muted);font-family:var(--font-mono)">Gallery is empty.</p>';
    return;
  }

  grid.innerHTML = "";
  items.forEach(item => {
    const url = base + item.filename;
    const el = document.createElement("div");
    el.className = "gallery-item";
    const img = document.createElement("img");
    img.src = url;
    img.alt = item.caption || "";
    img.loading = "lazy";
    const cap = document.createElement("div");
    cap.className = "gallery-caption";
    cap.textContent = item.caption || item.filename;
    el.appendChild(img);
    el.appendChild(cap);
    img.addEventListener("click", () => openLightbox(url));
    grid.appendChild(el);
  });
}

// ── Lightbox ────────────────────────────────────────────────────

const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");

function openLightbox(src) {
  lightboxImg.src = src;
  lightbox.classList.add("open");
}

document.getElementById("lightbox-close").addEventListener("click", () => lightbox.classList.remove("open"));
lightbox.addEventListener("click", e => { if (e.target === lightbox) lightbox.classList.remove("open"); });
document.addEventListener("keydown", e => { if (e.key === "Escape") lightbox.classList.remove("open"); });

// ── Conditional controls ────────────────────────────────────────

function updateConditionalControls() {
  const field    = getStr("field-select");
  const color    = getStr("color-select");
  const kernel   = getStr("kernel-select");
  const pixelMode = activeMode();

  toggle("complex-row", field === "complex");
  toggle("yt-notice", field === "mhd_cluster" || field === "wd_merger");
  toggle("voronoi-cells-row", pixelMode === "voronoi");
  toggle("pure-opts", color === "pure_rgb");
  toggle("colormap-row", color === "colormap");
  toggle("kernel-sigma-row", kernel === "gaussian" || kernel === "raised_cosine");
}

function toggle(id, show) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("hidden", !show);
}

// Pixel mode toggle
document.querySelectorAll(".toggle-btn[data-mode]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".toggle-btn[data-mode]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    updateConditionalControls();
    scheduleRender();
  });
});

document.getElementById("field-select").addEventListener("change", () => { updateConditionalControls(); scheduleRender(); });
document.getElementById("color-select").addEventListener("change", () => { updateConditionalControls(); scheduleRender(); });
document.getElementById("kernel-select").addEventListener("change", () => { updateConditionalControls(); scheduleRender(); });

// Slider value display
document.querySelectorAll("input[type=range]").forEach(input => {
  const display = input.parentElement.querySelector(".slider-val");
  if (display) {
    display.textContent = input.value;
    input.addEventListener("input", () => { display.textContent = input.value; });
  }
});

// Live re-render on any control change (debounced)
document.querySelectorAll(".controls input, .controls select").forEach(el => {
  const ev = (el.tagName === "SELECT" || el.type === "number" || el.type === "text") ? "change" : "input";
  el.addEventListener(ev, scheduleRender);
});

// ── Param assembly ──────────────────────────────────────────────

function getInt(id)   { return parseInt(document.getElementById(id).value, 10); }
function getFloat(id) { return parseFloat(document.getElementById(id).value); }
function getStr(id)   { return document.getElementById(id).value; }
function activeMode() { return document.querySelector(".toggle-btn.active[data-mode]")?.dataset.mode ?? "voronoi"; }

function buildParams() {
  const heightRaw = getInt("height-input");
  return {
    field:         getStr("field-select"),
    field_expr:    getStr("field-expr"),
    color:         getStr("color-select"),
    pixel_mode:    activeMode(),
    noise:         getStr("noise-select"),
    steps:         getInt("steps-slider"),
    kernel:        getStr("kernel-select"),
    kernel_sigma:  getFloat("kernel-sigma-slider"),
    boundary:      getStr("boundary-select"),
    enhance:       getStr("enhance-select"),
    voronoi_cells: getInt("cells-slider"),
    pure_mode:     getStr("pure-mode-select"),
    pure_sharpen:  getFloat("sharpen-slider"),
    colormap:      getStr("colormap-select"),
    width:         getInt("width-input"),
    height:        Number.isFinite(heightRaw) ? heightRaw : null,
    seed:          getInt("seed-input"),
  };
}

// ── Render ──────────────────────────────────────────────────────

const btn         = document.getElementById("btn-generate");
const status      = document.getElementById("status-text");
const canvas      = document.getElementById("result-canvas");
const actions     = document.getElementById("result-actions");
const placeholder = document.getElementById("result-placeholder");

let ctx = null;        // pipeline context (created lazily, once)
let ctxError = null;   // init failure message, if any
let hasRendered = false;

function ensureCtx() {
  if (ctx || ctxError) return ctx;
  try {
    ctx = pipeline.init(canvas);
  } catch (err) {
    ctxError = err && err.message ? err.message : String(err);
  }
  return ctx;
}

async function render() {
  if (!ensureCtx()) {
    placeholder.style.display = "flex";
    placeholder.textContent = `WebGL2 error: ${ctxError}`;
    canvas.style.display = "none";
    actions.style.display = "none";
    status.textContent = "";
    return;
  }

  const params = buildParams();
  status.innerHTML = '<span class="spinner"></span> Rendering…';
  const t0 = performance.now();

  try {
    // pipeline.render* are async (they fetch shaders / colormaps / field assets
    // and await the color pass). Await so async errors surface here and the
    // success state is only shown once the canvas has actually been drawn.
    if (params.pixel_mode === "voronoi") {
      await pipeline.renderVoronoi(ctx, params);
    } else {
      await pipeline.renderGrid(ctx, params);
    }
  } catch (err) {
    placeholder.style.display = "flex";
    placeholder.textContent = `Render error: ${err && err.message ? err.message : err}`;
    canvas.style.display = "none";
    actions.style.display = "none";
    status.textContent = "";
    return;
  }

  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

  placeholder.style.display = "none";
  canvas.style.display = "block";
  actions.style.display = "flex";
  hasRendered = true;
  status.textContent = `Rendered in ${elapsed} s`;
}

// Debounced scheduler. Voronoi is heavier → longer debounce.
let renderTimer = null;
function scheduleRender() {
  if (!hasRendered) return;        // don't auto-render until the user has rendered once
  if (renderTimer) clearTimeout(renderTimer);
  const delay = activeMode() === "voronoi" ? 400 : 150;
  renderTimer = setTimeout(render, delay);
}

btn.addEventListener("click", () => {
  if (renderTimer) clearTimeout(renderTimer);
  render();
});

// ── Download ────────────────────────────────────────────────────

document.getElementById("btn-download").addEventListener("click", async (e) => {
  e.preventDefault();
  if (!ctx) return;
  try {
    const blob = await pipeline.exportPNG(ctx);
    if (!blob) throw new Error("export returned no blob");
    const p = buildParams();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lic_${p.field}_${p.color}_${p.pixel_mode}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (err) {
    status.textContent = `Download failed: ${err && err.message ? err.message : err}`;
  }
});

// ── Init ────────────────────────────────────────────────────────

updateConditionalControls();
loadGallery();
