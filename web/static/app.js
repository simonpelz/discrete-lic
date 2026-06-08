"use strict";

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

async function loadGallery() {
  const grid = document.getElementById("gallery-grid");
  let items;
  try {
    const res = await fetch("/gallery");
    items = await res.json();
  } catch {
    grid.innerHTML = '<p style="color:var(--muted);font-family:var(--font-mono)">Could not load gallery.</p>';
    return;
  }

  if (!items.length) {
    grid.innerHTML = '<p style="color:var(--muted);font-family:var(--font-mono)">No gallery images yet. Run <code>python scripts/generate_gallery.py</code>.</p>';
    return;
  }

  grid.innerHTML = "";
  items.forEach(item => {
    const el = document.createElement("div");
    el.className = "gallery-item";
    el.innerHTML = `
      <img src="/gallery/${item.filename}" alt="${item.caption}" loading="lazy">
      <div class="gallery-caption">${item.caption}</div>
    `;
    el.querySelector("img").addEventListener("click", () => openLightbox(`/gallery/${item.filename}`));
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

// ── Generator controls ──────────────────────────────────────────

async function loadFields() {
  const sel = document.getElementById("field-select");
  try {
    const res = await fetch("/fields");
    const fields = await res.json();
    sel.innerHTML = fields.map(f =>
      `<option value="${f.name}"${f.requires_yt ? " data-yt" : ""}>${f.name}${f.requires_yt ? " (yt)" : ""}</option>`
    ).join("");
    sel.value = "rotation";
  } catch { /* fallback options already in HTML */ }
  updateConditionalControls();
}

function updateConditionalControls() {
  const field    = document.getElementById("field-select").value;
  const color    = document.getElementById("color-select").value;
  const pixelMode = document.querySelector(".toggle-btn.active[data-mode]")?.dataset.mode ?? "voronoi";

  // Complex expression
  toggle("complex-row", field === "complex");
  // yt fields notice
  toggle("yt-notice", field === "mhd_cluster" || field === "wd_merger");
  // Voronoi cells slider
  toggle("voronoi-cells-row", pixelMode === "voronoi");
  // Pure RGB options
  toggle("pure-opts", color === "pure_rgb");
  // Colormap picker
  toggle("colormap-row", color === "colormap");
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
  });
});

document.getElementById("field-select").addEventListener("change", updateConditionalControls);
document.getElementById("color-select").addEventListener("change", updateConditionalControls);

// Slider value display
document.querySelectorAll("input[type=range]").forEach(input => {
  const display = input.parentElement.querySelector(".slider-val");
  if (display) {
    display.textContent = input.value;
    input.addEventListener("input", () => { display.textContent = input.value; });
  }
});

// ── Generate ────────────────────────────────────────────────────

const btn      = document.getElementById("btn-generate");
const status   = document.getElementById("status-text");
const result   = document.getElementById("result-img");
const actions  = document.getElementById("result-actions");
const placeholder = document.getElementById("result-placeholder");

function getInt(id)   { return parseInt(document.getElementById(id).value, 10); }
function getFloat(id) { return parseFloat(document.getElementById(id).value); }
function getStr(id)   { return document.getElementById(id).value; }
function activeMode() { return document.querySelector(".toggle-btn.active[data-mode]")?.dataset.mode ?? "voronoi"; }

btn.addEventListener("click", async () => {
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Generating…';
  placeholder.style.display = "none";
  result.style.display = "none";
  actions.style.display = "none";

  const params = {
    field:             getStr("field-select"),
    field_expr:        getStr("field-expr"),
    color:             getStr("color-select"),
    pixel_mode:        activeMode(),
    steps:             getInt("steps-slider"),
    voronoi_cells:     getInt("cells-slider"),
    pure_mode:         getStr("pure-mode-select"),
    pure_sharpen:      getFloat("sharpen-slider"),
    colormap:          getStr("colormap-select"),
    width:             getInt("width-input"),
    height:            getInt("height-input") || null,
    seed:              getInt("seed-input"),
    kernel:            getStr("kernel-select"),
    enhance:           getStr("enhance-select"),
  };

  const t0 = Date.now();

  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || res.statusText);
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    result.src = url;
    result.style.display = "block";
    placeholder.style.display = "none";

    const dl = document.getElementById("btn-download");
    dl.href = url;
    dl.download = `lic_${params.field}_${params.color}_${params.pixel_mode}.png`;
    actions.style.display = "flex";

    status.textContent = `Done in ${elapsed} s`;
  } catch (err) {
    placeholder.style.display = "flex";
    placeholder.textContent = `Error: ${err.message}`;
    status.textContent = "";
  } finally {
    btn.disabled = false;
  }
});

// ── Init ────────────────────────────────────────────────────────

loadGallery();
loadFields();
