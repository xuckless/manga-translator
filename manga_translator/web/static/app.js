/* Manga Translator web UI. No framework, no build step. */
"use strict";

// ------------------------------------------------------------------ helpers
const $ = (sel, el = document) => el.querySelector(sel);
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// replaceChildren, minus the null/false placeholders of optional panels.
const mount = (el, ...nodes) => el.replaceChildren(...nodes.flat().filter((n) => n != null && n !== false));

function toast(msg, isError = false) {
  const el = h("div", { class: isError ? "error" : "" }, msg);
  $("#toast").append(el);
  setTimeout(() => el.remove(), isError ? 6000 : 3000);
}

const enc = encodeURIComponent;
const img = (ch, kind, page, v) => `/img/${enc(ch)}/${kind}/${enc(page)}?v=${v || 0}`;

// Browser storage can be unavailable (private mode, blocked site data); never let that break the app.
const storage = {
  get(k, area = localStorage) { try { return area.getItem(k); } catch { return null; } },
  set(k, v, area = localStorage) { try { v == null ? area.removeItem(k) : area.setItem(k, v); } catch {} },
};
const keyStore = {
  get() { return storage.get("mt.key") || storage.get("mt.key", sessionStorage) || ""; },
  set(key, remember) {
    storage.set("mt.key", null); storage.set("mt.key", null, sessionStorage);
    if (key) storage.set("mt.key", key, remember ? localStorage : sessionStorage);
  },
  remembered() { return !!storage.get("mt.key"); },
};

async function api(path, { method = "GET", body, headers = {}, form } = {}) {
  const opts = { method, headers: { ...headers }, credentials: "same-origin" };
  if (form) opts.body = form;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const res = await fetch(path, opts);
  if (res.status === 401 && path !== "/api/login") { showLogin(); throw new Error("Sign in required"); }
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(data?.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText);
  return data;
}

let CONFIG = null;
async function config() {
  if (!CONFIG) CONFIG = await api("/api/config");
  return CONFIG;
}

// ------------------------------------------------------------------- router
let cleanup = null;   // teardown for the current view (timers, key handlers)

async function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  document.body.classList.remove("reading");
  const view = $("#view");
  const hash = decodeURIComponent(location.hash.slice(1) || "/");
  const [path, query] = hash.split("?");
  const params = new URLSearchParams(query || "");
  const parts = path.split("/").filter(Boolean);
  try {
    if (parts[0] === "c" && parts.length === 2) await chapterView(view, parts[1]);
    else if (parts[0] === "c" && parts[2] === "edit") await editorView(view, parts[1], parts.slice(3).join("/"));
    else if (parts[0] === "c" && parts[2] === "read") await readerView(view, parts[1], parseInt(params.get("p") || "", 10));
    else await libraryView(view);
  } catch (e) {
    mount(view, h("div", { class: "empty" }, h("p", {}, e.message), h("a", { href: "#/" }, "Back to library")));
  }
  updateKeyDot();
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ library
async function libraryView(view) {
  const chapters = await api("/api/chapters");
  const input = h("input", { type: "text", placeholder: "New chapter name, e.g. My Manga - Ch 1", maxlength: 80, "aria-label": "New chapter name" });
  const create = async (e) => {
    e.preventDefault();
    const name = input.value.trim();
    if (!name) return input.focus();
    try { await api("/api/chapters", { method: "POST", body: { name } }); location.hash = `#/c/${enc(name)}`; }
    catch (err) { toast(err.message, true); }
  };
  const grid = chapters.length
    ? h("div", { class: "grid" }, chapters.map((c) => {
        const pct = c.pages ? Math.round((100 * c.rendered) / c.pages) : 0;
        return h("a", { class: "card chapter-card", href: `#/c/${enc(c.name)}` },
          c.cover ? h("img", { class: "cover", loading: "lazy", alt: "", src: img(c.name, "thumb", c.cover, c.cover_v) }) : h("div", { class: "cover" }),
          h("div", { class: "meta" },
            h("div", { class: "title", title: c.name }, c.name),
            h("div", { class: "muted", style: { fontSize: ".8rem" } },
              c.pages ? `${c.pages} pages · ${c.rendered} translated` : "No pages yet"),
            h("div", { class: "progress" }, h("i", { style: { width: pct + "%" } }))));
      }))
    : h("div", { class: "card empty" }, h("p", {}, "No chapters yet. Create one, then add its pages."));
  mount(view, 
    h("div", { class: "page-head" }, h("h1", {}, "Library")),
    h("form", { class: "new-chapter", onsubmit: create, style: { marginBottom: "1.25rem" } }, input, h("button", {}, "Create")),
    grid);
}

// ------------------------------------------------------------------ chapter
async function chapterView(view, name) {
  const cfg = await config();
  let data = await api(`/api/chapters/${enc(name)}`);
  let staged = [];          // [{file, url}] picked but not uploaded yet
  let pollTimer = null;
  let lastJobId = null;

  const fileInput = h("input", { type: "file", multiple: true, accept: "image/png,image/jpeg,image/webp,image/*", hidden: true,
    onchange: () => { addFiles([...fileInput.files]); fileInput.value = ""; } });

  function addFiles(files) {
    const imgs = files.filter((f) => f.type.startsWith("image/"));
    if (!imgs.length) return toast("Pick image files (JPG, PNG or WebP).", true);
    // File pickers return files in arbitrary order: start from natural filename order (page2 before page10).
    imgs.sort((a, b) => collator.compare(a.name, b.name));
    staged.push(...imgs.map((file) => ({ file, url: URL.createObjectURL(file) })));
    render();
  }

  function clearStaged() { staged.forEach((s) => URL.revokeObjectURL(s.url)); staged = []; }

  async function upload(thenTranslate) {
    const total = staged.length;
    const batch = 4;
    const btns = view.querySelectorAll(".staging button");
    btns.forEach((b) => (b.disabled = true));
    const status = $(".staging .upload-status", view);
    try {
      // Sequential batches keep the order exactly as arranged and give visible progress.
      for (let i = 0; i < total; i += batch) {
        status.textContent = `Uploading ${Math.min(i + batch, total)} of ${total}…`;
        const form = new FormData();
        staged.slice(i, i + batch).forEach((s) => form.append("files", s.file, s.file.name));
        await api(`/api/chapters/${enc(name)}/pages`, { method: "POST", form });
      }
      clearStaged();
      toast(`Added ${total} page${total === 1 ? "" : "s"}.`);
      await refresh();
      if (thenTranslate) await startJob("translate");
    } catch (e) {
      toast(e.message, true);
      await refresh();
    }
  }

  async function startJob(action, extra = {}) {
    const key = keyStore.get();
    if (action === "translate" && !key && !cfg.server_key) {
      toast("Add your Anthropic API key in Settings first.", true);
      openSettings();
      return;
    }
    try {
      const job = await api(`/api/chapters/${enc(name)}/jobs`, {
        method: "POST", body: { action, ...extra }, headers: key ? { "X-Anthropic-Key": key } : {},
      });
      data.job = job;
      render();
      poll();
    } catch (e) { toast(e.message, true); }
  }

  async function refresh() {
    data = await api(`/api/chapters/${enc(name)}`);
    render();
  }

  function poll() {
    clearTimeout(pollTimer);
    if (!data.job) return;
    lastJobId = data.job.id;
    pollTimer = setTimeout(async () => {
      try {
        const job = await api(`/api/jobs/${data.job.id}`);
        data.job = job;
        if (job.status === "running" || job.status === "queued") { renderJob(); poll(); }
        else {
          const final = job;
          data = await api(`/api/chapters/${enc(name)}`);
          data.lastJob = final;
          render();
          if (final.status === "done") toast(final.message || "Finished.");
          else if (final.status === "failed") toast(final.message, true);
        }
      } catch (e) { poll(); }
    }, 1000);
  }

  async function saveSetting(k, v) {
    try { data.settings = await api(`/api/chapters/${enc(name)}/settings`, { method: "PUT", body: { [k]: v } }); }
    catch (e) { toast(e.message, true); }
  }

  function settingsPanel() {
    const s = data.settings;
    const sel = (k, label, options) => h("label", { class: "field" }, h("span", {}, label),
      h("select", { onchange: (e) => saveSetting(k, e.target.value) },
        options.map(([v, t]) => h("option", { value: v, selected: s[k] === v }, t))));
    const chk = (k, label) => h("label", { class: "check" },
      h("input", { type: "checkbox", checked: !!s[k], onchange: (e) => saveSetting(k, e.target.checked) }), label);
    return h("details", { class: "card panel" },
      h("summary", {}, "Translation & lettering options"),
      h("div", { class: "settings-grid" },
        sel("model", "Model", cfg.models.map((m) => [m.id, m.label])),
        sel("effort", "Effort", [["low", "Low"], ["medium", "Medium"], ["high", "High"]]),
        sel("honorifics", "Honorifics", [["keep", "Keep (-san, -sama)"], ["localize", "Localize (Mr., Lord)"]]),
        sel("font", "Font", cfg.fonts.map((f) => [f, f.replace(/\.(ttf|otf)$/i, "")])),
        sel("inpaint", "Clean art with", [["lama", "LaMa (best)"], ["opencv", "OpenCV (fast)"]]),
        h("div", { style: { display: "grid", gap: ".35rem" } }, chk("uppercase", "ALL CAPS lettering"), chk("sfx", "Replace sound effects"))),
      h("p", { class: "hint" }, "Changing lettering options only needs “Re-letter” — no new translation."));
  }

  function jobPanel() {
    const job = data.job || data.lastJob;
    if (!job) return null;
    const active = job.status === "running" || job.status === "queued";
    const stageName = { detect: "Finding text", translate: "Translating", render: "Lettering", done: "Finished" }[job.stage] || "Waiting";
    const pct = job.total ? Math.round((100 * job.done) / job.total) : (job.stage === "done" ? 100 : 0);
    const text = active
      ? `${job.status === "queued" ? "Queued" : stageName}${job.total ? ` · page ${Math.min(job.done + 1, job.total)} of ${job.total}` : ""}${job.page ? ` · ${job.page}` : ""}`
      : job.status === "done" ? (job.message || "Finished") : job.status === "cancelled" ? "Cancelled" : `Failed: ${job.message}`;
    return h("div", { class: `card job ${active ? "" : job.status}`, id: "job" },
      h("div", { class: "row" },
        h("strong", {}, text), h("span", { class: "spacer" }),
        active ? h("button", { class: "secondary small", onclick: async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); } }, "Cancel")
               : h("button", { class: "ghost small", onclick: () => { data.lastJob = null; render(); } }, "Dismiss")),
      h("div", { class: "progress" }, h("i", { style: { width: pct + "%" } })),
      job.log?.length ? h("details", {}, h("summary", { class: "muted", style: { fontSize: ".8rem", marginTop: ".4rem", cursor: "pointer" } }, "Log"),
        h("pre", { class: "log" }, job.log.slice(-60).join("\n"))) : null);
  }

  function renderJob() {
    const old = $("#job", view);
    const next = jobPanel();
    if (old && next) {
      const wasOpen = old.querySelector("details")?.open;
      if (wasOpen) next.querySelector("details")?.setAttribute("open", "");
      old.replaceWith(next);
    } else render();
  }

  function stagingPanel() {
    if (!staged.length) return null;
    let dragFrom = null;
    const move = (from, to) => { const [it] = staged.splice(from, 1); staged.splice(to, 0, it); render(); };
    const start = data.pages.length;
    return h("div", { class: "card staging" },
      h("div", { class: "row" },
        h("strong", {}, `${staged.length} page${staged.length === 1 ? "" : "s"} ready to add`),
        h("span", { class: "dropzone-hint" }, "Drag to reorder. They are added after the current pages, in this order."),
        h("span", { class: "spacer" }),
        h("button", { class: "ghost small", onclick: () => { staged.sort((a, b) => collator.compare(a.file.name, b.file.name)); render(); } }, "Sort by name"),
        h("button", { class: "ghost small", onclick: () => { staged.reverse(); render(); } }, "Reverse")),
      h("div", { class: "staged" }, staged.map((s, i) => h("div", {
          class: "item", draggable: "true",
          ondragstart: (e) => { dragFrom = i; e.currentTarget.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; },
          ondragend: (e) => e.currentTarget.classList.remove("dragging"),
          ondragover: (e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); },
          ondragleave: (e) => e.currentTarget.classList.remove("drag-over"),
          ondrop: (e) => { e.preventDefault(); if (dragFrom !== null && dragFrom !== i) move(dragFrom, i); },
        },
        h("img", { src: s.url, alt: "" }),
        h("span", { class: "num" }, start + i + 1),
        h("span", { class: "ctl" },
          i > 0 ? h("button", { title: "Move earlier", "aria-label": "Move earlier", onclick: () => move(i, i - 1) }, "◀") : null,
          i < staged.length - 1 ? h("button", { title: "Move later", "aria-label": "Move later", onclick: () => move(i, i + 1) }, "▶") : null,
          h("button", { title: "Remove", "aria-label": "Remove", onclick: () => { URL.revokeObjectURL(s.url); staged.splice(i, 1); render(); } }, "✕")),
        h("div", { class: "fname", title: s.file.name }, s.file.name)))),
      h("div", { class: "row" },
        h("button", { onclick: () => upload(true) }, "Add & translate"),
        h("button", { class: "secondary", onclick: () => upload(false) }, "Add only"),
        h("button", { class: "ghost", onclick: () => fileInput.click() }, "Pick more…"),
        h("span", { class: "spacer" }),
        h("span", { class: "upload-status muted" }),
        h("button", { class: "ghost danger", onclick: () => { clearStaged(); render(); } }, "Discard")));
  }

  function pageGrid() {
    if (!data.pages.length) {
      return h("div", { class: "card empty" },
        h("p", {}, "No pages yet."),
        h("button", { onclick: () => fileInput.click() }, "Select pages…"));
    }
    const busy = !!data.job;
    return h("div", { class: "grid" }, data.pages.map((p, i) => {
      const done = p.regions > 0 && p.translated === p.regions;
      const chip = !p.detected ? h("span", { class: "chip" }, "new")
        : p.rendered && done ? h("span", { class: "chip done" }, "done")
        : p.regions === 0 ? h("span", { class: "chip" }, "no text")
        : h("span", { class: "chip part" }, `${p.translated}/${p.regions}`);
      return h("div", { class: "card page-card" },
        h("a", { href: `#/c/${enc(name)}/edit/${enc(p.name)}`, title: "Review and edit this page" },
          h("img", { loading: "lazy", alt: `Page ${i + 1}`, src: p.rendered ? img(name, "thumb-translated", p.name, p.rendered) : img(name, "thumb", p.name, p.v) }),
          h("div", { class: "label" }, h("span", {}, `${i + 1}`), chip)),
        h("button", { class: "small del", disabled: busy, title: "Delete page", "aria-label": `Delete page ${i + 1}`, onclick: async () => {
          if (!confirm(`Delete page ${i + 1} (${p.name}) and its translation?`)) return;
          try { await api(`/api/chapters/${enc(name)}/pages/${enc(p.name)}`, { method: "DELETE" }); await refresh(); }
          catch (e) { toast(e.message, true); }
        } }, "✕"));
    }));
  }

  function render() {
    const busy = !!data.job;
    const pending = data.pages.filter((p) => !p.detected || p.translated < p.regions || !p.rendered).length;
    const anyRendered = data.pages.some((p) => p.rendered);
    mount(view, 
      fileInput,
      h("div", { class: "page-head" },
        h("div", {},
          h("a", { class: "crumb", href: "#/" }, "Library /"),
          h("h1", {}, name),
          h("div", { class: "muted", style: { fontSize: ".85rem" } },
            `${data.pages.length} pages` + (data.pages.length ? ` · ${data.pages.filter((p) => p.rendered).length} lettered` : ""))),
        h("div", { class: "actions" },
          h("button", { class: "secondary", onclick: () => fileInput.click(), disabled: busy }, "Add pages…"),
          h("button", { onclick: () => startJob("translate"), disabled: busy || !data.pages.length,
            title: "Find text, translate and letter every page that isn't done yet" },
            pending ? `Translate ${pending} page${pending === 1 ? "" : "s"}` : "Translate"),
          h("button", { class: "secondary", onclick: () => startJob("render"), disabled: busy || !data.pages.some((p) => p.detected),
            title: "Letter the pages again with the current options (no API calls)" }, "Re-letter"),
          h("a", { class: "button secondary", href: `#/c/${enc(name)}/read`, "aria-disabled": !data.pages.length }, "Read ▸"),
          h("button", { class: "ghost danger", disabled: busy, title: "Delete chapter", onclick: async () => {
            if (!confirm(`Delete “${name}” and all its pages and translations? This cannot be undone.`)) return;
            try { await api(`/api/chapters/${enc(name)}`, { method: "DELETE" }); location.hash = "#/"; }
            catch (e) { toast(e.message, true); }
          } }, "Delete"))),
      jobPanel(),
      stagingPanel(),
      settingsPanel(),
      !anyRendered && data.pages.length && !busy ? h("p", { class: "hint" }, "Tip: click a page to review its translation before or after lettering.") : null,
      pageGrid());
  }

  // Files can also be dropped anywhere on the chapter page.
  const onDragOver = (e) => { if (e.dataTransfer?.types?.includes("Files")) e.preventDefault(); };
  const onDrop = (e) => { if (e.dataTransfer?.files?.length) { e.preventDefault(); addFiles([...e.dataTransfer.files]); } };
  window.addEventListener("dragover", onDragOver);
  window.addEventListener("drop", onDrop);
  cleanup = () => { clearTimeout(pollTimer); clearStaged(); window.removeEventListener("dragover", onDragOver); window.removeEventListener("drop", onDrop); };

  render();
  if (data.job) poll();
}

// ------------------------------------------------------------------- editor
async function editorView(view, name, page) {
  const [chapter, detail] = await Promise.all([api(`/api/chapters/${enc(name)}`), api(`/api/chapters/${enc(name)}/pages/${enc(page)}`)]);
  const idx = chapter.pages.findIndex((p) => p.name === page);
  const meta = chapter.pages[idx];
  if (!meta) throw new Error("Page not found");
  let showing = meta.rendered ? "translated" : "original";
  let showBoxes = !meta.rendered;
  const dirty = new Map();          // region id -> {translation, short_translation, category}
  let active = null;
  let pollTimer = null;

  const prev = chapter.pages[idx - 1], next = chapter.pages[idx + 1];
  const go = (p) => { if (!dirty.size || confirm("Discard unsaved changes?")) location.hash = `#/c/${enc(name)}/edit/${enc(p.name)}`; };

  const image = h("img", { alt: `Page ${idx + 1}` });
  const stage = h("div", { class: "stage" + (showBoxes ? "" : " hide-boxes") }, image);
  const setImage = () => { image.src = showing === "translated" && meta.rendered ? img(name, "translated", page, meta.rendered) : img(name, "original", page, meta.v); };

  const boxes = detail.regions.map((r, i) => {
    const [x1, y1, x2, y2] = r.box;
    const el = h("button", { class: `box ${r.category === "noise" ? "noise" : ""}`, type: "button", title: r.translation || "",
      "aria-label": `Region ${i + 1}`,
      style: { left: `${(100 * x1) / detail.width}%`, top: `${(100 * y1) / detail.height}%`,
               width: `${(100 * (x2 - x1)) / detail.width}%`, height: `${(100 * (y2 - y1)) / detail.height}%` },
      onclick: () => focus(r.id, true) }, h("span", {}, i + 1));
    stage.append(el);
    return el;
  });

  const rows = detail.regions.map((r, i) => {
    const edit = (field, value) => {
      const d = dirty.get(r.id) || {};
      d[field] = value;
      dirty.set(r.id, d);
      row.classList.add("dirty");
      saveBtns();
    };
    const row = h("div", { class: "region", onfocusin: () => focus(r.id, false) },
      h("header", {},
        h("span", { class: "n" }, `#${i + 1}`),
        h("select", { "aria-label": "Category", onchange: (e) => edit("category", e.target.value) },
          ["dialogue", "thought", "narration", "sign", "sfx", "noise"].map((c) => h("option", { value: c, selected: r.category === c }, c))),
        h("span", { class: "muted", style: { fontSize: ".75rem" } }, r.kind === "bubble" ? "balloon" : "over art")),
      r.original_text ? h("p", { class: "jp", lang: "ja" }, r.original_text) : null,
      h("textarea", { rows: 2, "aria-label": `Translation ${i + 1}`, placeholder: r.translation == null ? "Not translated yet" : "",
        oninput: (e) => edit("translation", e.target.value) }, r.translation || ""),
      h("details", { class: "short" }, h("summary", {}, "Shorter version (used if the full text doesn't fit)"),
        h("textarea", { rows: 2, oninput: (e) => edit("short_translation", e.target.value) }, r.short_translation || "")));
    return row;
  });

  function focus(id, scroll) {
    active = id;
    detail.regions.forEach((r, i) => {
      boxes[i].classList.toggle("active", r.id === id);
      rows[i].classList.toggle("active", r.id === id);
      if (r.id === id && scroll) { rows[i].scrollIntoView({ behavior: "smooth", block: "center" }); rows[i].querySelector("textarea")?.focus({ preventScroll: true }); }
    });
    if (!showBoxes) { showBoxes = true; stage.classList.remove("hide-boxes"); boxToggle.setAttribute("aria-pressed", "true"); }
  }

  const saveBtn = h("button", { disabled: true, onclick: () => save(true) }, "Save & re-letter page");
  const saveOnly = h("button", { class: "secondary", disabled: true, onclick: () => save(false) }, "Save");
  const status = h("span", { class: "muted" });
  function saveBtns() { saveBtn.disabled = saveOnly.disabled = false; status.textContent = `${dirty.size} unsaved`; }

  async function save(reletter) {
    saveBtn.disabled = saveOnly.disabled = true;
    try {
      for (const [id, body] of dirty) await api(`/api/chapters/${enc(name)}/pages/${enc(page)}/regions/${enc(id)}`, { method: "PATCH", body });
      dirty.clear();
      rows.forEach((r) => r.classList.remove("dirty"));
      status.textContent = "Saved";
      if (reletter) await reletterPage();
    } catch (e) { toast(e.message, true); saveBtns(); }
  }

  async function reletterPage() {
    status.textContent = "Lettering…";
    const job = await api(`/api/chapters/${enc(name)}/jobs`, { method: "POST", body: { action: "render", pages: [page] } });
    const wait = async () => {
      const j = await api(`/api/jobs/${job.id}`);
      if (j.status === "queued" || j.status === "running") { pollTimer = setTimeout(wait, 700); return; }
      if (j.status !== "done") { status.textContent = ""; return toast(j.message || "Lettering failed", true); }
      const fresh = await api(`/api/chapters/${enc(name)}`);
      Object.assign(meta, fresh.pages.find((p) => p.name === page));
      showing = "translated";
      viewToggle.querySelector("[data-v=translated]").disabled = false;
      viewToggle.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === showing)));
      setImage();
      status.textContent = "Lettered";
    };
    wait();
  }

  async function translatePage(retranslate) {
    const key = keyStore.get();
    const cfg = await config();
    if (!key && !cfg.server_key) { toast("Add your Anthropic API key in Settings first.", true); return openSettings(); }
    if (dirty.size && !confirm("Translating replaces your unsaved edits on this page. Continue?")) return;
    try {
      await api(`/api/chapters/${enc(name)}/jobs`, { method: "POST", body: { action: "translate", pages: [page], retranslate },
        headers: key ? { "X-Anthropic-Key": key } : {} });
      location.hash = `#/c/${enc(name)}`;
    } catch (e) { toast(e.message, true); }
  }

  const viewToggle = h("div", { class: "seg", role: "group", "aria-label": "Image" },
    ["original", "translated"].map((v) => h("button", { type: "button", "data-v": v, "aria-pressed": String(showing === v), disabled: v === "translated" && !meta.rendered,
      onclick: (e) => { showing = v; viewToggle.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget))); setImage(); } },
      v === "original" ? "Original" : "Translated")));
  const boxToggle = h("button", { class: "secondary small", "aria-pressed": String(showBoxes), onclick: () => {
    showBoxes = !showBoxes; stage.classList.toggle("hide-boxes", !showBoxes); boxToggle.setAttribute("aria-pressed", String(showBoxes)); } }, "Boxes");

  const onKey = (e) => {
    if (e.target.closest("textarea, input, select")) return;
    if (e.key === "o") viewToggle.querySelector(`[data-v=${showing === "original" ? "translated" : "original"}]`)?.click();
  };
  const onLeave = (e) => { if (dirty.size) { e.preventDefault(); e.returnValue = ""; } };
  window.addEventListener("keydown", onKey);
  window.addEventListener("beforeunload", onLeave);
  cleanup = () => { clearTimeout(pollTimer); window.removeEventListener("keydown", onKey); window.removeEventListener("beforeunload", onLeave); };

  setImage();
  mount(view, 
    h("div", { class: "page-head" },
      h("div", {}, h("a", { class: "crumb", href: `#/c/${enc(name)}` }, `${name} /`), h("h1", {}, `Page ${idx + 1}`)),
      h("div", { class: "actions" },
        h("button", { class: "secondary", disabled: !prev, onclick: () => go(prev) }, "◀ Prev"),
        h("button", { class: "secondary", disabled: !next, onclick: () => go(next) }, "Next ▶"),
        h("a", { class: "button secondary", href: `#/c/${enc(name)}/read?p=${idx}` }, "Read from here"))),
    h("div", { class: "editor" },
      h("div", { class: "stage-wrap" },
        h("div", { class: "row", style: { marginBottom: ".5rem" } }, viewToggle, boxToggle, h("span", { class: "muted", style: { fontSize: ".8rem" } }, "O toggles original")),
        stage),
      h("div", {},
        h("div", { class: "card panel", style: { position: "sticky", top: "64px", zIndex: 2 } },
          h("div", { class: "row" }, saveBtn, saveOnly, status, h("span", { class: "spacer" }),
            h("button", { class: "ghost small", onclick: () => translatePage(true), title: "Ask the model again for this page" },
              detail.regions.some((r) => r.translation != null) ? "Re-translate page" : "Translate page"))),
        detail.detected
          ? (detail.regions.length ? h("div", { class: "regions" }, rows) : h("div", { class: "card empty" }, "No text found on this page."))
          : h("div", { class: "card empty" }, h("p", {}, "This page hasn't been processed yet."), h("button", { onclick: () => translatePage(false) }, "Translate page")))));
}

// ------------------------------------------------------------------- reader
async function readerView(view, name, startAt) {
  const [chapter, all] = await Promise.all([api(`/api/chapters/${enc(name)}`), api("/api/chapters")]);
  const pages = chapter.pages;
  if (!pages.length) throw new Error("This chapter has no pages yet.");
  const pos = all.findIndex((c) => c.name === name);
  const nextChapter = all[pos + 1];
  const prefs = JSON.parse(storage.get("mt.reader") || "{}");
  const st = {
    i: Number.isFinite(startAt) ? startAt : parseInt(storage.get(`mt.pos.${name}`) || "0", 10) || 0,
    mode: prefs.mode || "paged",        // paged | scroll
    dir: prefs.dir || "rtl",            // rtl (manga) | ltr
    fit: prefs.fit || "height",         // height | width (paged mode)
    original: false,
    ui: true,
    end: false,
  };
  st.i = Math.max(0, Math.min(pages.length - 1, st.i));
  const savePrefs = () => storage.set("mt.reader", JSON.stringify({ mode: st.mode, dir: st.dir, fit: st.fit }));
  const src = (p) => (!st.original && p.rendered ? img(name, "translated", p.name, p.rendered) : img(name, "original", p.name, p.v));

  document.body.classList.add("reading");
  const root = h("div", { class: "reader" });
  const pageBox = h("div", { class: "pages" });
  const scrollBox = h("div", { class: "scroll" });
  const badge = h("div", { class: "badge", hidden: true });
  const count = h("span", { class: "count" });
  const slider = h("input", { type: "range", min: 1, max: pages.length, value: st.i + 1, "aria-label": "Page",
    oninput: (e) => show(parseInt(e.target.value, 10) - 1) });

  const toggle = (label, key, values, onchange, title) => {
    const b = h("button", { class: "small", title, onclick: () => {
      st[key] = values[(values.indexOf(st[key]) + 1) % values.length];
      savePrefs(); onchange(); sync(); } });
    b.dataset.key = key; b.dataset.label = JSON.stringify(label);
    return b;
  };
  const modeBtn = toggle({ paged: "Paged", scroll: "Scroll" }, "mode", ["paged", "scroll"], () => layout(), "Paged or vertical scroll (M)");
  const dirBtn = toggle({ rtl: "Right → left", ltr: "Left → right" }, "dir", ["rtl", "ltr"], () => {}, "Reading direction (D)");
  const fitBtn = toggle({ height: "Fit height", width: "Fit width" }, "fit", ["height", "width"], () => layout(), "Fit (F)");
  const origBtn = h("button", { class: "small", title: "Show the original Japanese page (O)", onclick: () => setOriginal(!st.original) }, "Original");

  const top = h("div", { class: "bar top" },
    h("a", { class: "button small", href: `#/c/${enc(name)}`, title: "Back (Esc)" }, "✕"),
    h("span", { class: "title" }, name),
    h("span", { class: "spacer" }),
    origBtn, modeBtn, dirBtn, fitBtn,
    h("button", { class: "small hide-sm", title: "Full screen", onclick: () => document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen?.() }, "⛶"));
  const bottom = h("div", { class: "bar bottom" },
    h("button", { class: "small", onclick: () => step(-1), "aria-label": "Previous page" }, "‹"),
    slider,
    h("button", { class: "small", onclick: () => step(1), "aria-label": "Next page" }, "›"),
    count,
    h("span", { class: "help" }, "← → or click the sides · O original · M mode"));

  const zoneL = h("div", { class: "zone left", onclick: () => step(st.dir === "rtl" ? 1 : -1) });
  const zoneR = h("div", { class: "zone right", onclick: () => step(st.dir === "rtl" ? -1 : 1) });
  pageBox.addEventListener("click", (e) => { if (!e.target.closest(".zone, a, button")) { st.ui = !st.ui; sync(); } });
  scrollBox.addEventListener("click", (e) => { if (!e.target.closest("a, button")) { st.ui = !st.ui; sync(); } });

  root.append(pageBox, scrollBox, badge, top, bottom);
  mount(view, root);

  function sync() {
    root.classList.toggle("ui-hidden", !st.ui);
    root.classList.toggle("fit-width", st.mode === "paged" && st.fit === "width");
    for (const b of [modeBtn, dirBtn, fitBtn]) b.textContent = JSON.parse(b.dataset.label)[st[b.dataset.key]];
    fitBtn.hidden = st.mode !== "paged";
    dirBtn.hidden = st.mode !== "paged";
    origBtn.setAttribute("aria-pressed", String(st.original));
    count.textContent = st.end ? "End" : `${st.i + 1} / ${pages.length}`;
    slider.value = st.i + 1;
    // In RTL the slider reads right-to-left too, like a manga's page edge.
    slider.style.direction = st.mode === "paged" && st.dir === "rtl" ? "rtl" : "ltr";
    const p = pages[st.i];
    badge.hidden = st.end || (!st.original && !!p.rendered);
    badge.textContent = st.original ? "Original" : "Not translated yet — showing original";
  }

  let scrollObserver = null;
  function layout() {
    pageBox.hidden = st.mode !== "paged";
    scrollBox.hidden = st.mode !== "scroll";
    if (scrollObserver) { scrollObserver.disconnect(); scrollObserver = null; }
    if (st.mode === "scroll") {
      const imgs = pages.map((p, i) => h("img", { src: src(p), loading: "lazy", alt: `Page ${i + 1}`, "data-i": i }));
      scrollBox.replaceChildren(...imgs, endCard());
      scrollObserver = new IntersectionObserver((entries) => {
        for (const en of entries) if (en.isIntersecting) { st.i = +en.target.dataset.i; st.end = false; remember(); sync(); }
      }, { root: scrollBox, rootMargin: "-45% 0px -45% 0px" });
      imgs.forEach((im) => scrollObserver.observe(im));
      requestAnimationFrame(() => imgs[st.i]?.scrollIntoView());
    } else {
      show(st.i);
    }
    sync();
  }

  function endCard() {
    return h("div", { class: "end-card" },
      h("p", {}, `End of ${name}.`),
      nextChapter ? h("p", {}, h("a", { href: `#/c/${enc(nextChapter.name)}/read?p=0` }, `Next: ${nextChapter.name} ›`)) : null,
      h("p", {}, h("a", { href: `#/c/${enc(name)}` }, "Back to chapter")));
  }

  function show(i) {
    st.end = false;
    st.i = Math.max(0, Math.min(pages.length - 1, i));
    remember();
    if (st.mode === "scroll") {
      scrollBox.querySelector(`img[data-i="${st.i}"]`)?.scrollIntoView();
      return sync();
    }
    const im = h("img", { src: src(pages[st.i]), alt: `Page ${st.i + 1}` });
    pageBox.replaceChildren(im, zoneL, zoneR);
    pageBox.scrollTop = 0;
    // Preload the next pages so turning is instant.
    for (const k of [1, 2]) if (pages[st.i + k]) new Image().src = src(pages[st.i + k]);
    sync();
  }

  function step(d) {
    st.ui = false;   // bars get out of the way once reading starts; tap the middle to bring them back
    if (st.mode === "scroll") {
      scrollBox.scrollBy({ top: d * scrollBox.clientHeight * 0.85, behavior: "smooth" });
      return;
    }
    if (st.end) { if (d < 0) show(pages.length - 1); return; }
    if (d > 0 && st.i === pages.length - 1) {
      st.end = true;
      pageBox.replaceChildren(endCard(), zoneL, zoneR);
      return sync();
    }
    show(st.i + d);
  }

  function setOriginal(on) {
    st.original = on;
    if (st.mode === "scroll") scrollBox.querySelectorAll("img").forEach((im) => { im.src = src(pages[+im.dataset.i]); });
    else if (!st.end) pageBox.querySelector("img").src = src(pages[st.i]);
    sync();
  }

  function remember() { storage.set(`mt.pos.${name}`, String(st.i)); }

  const onKey = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    const rtl = st.mode === "paged" && st.dir === "rtl";
    if (k === "ArrowLeft" || k === "a") { step(rtl ? 1 : -1); e.preventDefault(); }
    else if (k === "ArrowRight" || k === "d") { step(rtl ? -1 : 1); e.preventDefault(); }
    else if ((k === "ArrowDown" || k === "ArrowUp") && st.mode === "paged" && st.fit === "height") { step(k === "ArrowDown" ? 1 : -1); e.preventDefault(); }
    else if (k === " " ) { step(e.shiftKey ? -1 : 1); e.preventDefault(); }
    else if (k === "o" || k === "O") setOriginal(!st.original);
    else if (k === "m") modeBtn.click();
    else if (k === "f") fitBtn.click();
    else if (k === "Escape" && !document.fullscreenElement) location.hash = `#/c/${enc(name)}`;
  };
  window.addEventListener("keydown", onKey);
  cleanup = () => {
    window.removeEventListener("keydown", onKey);
    if (scrollObserver) scrollObserver.disconnect();
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  };
  layout();
}

// ---------------------------------------------------------- settings dialog
function updateKeyDot() {
  const dot = $("#key-dot");
  const has = !!keyStore.get() || !!CONFIG?.server_key;
  dot.className = "dot " + (has ? "ok" : "none");
  dot.title = has ? "API key set" : "No API key yet";
}

async function openSettings() {
  const cfg = await config().catch(() => ({}));
  $("#key-input").value = keyStore.get();
  $("#key-remember").checked = keyStore.remembered() || !keyStore.get();
  $("#server-key-hint").hidden = !cfg.server_key;
  $("#key-result").textContent = "";
  $("#settings").showModal();
}

$("#open-settings").addEventListener("click", openSettings);
$("#settings").addEventListener("close", () => {
  keyStore.set($("#key-input").value.trim(), $("#key-remember").checked);
  updateKeyDot();
});
$("#key-clear").addEventListener("click", () => { $("#key-input").value = ""; keyStore.set(""); $("#key-result").textContent = "Forgotten."; updateKeyDot(); });
$("#key-check").addEventListener("click", async () => {
  const key = $("#key-input").value.trim();
  const out = $("#key-result");
  if (!key && !CONFIG?.server_key) { out.textContent = "Enter a key first."; return; }
  out.textContent = "Checking…";
  try {
    const r = await api("/api/key/check", { method: "POST", headers: key ? { "X-Anthropic-Key": key } : {} });
    out.textContent = r.ok ? "✓ Key works" : r.error;
    out.className = r.ok ? "" : "error";
  } catch (e) { out.textContent = e.message; out.className = "error"; }
});

// ------------------------------------------------------------------- login
function showLogin() {
  const dlg = $("#login");
  if (!dlg.open) { $("#login-error").hidden = true; dlg.showModal(); }
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/login", { method: "POST", body: { token: $("#login-token").value } });
    $("#login").close();
    CONFIG = null;
    route();
  } catch (err) { $("#login-error").textContent = "That token is not right."; $("#login-error").hidden = false; }
});

route();
