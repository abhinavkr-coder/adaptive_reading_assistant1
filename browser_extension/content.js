/**
 * content.js — Adaptive Reading Assistant
 *
 * PIPELINE
 * ────────
 * 1. Collect block-level elements (p, h1-h6, li, etc.) from the page.
 * 2. Send batches of 3 to POST /rewrite.
 * 3. For each result:
 *    a. If changed=true, replace the element's text with the simplified version.
 *    b. Regardless of changed, inject yellow <span class="ara-word-hint"> elements
 *       for any medium-difficulty words listed in `annotations`.
 * 4. A global singleton popup (#ara-def-popup) handles hover definitions.
 *    On mouseover of .ara-word-hint, the popup fetches /define and displays
 *    the word's CEFR level + WordNet definition with a smooth fade-in animation.
 *
 * ANNOTATION INJECTION
 * ────────────────────
 * The API returns character positions relative to the flat text content of each
 * element. annotateElement() uses a TreeWalker to locate the exact text nodes,
 * then rebuilds each affected text node as a DocumentFragment containing spans.
 * This handles elements with mixed content (bold, links, etc.) without breaking DOM.
 */

const API       = "http://localhost:5000";
const DONE_ATTR = "data-ara-done";
const ORIG_ATTR = "data-ara-original";

const BLOCK_SELECTOR = [
  "p", "h1", "h2", "h3", "h4", "h5", "h6",
  "li", "td", "th", "blockquote", "figcaption",
  "summary", "dt", "dd",
].join(", ");

const SKIP_ANCESTORS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "CODE", "PRE",
  "TEXTAREA", "INPUT", "SELECT", "IFRAME",
]);


// ── DOM helpers ───────────────────────────────────────────────

function isSkippable(el) {
  if (el.hasAttribute(DONE_ATTR)) return true;
  let node = el;
  while (node) {
    if (SKIP_ANCESTORS.has(node.tagName?.toUpperCase())) return true;
    node = node.parentElement;
  }
  const text    = (el.textContent || "").trim();
  const linkLen = Array.from(el.querySelectorAll("a"))
    .reduce((n, a) => n + (a.textContent || "").length, 0);
  return text.length < 40 || (text.length > 0 && linkLen / text.length > 0.6);
}

function collectBlocks() {
  return Array.from(document.querySelectorAll(BLOCK_SELECTOR))
    .filter(el => !isSkippable(el));
}


// ── Annotation injection ──────────────────────────────────────

function annotateElement(el, annotations, trimOffset) {
  if (!annotations || annotations.length === 0) return;

  const textNodes = [];
  let cumOffset   = 0;

  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const len = node.nodeValue.length;
    textNodes.push({ node, start: cumOffset, end: cumOffset + len, anns: [] });
    cumOffset += len;
  }

  for (const ann of annotations) {
    const absStart = ann.start + trimOffset;
    const absEnd   = ann.end   + trimOffset;
    const tn = textNodes.find(t => t.start <= absStart && absEnd <= t.end);
    if (!tn) continue;
    tn.anns.push({
      ...ann,
      localStart: absStart - tn.start,
      localEnd:   absEnd   - tn.start,
    });
  }

  for (const tn of textNodes) {
    if (tn.anns.length === 0) continue;

    const text     = tn.node.nodeValue;
    const parent   = tn.node.parentNode;
    const fragment = document.createDocumentFragment();
    let pos = 0;

    for (const ann of tn.anns.sort((a, b) => a.localStart - b.localStart)) {
      if (ann.localStart > pos) {
        fragment.appendChild(document.createTextNode(text.slice(pos, ann.localStart)));
      }
      const span = document.createElement("span");
      span.className     = "ara-word-hint";
      span.dataset.word  = ann.word;
      span.dataset.lemma = ann.lemma;
      span.dataset.level = ann.level;
      span.dataset.pos   = ann.pos;
      span.textContent   = text.slice(ann.localStart, ann.localEnd);
      fragment.appendChild(span);
      pos = ann.localEnd;
    }
    if (pos < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(pos)));
    }
    parent.replaceChild(fragment, tn.node);
  }
}


// ── API helpers ───────────────────────────────────────────────

async function rewriteBatch(paragraphs, userLevel) {
  const res = await fetch(`${API}/rewrite`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ paragraphs, user_level: userLevel }),
  });
  if (!res.ok) throw new Error(`API returned ${res.status}`);
  return (await res.json()).results ?? [];
}


// ── Apply result to DOM element ───────────────────────────────

function applyResult(el, result) {
  const rawText    = el.textContent || "";
  const trimOffset = rawText.indexOf(rawText.trim());

  el.setAttribute(DONE_ATTR, "true");

  if (result.changed) {
    el.setAttribute(ORIG_ATTR, el.innerHTML);
    el.textContent = result.rewritten;
    el.classList.add("ara-modified");
    const preview = result.original.length > 160
      ? result.original.slice(0, 157) + "…"
      : result.original;
    el.title = `Original: ${preview}`;
  }

  annotateElement(el, result.annotations, result.changed ? 0 : trimOffset);
}


// ── Progress reporting ────────────────────────────────────────

function reportProgress(done, total, extra = {}) {
  const pct = total === 0 ? 100 : Math.min(100, Math.round((done / total) * 100));
  chrome.storage.local.set({ araProgress: pct, araTotal: total, araDone: done, ...extra });
}


// ── Main simplification loop ──────────────────────────────────

async function runSimplification(userLevel) {
  const blocks = collectBlocks();
  const total  = blocks.length;

  if (total === 0) {
    chrome.storage.local.set({
      araRunning: false, araProgress: 100,
      araStatus: "No paragraphs found on this page.",
    });
    return;
  }

  reportProgress(0, total, {
    araRunning: true,
    araStatus:  `Simplifying… 0 of ${total} paragraphs`,
    araLevel:   userLevel,
  });

  const BATCH = 3;
  let processed = 0;

  for (let i = 0; i < blocks.length; i += BATCH) {
    const batch = blocks.slice(i, i + BATCH);
    const texts = batch.map(el => (el.textContent || "").trim());

    try {
      const results = await rewriteBatch(texts, userLevel);
      results.forEach((result, j) => applyResult(batch[j], result));
    } catch (err) {
      console.warn("[ARA] Batch error:", err.message);
      batch.forEach(el => el.setAttribute(DONE_ATTR, "true"));
    }

    processed += batch.length;
    reportProgress(processed, total, {
      araRunning: processed < total,
      araStatus:  processed < total
        ? `Simplifying… ${processed} of ${total} paragraphs`
        : `Done — hover yellow words for definitions.`,
    });
  }

  chrome.storage.local.set({
    araRunning:  false,
    araProgress: 100,
    araStatus:   `Done — hover yellow words for definitions.`,
    araHasDone:  true,
  });
}


// ── Undo ──────────────────────────────────────────────────────

function undoSimplification() {
  let restored = 0;
  document.querySelectorAll(`[${DONE_ATTR}]`).forEach(el => {
    const original = el.getAttribute(ORIG_ATTR);
    if (original !== null) {
      el.innerHTML = original;
      el.removeAttribute(ORIG_ATTR);
      el.classList.remove("ara-modified");
      el.removeAttribute("title");
      restored++;
    }
    el.removeAttribute(DONE_ATTR);
  });
  chrome.storage.local.set({
    araRunning: false, araProgress: 0,
    araStatus: `Restored ${restored} paragraphs.`,
    araHasDone: false, araDone: 0, araTotal: 0,
  });
}


// ── Definition hover popup ────────────────────────────────────

// Build the singleton popup once.  Guard against re-injection.
let defPopup = document.getElementById("ara-def-popup");
if (!defPopup) {
  defPopup = document.createElement("div");
  defPopup.id = "ara-def-popup";
  // ara-pop-inner wraps all content so the ::before gradient bar
  // can touch the popup's edges (if we put padding on #ara-def-popup
  // itself, the bar would be indented).
  defPopup.innerHTML = `
    <div class="ara-pop-inner">
      <span class="ara-pop-word"  id="ara-pw"></span>
      <div class="ara-pop-meta">
        <span class="ara-level-badge" id="ara-plb"></span>
        <span class="ara-pop-pos"     id="ara-pp"></span>
      </div>
      <span class="ara-pop-def"     id="ara-pd"></span>
      <span class="ara-pop-example" id="ara-pe"></span>
    </div>
  `;
  document.body.appendChild(defPopup);
}

// Client-side definition cache — one /define call per (lemma, pos) pair
const defCache  = new Map();
let   showTimer = null;
let   hideTimer = null;

async function fetchDefinition(lemma, pos) {
  const key = `${lemma}:${pos}`;
  if (defCache.has(key)) return defCache.get(key);
  try {
    const r    = await fetch(
      `${API}/define?word=${encodeURIComponent(lemma)}&pos=${encodeURIComponent(pos)}`
    );
    const data = await r.json();
    defCache.set(key, data);
    return data;
  } catch {
    return null;
  }
}

/**
 * Position the popup relative to the hovered word's bounding rect.
 *
 * FIX: The popup uses `position: fixed`, which means its coordinates are
 * always relative to the viewport — exactly what getBoundingClientRect()
 * returns. The old code was adding window.scrollX/scrollY, which converted
 * those viewport-relative coords into document coords. That caused the popup
 * to be displaced downward by the current scroll offset, sending it off-screen
 * for anything below the first screenful of the page.
 *
 * Rule of thumb:
 *   position: fixed  → use getBoundingClientRect() values directly (no scroll offset)
 *   position: absolute → add window.scrollX / scrollY to getBoundingClientRect() values
 */
function positionPopup(rect) {
  const POPUP_WIDTH = 285;
  const GAP         = 10;
  const vpW         = window.innerWidth;
  const vpH         = window.innerHeight;

  // Centre the popup horizontally under the word, clamped to the viewport.
  let left = rect.left + rect.width / 2 - POPUP_WIDTH / 2;
  left = Math.max(GAP, Math.min(left, vpW - POPUP_WIDTH - GAP));

  // Prefer showing below the word; fall back to above if there isn't room.
  // Use Math.max to ensure we never go above the top of the viewport.
  const spaceBelow = vpH - rect.bottom - GAP;
  const top = spaceBelow >= 130
    ? rect.bottom + GAP
    : Math.max(GAP, rect.top - 150 - GAP);

  // No scroll offset needed — `fixed` positioning is already viewport-relative.
  defPopup.style.left = `${left}px`;
  defPopup.style.top  = `${top}px`;
}

function showPopup(span, data) {
  if (!data) return;

  document.getElementById("ara-pw").textContent = span.dataset.word || data.word;
  document.getElementById("ara-pd").textContent = data.definition   || "No definition available.";

  const badge = document.getElementById("ara-plb");
  const lvl   = (data.level && data.level !== "unknown") ? data.level : "?";
  badge.textContent = lvl;
  badge.className   = `ara-level-badge ara-lvl-${data.level || "B1"}`;

  document.getElementById("ara-pp").textContent = data.pos || "";

  const exEl = document.getElementById("ara-pe");
  if (data.example) {
    exEl.textContent   = `"${data.example}"`;
    exEl.style.display = "block";
  } else {
    exEl.style.display = "none";
  }

  positionPopup(span.getBoundingClientRect());
  defPopup.classList.add("ara-pop-visible");
}

function hidePopup() {
  defPopup.classList.remove("ara-pop-visible");
}

// Delegated listeners — one pair handles all .ara-word-hint spans
// including ones created after the page loaded.
document.addEventListener("mouseover", async e => {
  const span = e.target.closest(".ara-word-hint");
  if (!span) return;
  clearTimeout(hideTimer);
  clearTimeout(showTimer);
  // 260ms delay prevents popup flicker on fast mouse movement
  showTimer = setTimeout(async () => {
    const lemma = (span.dataset.lemma || span.dataset.word || span.textContent).toLowerCase();
    const pos   = span.dataset.pos || "NOUN";
    const data  = await fetchDefinition(lemma, pos);
    showPopup(span, data);
  }, 260);
});

document.addEventListener("mouseout", e => {
  if (!e.target.closest(".ara-word-hint")) return;
  clearTimeout(showTimer);
  hideTimer = setTimeout(hidePopup, 180);
});


// ── Message listener ──────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.action === "simplify") {
    runSimplification(msg.userLevel)
      .then(()  => reply({ status: "done" }))
      .catch(e  => reply({ status: "error", error: e.message }));
    return true;
  }
  if (msg.action === "undo") {
    undoSimplification();
    reply({ status: "done" });
    return true;
  }
});