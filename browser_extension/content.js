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
 *    On mouseover of any .ara-word-hint, the popup fetches /define and displays
 *    the word's CEFR level + WordNet definition with a smooth fade-in animation.
 *
 * ANNOTATION INJECTION
 * ────────────────────
 * The API returns character positions relative to the flat text content of each
 * element.  annotateElement() uses a TreeWalker to locate the exact text nodes
 * those positions fall within, then rebuilds each affected text node as a
 * DocumentFragment containing the yellow spans.  This correctly handles
 * elements with mixed content (bold, links, etc.) without breaking the DOM.
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


// ── DOM helpers ───────────────────────────────────────────────────────────────

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


// ── Annotation injection ──────────────────────────────────────────────────────

/**
 * Wraps medium-difficulty words in yellow highlight spans.
 *
 * `annotations` is the array returned by the API — character positions
 * relative to the element's flat textContent (after trimming).
 * `trimOffset` is how many leading characters were trimmed before the
 * text was sent to the API, so we can convert back to DOM positions.
 */
function annotateElement(el, annotations, trimOffset) {
  if (!annotations || annotations.length === 0) return;

  // Build a map of text nodes to their character range in the element's
  // full textContent.  This lets us convert an annotation's absolute
  // position to (textNode, localStart, localEnd).
  const textNodes = [];
  let cumOffset   = 0;

  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const len = node.nodeValue.length;
    textNodes.push({ node, start: cumOffset, end: cumOffset + len, anns: [] });
    cumOffset += len;
  }

  // Assign each annotation to the text node it falls within, converting
  // from API-relative positions (in stripped text) to DOM positions.
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

  // For each text node that has annotations, rebuild it as a fragment
  // of plain text nodes and yellow span elements.
  for (const tn of textNodes) {
    if (tn.anns.length === 0) continue;

    const text    = tn.node.nodeValue;
    const parent  = tn.node.parentNode;
    const fragment = document.createDocumentFragment();
    let pos = 0;

    // Sort ascending so we walk left-to-right
    for (const ann of tn.anns.sort((a, b) => a.localStart - b.localStart)) {
      if (ann.localStart > pos) {
        fragment.appendChild(document.createTextNode(text.slice(pos, ann.localStart)));
      }
      const span = document.createElement("span");
      span.className      = "ara-word-hint";
      span.dataset.word   = ann.word;    // surface form for display
      span.dataset.lemma  = ann.lemma;   // base form for /define lookup
      span.dataset.level  = ann.level;
      span.dataset.pos    = ann.pos;
      span.textContent    = text.slice(ann.localStart, ann.localEnd);
      fragment.appendChild(span);
      pos = ann.localEnd;
    }
    if (pos < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(pos)));
    }
    parent.replaceChild(fragment, tn.node);
  }
}


// ── API helpers ───────────────────────────────────────────────────────────────

async function rewriteBatch(paragraphs, userLevel) {
  const res = await fetch(`${API}/rewrite`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ paragraphs, user_level: userLevel }),
  });
  if (!res.ok) throw new Error(`API returned ${res.status}`);
  return (await res.json()).results ?? [];
}


// ── Apply a single rewrite result to its DOM element ─────────────────────────

function applyResult(el, result) {
  const rawText     = el.textContent || "";
  const trimOffset  = rawText.indexOf(rawText.trim());  // leading whitespace count

  el.setAttribute(DONE_ATTR, "true");

  if (result.changed) {
    // Preserve the original HTML for undo / hover tooltip
    el.setAttribute(ORIG_ATTR, el.innerHTML);
    // Replace visible content with the simplified text.
    // Using textContent (not innerHTML) prevents XSS from model output.
    el.textContent = result.rewritten;
    el.classList.add("ara-modified");
    const preview = result.original.length > 160
      ? result.original.slice(0, 157) + "…"
      : result.original;
    el.title = `Original: ${preview}`;
  }

  // Annotate medium-difficulty words in whichever text is now displayed.
  // For changed paragraphs, annotations are relative to result.rewritten
  // (which is now el.textContent).  For unchanged ones, they're relative
  // to the original stripped text.  Either way, trimOffset corrects for
  // leading whitespace in the DOM text node.
  annotateElement(el, result.annotations, result.changed ? 0 : trimOffset);
}


// ── Progress reporting ────────────────────────────────────────────────────────

function reportProgress(done, total, extra = {}) {
  const pct = total === 0 ? 100 : Math.min(100, Math.round((done / total) * 100));
  chrome.storage.local.set({ araProgress: pct, araTotal: total, araDone: done, ...extra });
}


// ── Main simplification flow ──────────────────────────────────────────────────

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
    const texts = batch.map(el => el.textContent || "");

    try {
      const results = await rewriteBatch(
        texts.map(t => t.trim()),
        userLevel
      );
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


// ── Undo ──────────────────────────────────────────────────────────────────────

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


// ── Definition hover popup ────────────────────────────────────────────────────

// Create the singleton popup element once per page load.
// Guard prevents duplicate creation if the content script is somehow
// re-injected without a full page reload.
let defPopup = document.getElementById("ara-def-popup");
if (!defPopup) {
  defPopup = document.createElement("div");
  defPopup.id = "ara-def-popup";
  defPopup.innerHTML = `
    <div class="ara-pop-word"  id="ara-pw"></div>
    <div class="ara-pop-meta">
      <span class="ara-level-badge" id="ara-plb"></span>
      <span class="ara-pop-pos"     id="ara-pp"></span>
    </div>
    <div class="ara-pop-def"     id="ara-pd"></div>
    <div class="ara-pop-example" id="ara-pe"></div>
  `;
  document.body.appendChild(defPopup);
}

// Client-side cache so we never call /define twice for the same word+pos
const defCache  = new Map();
let   showTimer = null;
let   hideTimer = null;

async function fetchDefinition(lemma, pos) {
  const key = `${lemma}:${pos}`;
  if (defCache.has(key)) return defCache.get(key);
  try {
    const r    = await fetch(`${API}/define?word=${encodeURIComponent(lemma)}&pos=${encodeURIComponent(pos)}`);
    const data = await r.json();
    defCache.set(key, data);
    return data;
  } catch {
    return null;
  }
}

function positionPopup(targetRect) {
  const PW  = 280;   // popup width (matches CSS)
  const GAP = 10;    // gap between word and popup edge
  const vpW = window.innerWidth;
  const vpH = window.innerHeight;

  // Centre horizontally on the target word; clamp to viewport edges
  let left = targetRect.left + targetRect.width / 2 - PW / 2;
  left = Math.max(GAP, Math.min(left, vpW - PW - GAP));

  // Prefer below; fall back to above if not enough room
  const spaceBelow = vpH - targetRect.bottom - GAP;
  const top = spaceBelow >= 120
    ? targetRect.bottom + GAP
    : targetRect.top - 130 - GAP;   // 130 ≈ popup height estimate

  defPopup.style.left = `${left}px`;
  defPopup.style.top  = `${top}px`;
}

function showPopup(span, data) {
  if (!data) return;

  document.getElementById("ara-pw").textContent  = span.dataset.word || data.word;
  document.getElementById("ara-pd").textContent  = data.definition || "No definition available.";

  const badge = document.getElementById("ara-plb");
  badge.textContent = data.level && data.level !== "unknown" ? data.level : "?";
  badge.className   = `ara-level-badge ara-lvl-${data.level || "B1"}`;

  document.getElementById("ara-pp").textContent = data.pos || "";

  const exEl = document.getElementById("ara-pe");
  if (data.example) {
    exEl.textContent  = `"${data.example}"`;
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

// Event delegation — one listener for all .ara-word-hint spans (present and future)
document.addEventListener("mouseover", async (e) => {
  const span = e.target.closest(".ara-word-hint");
  if (!span) return;

  clearTimeout(hideTimer);
  clearTimeout(showTimer);

  // 280ms delay prevents the popup from flickering when the user
  // moves the mouse across words quickly without intending to read them
  showTimer = setTimeout(async () => {
    const lemma = span.dataset.lemma || span.dataset.word || span.textContent;
    const pos   = span.dataset.pos   || "NOUN";
    const data  = await fetchDefinition(lemma.toLowerCase(), pos);
    showPopup(span, data);
  }, 280);
});

document.addEventListener("mouseout", (e) => {
  if (!e.target.closest(".ara-word-hint")) return;
  clearTimeout(showTimer);
  hideTimer = setTimeout(hidePopup, 180);
});


// ── Message listener ──────────────────────────────────────────────────────────

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