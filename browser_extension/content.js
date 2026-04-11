/**
 * content.js — Adaptive Reading Assistant
 *
 * ARCHITECTURE: PARAGRAPH-LEVEL REWRITING
 * ─────────────────────────────────────────
 * Previous approach: walk all text nodes, split into sentences, simplify one-by-one.
 * Problem: the model loses all context. "She met him there" has no referents.
 *
 * New approach:
 *  1. Collect block-level elements (p, h1-h6, li, td, blockquote, etc.)
 *  2. Extract the full text of each block as the unit of simplification.
 *  3. Send batches of 3 paragraphs to POST /rewrite.
 *  4. Replace block's innerHTML with the rewritten text, preserving a data-
 *     attribute copy of the original for hover/undo.
 *  5. Report progress (0–100%) to chrome.storage.local so the popup
 *     shows a live percentage even if it was closed and reopened.
 *
 * UNDO: Stores original innerHTML. The "Restore" button in the popup
 *       sends an "undo" message that this script handles.
 */

const API          = "http://localhost:5000";
const DONE_ATTR    = "data-ara-done";
const ORIGINAL_KEY = "data-ara-original";

// Block elements whose full text content constitutes a meaningful paragraph.
// We avoid inline tags (span, a, em) because they rarely stand alone.
const BLOCK_SELECTOR = [
  "p", "h1", "h2", "h3", "h4", "h5", "h6",
  "li", "td", "th", "blockquote", "figcaption",
  "summary", "dt", "dd", "article", "section > div",
].join(", ");

// Elements that should never be rewritten
const SKIP_ANCESTORS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "CODE", "PRE",
  "TEXTAREA", "INPUT", "SELECT", "IFRAME",
]);

// ── DOM helpers ───────────────────────────────────────────────────────────────

function isSkippable(el) {
  // Already processed in this run
  if (el.hasAttribute(DONE_ATTR)) return true;

  // Inside a no-rewrite ancestor
  let node = el;
  while (node) {
    if (SKIP_ANCESTORS.has(node.tagName?.toUpperCase())) return true;
    node = node.parentElement;
  }

  // Too short to be a real paragraph (labels, bylines, nav items)
  const text = (el.innerText || el.textContent || "").trim();
  if (text.length < 40) return true;

  // Mostly a navigation element (>60% of text inside <a> tags)
  const linkText = Array.from(el.querySelectorAll("a"))
    .map(a => a.innerText || "").join("").length;
  if (text.length > 0 && linkText / text.length > 0.6) return true;

  return false;
}

function collectBlocks() {
  return Array.from(document.querySelectorAll(BLOCK_SELECTOR))
    .filter(el => !isSkippable(el));
}

// ── API helpers ───────────────────────────────────────────────────────────────

async function rewriteBatch(paragraphs, userLevel) {
  const res = await fetch(`${API}/rewrite`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ paragraphs, user_level: userLevel }),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return (await res.json()).results ?? [];
}

// ── DOM replacement ───────────────────────────────────────────────────────────

function applyRewrite(el, result) {
  if (!result.changed) {
    // Mark processed but don't touch content
    el.setAttribute(DONE_ATTR, "true");
    return;
  }

  // Preserve original HTML for undo and hover tooltip
  el.setAttribute(ORIGINAL_KEY, el.innerHTML);

  // Replace visible content with simplified text.
  // We set textContent (not innerHTML) to avoid XSS from model output,
  // but preserve block-level structure via the parent element staying intact.
  el.textContent = result.rewritten;
  el.setAttribute(DONE_ATTR, "true");
  el.classList.add("ara-modified");

  // Truncated original for the browser tooltip
  const preview = result.original.length > 160
    ? result.original.slice(0, 157) + "…"
    : result.original;
  el.title = `Original: ${preview}`;
}

// ── Progress reporting ────────────────────────────────────────────────────────

function reportProgress(done, total, extra = {}) {
  const pct = total === 0 ? 100 : Math.min(100, Math.round((done / total) * 100));
  chrome.storage.local.set({
    araProgress: pct,
    araTotal:    total,
    araDone:     done,
    ...extra,
  });
}

// ── Main simplification flow ──────────────────────────────────────────────────

async function runSimplification(userLevel) {
  const blocks = collectBlocks();
  const total  = blocks.length;

  if (total === 0) {
    chrome.storage.local.set({
      araRunning:  false,
      araProgress: 100,
      araStatus:   "No paragraphs found to simplify on this page.",
    });
    return;
  }

  reportProgress(0, total, {
    araRunning: true,
    araStatus:  `Simplifying… 0 of ${total} paragraphs`,
    araLevel:   userLevel,
  });

  const BATCH_SIZE = 3;  // 3 paragraphs per API call balances throughput vs latency
  let processed = 0;

  for (let i = 0; i < blocks.length; i += BATCH_SIZE) {
    const batch   = blocks.slice(i, i + BATCH_SIZE);
    const texts   = batch.map(el => (el.innerText || el.textContent || "").trim());

    try {
      const results = await rewriteBatch(texts, userLevel);
      results.forEach((result, j) => applyRewrite(batch[j], result));
    } catch (err) {
      // Fail gracefully: mark as done (no retry), keep original text visible
      console.warn("[ARA] Batch error:", err.message);
      batch.forEach(el => el.setAttribute(DONE_ATTR, "true"));
    }

    processed += batch.length;
    reportProgress(processed, total, {
      araRunning: processed < total,
      araStatus:  processed < total
        ? `Simplifying… ${processed} of ${total} paragraphs`
        : `Done — ${total} paragraphs rewritten. Hover to see originals.`,
    });
  }

  // Final state: simplification complete
  chrome.storage.local.set({
    araRunning:  false,
    araProgress: 100,
    araStatus:   `Done — ${total} paragraphs rewritten. Hover to see originals.`,
    araHasDone:  true,
  });
}

// ── Undo (restore all original HTML) ─────────────────────────────────────────

function undoSimplification() {
  let restored = 0;
  document.querySelectorAll(`[${DONE_ATTR}]`).forEach(el => {
    const original = el.getAttribute(ORIGINAL_KEY);
    if (original !== null) {
      el.innerHTML = original;
      el.removeAttribute(ORIGINAL_KEY);
      el.classList.remove("ara-modified");
      el.removeAttribute("title");
      restored++;
    }
    el.removeAttribute(DONE_ATTR);
  });

  chrome.storage.local.set({
    araRunning:  false,
    araProgress: 0,
    araStatus:   `Restored — ${restored} paragraphs reverted to original.`,
    araHasDone:  false,
    araDone:     0,
    araTotal:    0,
  });
}

// ── Message listener ──────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.action === "simplify") {
    runSimplification(msg.userLevel)
      .then(()  => reply({ status: "done" }))
      .catch(e  => reply({ status: "error", error: e.message }));
    return true; // keep port open for async reply
  }

  if (msg.action === "undo") {
    undoSimplification();
    reply({ status: "done" });
    return true;
  }
});