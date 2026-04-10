/**
 * content.js — Adaptive Reading Assistant
 *
 * Mirrors how Google Translate replaces page text in-place:
 *  1. Walk the DOM and collect all readable text nodes.
 *  2. Split each node's text into sentences via the NLP API.
 *  3. Simplify sentences flagged as complex.
 *  4. Replace the original text node with the simplified version,
 *     wrapping changed sentences in highlighted spans.
 *     Hovering a span reveals the original text as a tooltip.
 */

const API = "http://localhost:5000";
const DONE_ATTR = "data-ara-done";   // marks already-processed parent elements
const SPAN_CLASS = "ara-simplified";

// ── DOM traversal ────────────────────────────────────────────────────────────

function collectTextNodes(root = document.body) {
  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA",
    "INPUT", "CODE", "PRE", "SELECT",
  ]);

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const p = node.parentElement;
      if (!p) return NodeFilter.FILTER_REJECT;
      if (SKIP_TAGS.has(p.tagName?.toUpperCase())) return NodeFilter.FILTER_REJECT;
      if (p.hasAttribute(DONE_ATTR)) return NodeFilter.FILTER_REJECT;
      if (!node.nodeValue?.trim()) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const nodes = [];
  let n;
  while ((n = walker.nextNode())) nodes.push(n);
  return nodes;
}

// ── API helpers ──────────────────────────────────────────────────────────────

async function splitText(text) {
  const r = await fetch(`${API}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return (await r.json()).sentences ?? [text];
}

async function simplifyBatch(sentences, userLevel) {
  const r = await fetch(`${API}/simplify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentences, user_level: userLevel }),
  });
  return (await r.json()).results ?? [];
}

// ── In-place DOM replacement ─────────────────────────────────────────────────

function applyResults(textNode, results) {
  const parent = textNode.parentElement;
  if (!parent || parent.hasAttribute(DONE_ATTR)) return;

  const fragment = document.createDocumentFragment();

  for (const result of results) {
    if (!result.changed) {
      // Unchanged sentence: keep as a plain text node
      fragment.appendChild(document.createTextNode(result.original + " "));
    } else {
      // Changed sentence: highlighted span with original text on hover
      const span = document.createElement("span");
      span.className = SPAN_CLASS;
      span.textContent = result.simplified;
      span.title = `Original: ${result.original}`;
      span.setAttribute("data-original", result.original);
      fragment.appendChild(span);
      fragment.appendChild(document.createTextNode(" "));
    }
  }

  // Mark the parent so we don't process it again on re-trigger
  parent.setAttribute(DONE_ATTR, "true");
  textNode.replaceWith(fragment);
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function runSimplification(userLevel) {
  const nodes = collectTextNodes();
  const BATCH = 4; // Process 4 nodes concurrently to keep the API responsive

  for (let i = 0; i < nodes.length; i += BATCH) {
    await Promise.all(
      nodes.slice(i, i + BATCH).map(async (node) => {
        const text = node.nodeValue?.trim() ?? "";
        if (text.length < 25) return; // Skip captions, labels, etc.

        try {
          const sentences = await splitText(text);
          const results   = await simplifyBatch(sentences, userLevel);
          applyResults(node, results);
        } catch {
          // API unreachable — fail silently so the page stays usable
        }
      })
    );
  }
}

// Listen for the "simplify" command from popup.js
chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.action === "simplify") {
    runSimplification(msg.userLevel)
      .then(() => reply({ status: "done" }))
      .catch((e) => reply({ status: "error", error: e.message }));
    return true; // Keep the port open for the async reply
  }
});