/**
 * content.js — Adaptive Reading Assistant (word-level replacement)
 *
 * Flow:
 *  1. Collect all readable text nodes on the page.
 *  2. For each node, split its text into sentences via /split.
 *  3. Send sentences to /simplify — the API returns per-sentence
 *     replacement arrays with {original, replacement, start, end}
 *     where start/end are character offsets inside that sentence.
 *  4. Convert those sentence-relative offsets to offsets inside the
 *     full text-node string.
 *  5. Rebuild the text node as a document fragment: plain text for
 *     unchanged characters, <span class="ara-word"> for swapped words.
 *     Hovering a span shows the original word as a tooltip.
 */

const API       = "http://localhost:5000";
const DONE_ATTR = "data-ara-done";
const WORD_CLS  = "ara-word";

// ── DOM helpers ──────────────────────────────────────────────────────────────

function collectTextNodes(root = document.body) {
    const SKIP = new Set([
        "SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA",
        "INPUT", "CODE", "PRE", "SELECT",
    ]);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            const p = node.parentElement;
            if (!p) return NodeFilter.FILTER_REJECT;
            if (SKIP.has(p.tagName?.toUpperCase())) return NodeFilter.FILTER_REJECT;
            if (p.closest(`[${DONE_ATTR}]`)) return NodeFilter.FILTER_REJECT;
            if (!node.nodeValue?.trim()) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        },
    });
    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);
    return nodes;
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function splitText(text) {
    const r = await fetch(`${API}/split`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    });
    return (await r.json()).sentences ?? [text];
}

async function simplifySentences(sentences, userLevel) {
    const r = await fetch(`${API}/simplify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences, user_level: userLevel }),
    });
    return (await r.json()).results ?? [];
}

// ── Fragment builder ─────────────────────────────────────────────────────────

/**
 * Given the raw text-node string and the array of sentence results,
 * builds a DocumentFragment that is identical to the original text
 * except that each replaced word is wrapped in a highlighted span.
 *
 * Strategy:
 *   - Find each sentence inside `fullText` sequentially (left-to-right)
 *     using indexOf with a running cursor so we handle repeated sentences.
 *   - Convert sentence-relative {start, end} to absolute positions in fullText.
 *   - Walk fullText character-by-character, emitting plain TextNodes for
 *     unchanged runs and <span> elements for replaced words.
 */
function buildFragment(fullText, results) {
    // Collect all replacements with absolute positions in fullText
    const absReplacements = [];
    let cursor = 0;

    for (const result of results) {
        if (!result.changed || !result.replacements?.length) {
            // Advance cursor past this sentence even if unchanged
            const idx = fullText.indexOf(result.original, cursor);
            if (idx !== -1) cursor = idx + result.original.length;
            continue;
        }

        const sentenceStart = fullText.indexOf(result.original, cursor);
        if (sentenceStart === -1) continue;  // safety — sentence not found

        for (const rep of result.replacements) {
            absReplacements.push({
                start:       sentenceStart + rep.start,
                end:         sentenceStart + rep.end,
                original:    rep.original,
                replacement: rep.replacement,
            });
        }
        cursor = sentenceStart + result.original.length;
    }

    // Sort by start position so we can walk left-to-right
    absReplacements.sort((a, b) => a.start - b.start);

    // Build the fragment
    const fragment = document.createDocumentFragment();
    let pos = 0;

    for (const rep of absReplacements) {
        if (rep.start < pos) continue;  // overlapping — skip (shouldn't happen)

        // Plain text before this replacement
        if (rep.start > pos) {
            fragment.appendChild(
                document.createTextNode(fullText.slice(pos, rep.start))
            );
        }

        // The replaced word wrapped in a span
        const span = document.createElement("span");
        span.className   = WORD_CLS;
        span.textContent = rep.replacement;
        span.title       = `Original: "${rep.original}"`;
        span.setAttribute("data-original", rep.original);
        fragment.appendChild(span);

        pos = rep.end;
    }

    // Any remaining plain text after the last replacement
    if (pos < fullText.length) {
        fragment.appendChild(document.createTextNode(fullText.slice(pos)));
    }

    return fragment;
}

// ── Main processing ───────────────────────────────────────────────────────────

async function processNode(node, userLevel) {
    const rawText = node.nodeValue ?? "";
    if (rawText.trim().length < 20) return;  // skip captions, labels etc.

    const parent = node.parentElement;
    if (!parent || parent.hasAttribute(DONE_ATTR)) return;

    const sentences = await splitText(rawText);
    const results   = await simplifySentences(sentences, userLevel);

    const anyChanged = results.some(r => r.changed);
    if (!anyChanged) return;  // nothing to do — leave DOM untouched

    const fragment = buildFragment(rawText, results);
    parent.setAttribute(DONE_ATTR, "true");
    node.replaceWith(fragment);
}

async function runSimplification(userLevel) {
    const nodes = collectTextNodes();
    const CONCURRENCY = 3;  // process 3 text nodes in parallel

    for (let i = 0; i < nodes.length; i += CONCURRENCY) {
        await Promise.all(
            nodes.slice(i, i + CONCURRENCY).map(n =>
                processNode(n, userLevel).catch(() => {})  // never crash the page
            )
        );
    }
}

// Listen for the popup's "Simplify" button
chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg.action === "simplify") {
        runSimplification(msg.userLevel)
            .then(() => reply({ status: "done" }))
            .catch(e => reply({ status: "error", error: e.message }));
        return true;  // keep message channel open for async reply
    }
});