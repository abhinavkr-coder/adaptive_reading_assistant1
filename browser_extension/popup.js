/**
 * popup.js — Adaptive Reading Assistant
 *
 * STATE PERSISTENCE STRATEGY
 * ──────────────────────────
 * Chrome extension popups are destroyed every time the user clicks away.
 * There is no way to prevent this. The standard solution is to keep all
 * state in chrome.storage.local (which persists across popup opens) and
 * restore it on DOMContentLoaded.
 *
 * Live progress during an active run: content.js writes araProgress to
 * storage on every batch. While the popup is open we listen for those
 * writes via chrome.storage.onChanged and update the UI immediately.
 * When the popup is reopened mid-run, we read the current value and
 * display it instantly — no stale "0%" flash.
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dotEl         = document.getElementById("dotEl");
const levelCards    = document.getElementById("levelCards");
const progressFill  = document.getElementById("progressFill");
const progressLabel = document.getElementById("progressLabel");
const progressPct   = document.getElementById("progressPct");
const statusMsg     = document.getElementById("statusMsg");
const btnSimplify   = document.getElementById("btnSimplify");
const btnSimplifyTx = document.getElementById("btnSimplifyText");
const btnUndo       = document.getElementById("btnUndo");
const modelLabel    = document.getElementById("modelLabel");

// ── Level card selection ──────────────────────────────────────────────────────
function selectLevel(value) {
  document.querySelectorAll(".level-card").forEach(card => {
    const isMatch = card.dataset.level === value;
    card.classList.toggle("selected", isMatch);
    card.querySelector("input[type=radio]").checked = isMatch;
  });
  chrome.storage.local.set({ araLevel: value });
}

levelCards.addEventListener("click", e => {
  const card = e.target.closest(".level-card");
  if (card) selectLevel(card.dataset.level);
});

// ── Progress UI ───────────────────────────────────────────────────────────────
function updateProgress(pct, label, message, isSuccess = false, isError = false) {
  const clamped = Math.max(0, Math.min(100, pct));

  progressFill.style.width = `${clamped}%`;
  progressFill.classList.toggle("done", clamped === 100 && !isError);

  progressPct.textContent = clamped > 0 ? `${clamped}%` : "—";
  progressLabel.textContent = label || "Ready";

  statusMsg.textContent = message || "";
  statusMsg.className = "status-msg"
    + (isSuccess ? " success" : "")
    + (isError   ? " error"   : "");
}

// ── Server health check ───────────────────────────────────────────────────────
async function checkServer() {
  try {
    const res  = await fetch("http://localhost:5000/ping", { signal: AbortSignal.timeout(2500) });
    const data = await res.json();
    dotEl.className = "status-dot"; // green
    dotEl.title = "Server online";
    // Show the model name from /ping (set in app.py)
    if (data.model) {
      const short = data.model.includes("/")
        ? data.model.split("/").pop()
        : data.model;
      modelLabel.textContent = `Model: ${short}`;
    }
  } catch {
    dotEl.className = "status-dot offline";
    dotEl.title = "Server offline — start app.py";
    modelLabel.textContent = "Model: offline";
    statusMsg.textContent = "⚠ Server not running. Start app.py first.";
    statusMsg.className = "status-msg error";
    btnSimplify.disabled = true;
  }
}

// ── Restore UI from stored state ──────────────────────────────────────────────
function restoreState(state) {
  // Restore selected level
  const level = state.araLevel || "intermediate";
  selectLevel(level);

  // Restore progress
  const pct     = state.araProgress ?? 0;
  const total   = state.araTotal    ?? 0;
  const done    = state.araDone     ?? 0;
  const running = state.araRunning  ?? false;
  const hasDone = state.araHasDone  ?? false;
  const msg     = state.araStatus   ?? "";

  if (pct > 0 || running) {
    const label = running
      ? `Processing ${done} of ${total}…`
      : pct === 100 ? "Complete" : "Paused";
    updateProgress(pct, label, msg, pct === 100 && !running);
  }

  // Show Undo button if there is content to restore
  btnUndo.disabled = !hasDone;

  // Disable Simplify while a run is active
  if (running) setRunningUI(true);
}

// ── Button states ─────────────────────────────────────────────────────────────
function setRunningUI(running) {
  btnSimplify.disabled = running;
  btnUndo.disabled     = running;

  if (running) {
    btnSimplify.querySelector(".btn-icon").textContent = "";
    // Add spinner
    if (!btnSimplify.querySelector(".spinner")) {
      const s = document.createElement("div");
      s.className = "spinner";
      btnSimplify.insertBefore(s, btnSimplifyTx);
    }
    btnSimplifyTx.textContent = "Simplifying…";
  } else {
    btnSimplify.querySelector(".spinner")?.remove();
    btnSimplify.querySelector(".btn-icon").textContent = "✨";
    btnSimplifyTx.textContent = "Simplify This Page";
  }
}

// ── Listen for live storage changes (while popup is open) ─────────────────────
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;

  const pct     = changes.araProgress?.newValue;
  const msg     = changes.araStatus?.newValue;
  const running = changes.araRunning?.newValue;
  const hasDone = changes.araHasDone?.newValue;
  const total   = changes.araTotal?.newValue;
  const done    = changes.araDone?.newValue;

  // Pull current values for anything not in this change batch
  chrome.storage.local.get(
    ["araProgress","araStatus","araRunning","araHasDone","araTotal","araDone"],
    cur => {
      const p  = pct     ?? cur.araProgress ?? 0;
      const m  = msg     ?? cur.araStatus   ?? "";
      const r  = running ?? cur.araRunning  ?? false;
      const hd = hasDone ?? cur.araHasDone  ?? false;
      const to = total   ?? cur.araTotal    ?? 0;
      const d  = done    ?? cur.araDone     ?? 0;

      const label = r ? `Processing ${d} of ${to}…`
                      : p === 100 ? "Complete" : "Ready";
      updateProgress(p, label, m, p === 100 && !r);
      setRunningUI(r);
      btnUndo.disabled = !hd || r;
    }
  );
});

// ── Simplify button ───────────────────────────────────────────────────────────
btnSimplify.addEventListener("click", async () => {
  const selectedCard = document.querySelector(".level-card.selected");
  const userLevel    = selectedCard?.dataset.level || "intermediate";

  // Reset progress in storage for a fresh run
  chrome.storage.local.set({
    araRunning:  true,
    araProgress: 0,
    araStatus:   "Starting…",
    araTotal:    0,
    araDone:     0,
    araHasDone:  false,
    araLevel:    userLevel,
  });
  setRunningUI(true);
  updateProgress(0, "Starting…", "Sending paragraphs to the AI…");

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    updateProgress(0, "Error", "No active tab found.", false, true);
    setRunningUI(false);
    return;
  }

  // Inject the content script if it isn't already there (e.g. on extension-
  // restricted pages like the Chrome Web Store).
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files:  ["content.js"],
    });
  } catch {
    // Script may already be injected — that's fine, ignore the error.
  }

  chrome.tabs.sendMessage(tab.id, { action: "simplify", userLevel }, response => {
    if (chrome.runtime.lastError) {
      updateProgress(0, "Error", "Content script not ready. Reload the page.", false, true);
      chrome.storage.local.set({ araRunning: false });
      setRunningUI(false);
      return;
    }
    // Content script will update storage on its own; we just mark not running.
    if (response?.status === "error") {
      updateProgress(0, "Error", response.error || "Unknown error.", false, true);
      chrome.storage.local.set({ araRunning: false });
      setRunningUI(false);
    }
    // Success path: storage.onChanged will update the UI as batches complete.
  });
});

// ── Undo button ───────────────────────────────────────────────────────────────
btnUndo.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  chrome.tabs.sendMessage(tab.id, { action: "undo" }, () => {
    // Storage is updated by content.js; UI follows via onChanged listener.
  });
});

// ── Initialise ────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // 1. Restore state from previous session / in-progress run
  chrome.storage.local.get(
    ["araLevel","araProgress","araStatus","araRunning","araHasDone","araTotal","araDone"],
    state => restoreState(state)
  );

  // 2. Check server health (sets the dot colour)
  checkServer();
});