"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dotEl      = document.getElementById("dotEl");
const levelGrid  = document.getElementById("levelGrid");
const progFill   = document.getElementById("progFill");
const progLabel  = document.getElementById("progLabel");
const progPct    = document.getElementById("progPct");
const progStatus = document.getElementById("progStatus");
const btnSimplify= document.getElementById("btnSimplify");
const btnTx      = document.getElementById("btnTx");
const btnUndo    = document.getElementById("btnUndo");
const modelLbl   = document.getElementById("modelLbl");


// ── Level selection ───────────────────────────────────────────────────────────

function selectLevel(value) {
  document.querySelectorAll(".level-card").forEach(card => {
    const match = card.dataset.level === value;
    card.classList.toggle("selected", match);
    card.querySelector("input[type=radio]").checked = match;
  });
  chrome.storage.local.set({ araLevel: value });
}

levelGrid.addEventListener("click", e => {
  const card = e.target.closest(".level-card");
  if (card) selectLevel(card.dataset.level);
});


// ── Progress UI ───────────────────────────────────────────────────────────────

function setProgress(pct, label, msg, tone = "neutral") {
  const clamped = Math.max(0, Math.min(100, pct));

  progFill.style.width = `${clamped}%`;
  progFill.classList.toggle("done",    clamped === 100 && tone !== "error");
  progFill.classList.toggle("running", tone === "running");

  progPct.textContent   = clamped > 0 ? `${clamped}%` : "—";
  progLabel.textContent = label || "Ready";

  progStatus.textContent = msg || "";
  progStatus.className   = "prog-status"
    + (tone === "ok"    ? " ok"  : "")
    + (tone === "error" ? " err" : "");
}


// ── Server health check ───────────────────────────────────────────────────────

async function checkServer() {
  try {
    const res  = await fetch("http://localhost:5000/ping",
      { signal: AbortSignal.timeout(2500) });
    const data = await res.json();

    dotEl.className = "status-dot online";
    dotEl.title     = "Server online";

    if (data.model) {
      const short = data.model.includes("/")
        ? data.model.split("/").pop()
        : data.model;
      modelLbl.textContent = `Model: ${short}`;
    }
  } catch {
    dotEl.className        = "status-dot offline";
    dotEl.title            = "Server offline — run app.py";
    modelLbl.textContent   = "Model: offline";
    btnSimplify.disabled   = true;
    setProgress(0, "Offline", "⚠ Start app.py to enable simplification.", "error");
  }
}


// ── Restore state on popup open ───────────────────────────────────────────────

function restoreState(s) {
  selectLevel(s.araLevel || "intermediate");

  const pct     = s.araProgress ?? 0;
  const running = s.araRunning  ?? false;
  const hasDone = s.araHasDone  ?? false;
  const msg     = s.araStatus   ?? "";
  const total   = s.araTotal    ?? 0;
  const done    = s.araDone     ?? 0;

  if (pct > 0 || running) {
    const label = running ? `Processing ${done} of ${total}…`
                          : pct === 100 ? "Complete" : "Paused";
    const tone  = running ? "running" : pct === 100 ? "ok" : "neutral";
    setProgress(pct, label, msg, tone);
  }

  btnUndo.disabled = !hasDone;
  if (running) setRunning(true);
}


// ── Button state helpers ──────────────────────────────────────────────────────

function setRunning(running) {
  btnSimplify.disabled = running;
  btnUndo.disabled     = running;

  if (running) {
    if (!btnSimplify.querySelector(".spinner")) {
      const s = document.createElement("div");
      s.className = "spinner";
      btnSimplify.insertBefore(s, btnTx);
    }
    btnSimplify.querySelector(".btn-icon").textContent = "";
    btnTx.textContent = "Simplifying…";
  } else {
    btnSimplify.querySelector(".spinner")?.remove();
    btnSimplify.querySelector(".btn-icon").textContent = "✨";
    btnTx.textContent = "Simplify This Page";
  }
}


// ── Live storage updates (while popup is open) ────────────────────────────────

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;

  chrome.storage.local.get(
    ["araProgress","araStatus","araRunning","araHasDone","araTotal","araDone"],
    cur => {
      const pct     = changes.araProgress?.newValue ?? cur.araProgress ?? 0;
      const msg     = changes.araStatus?.newValue   ?? cur.araStatus   ?? "";
      const running = changes.araRunning?.newValue  ?? cur.araRunning  ?? false;
      const hasDone = changes.araHasDone?.newValue  ?? cur.araHasDone  ?? false;
      const total   = changes.araTotal?.newValue    ?? cur.araTotal    ?? 0;
      const done    = changes.araDone?.newValue     ?? cur.araDone     ?? 0;

      const label = running ? `Processing ${done} of ${total}…`
                            : pct === 100 ? "Complete" : "Ready";
      const tone  = running ? "running" : pct === 100 ? "ok" : "neutral";

      setProgress(pct, label, msg, tone);
      setRunning(running);
      btnUndo.disabled = !hasDone || running;
    }
  );
});


// ── Simplify button ───────────────────────────────────────────────────────────

btnSimplify.addEventListener("click", async () => {
  const selected  = document.querySelector(".level-card.selected");
  const userLevel = selected?.dataset.level || "intermediate";

  chrome.storage.local.set({
    araRunning:  true,
    araProgress: 0,
    araStatus:   "Sending paragraphs to the AI…",
    araTotal: 0, araDone: 0, araHasDone: false,
    araLevel: userLevel,
  });
  setRunning(true);
  setProgress(0, "Starting…", "Sending paragraphs to the AI…", "running");

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    setProgress(0, "Error", "No active tab found.", "error");
    chrome.storage.local.set({ araRunning: false });
    setRunning(false);
    return;
  }

  // Re-inject content script in case it's not present (e.g. after extension update)
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
  } catch { /* already injected — ignore */ }

  chrome.tabs.sendMessage(tab.id, { action: "simplify", userLevel }, res => {
    if (chrome.runtime.lastError || res?.status === "error") {
      const msg = chrome.runtime.lastError?.message || res?.error || "Unknown error.";
      setProgress(0, "Error", msg, "error");
      chrome.storage.local.set({ araRunning: false });
      setRunning(false);
    }
    // Success: content.js drives progress via storage.onChanged
  });
});


// ── Undo button ───────────────────────────────────────────────────────────────

btnUndo.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  chrome.tabs.sendMessage(tab.id, { action: "undo" });
});


// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  chrome.storage.local.get(
    ["araLevel","araProgress","araStatus","araRunning","araHasDone","araTotal","araDone"],
    restoreState
  );
  checkServer();
});