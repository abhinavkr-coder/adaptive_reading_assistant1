"use strict";

// ══════════════════════════════════════════════════════════════
//  SCREEN MANAGEMENT
//
//  The popup has two screens: onboarding (#s-onboard) and the
//  main app (#s-main).  We persist a flag "araOnboarded" so
//  returning users skip straight to the main screen.
//  Both screens are always in the DOM; we toggle display + fade.
// ══════════════════════════════════════════════════════════════

const ONBOARD_KEY = "araOnboarded";

// True once initMain() has wired up event listeners.
// Prevents duplicate listener binding if the user navigates
// back to onboarding and then returns to the main screen.
let _mainReady = false;

// Module-level DOM references — populated by grabRefs()
// after the main screen is first made visible.
let dotEl, levelGrid, progFill, progLabel, progPct, progMsg,
    btnSimplify, btnTx, btnUndo, modelLbl;


// ── Transition: onboarding → main ────────────────────────────

function showMain(animate) {
  const ob   = document.getElementById("s-onboard");
  const main = document.getElementById("s-main");

  if (!animate) {
    // Instant swap for returning users — no flicker
    ob.style.display   = "none";
    main.style.display = "flex";
    main.style.opacity = "1";
    initMain();
    return;
  }

  // Fade the onboarding out, then fade the main screen in.
  ob.style.transition = "opacity 0.32s ease";
  ob.style.opacity    = "0";

  setTimeout(() => {
    ob.style.display    = "none";
    main.style.display  = "flex";
    main.style.opacity  = "0";
    main.style.transition = "opacity 0.28s ease";

    // Two rAF calls ensure the display change is painted
    // before the opacity transition begins, avoiding a flash.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        main.style.opacity = "1";
        initMain();
      });
    });
  }, 320);
}


// ── Transition: main → onboarding ────────────────────────────

function showOnboard() {
  const ob   = document.getElementById("s-onboard");
  const main = document.getElementById("s-main");

  main.style.transition = "opacity 0.24s ease";
  main.style.opacity    = "0";

  setTimeout(() => {
    main.style.display  = "none";
    ob.style.display    = "block";
    ob.style.opacity    = "0";
    ob.style.transition = "opacity 0.28s ease";

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        ob.style.opacity = "1";
      });
    });
  }, 240);
}


// ══════════════════════════════════════════════════════════════
//  MAIN APP INITIALISATION
// ══════════════════════════════════════════════════════════════

function initMain() {
  grabRefs();

  if (!_mainReady) {
    // Wire up event listeners exactly once — even if the user
    // goes back to onboarding and returns.
    bindMainEvents();
    _mainReady = true;
  }

  // Restore last-used state from storage, then ping the server.
  chrome.storage.local.get(
    ["araLevel","araProgress","araStatus","araRunning",
     "araHasDone","araTotal","araDone"],
    state => {
      restoreState(state);
      checkServer();
    }
  );
}

function grabRefs() {
  dotEl       = document.getElementById("dotEl");
  levelGrid   = document.getElementById("levelGrid");
  progFill    = document.getElementById("progFill");
  progLabel   = document.getElementById("progLabel");
  progPct     = document.getElementById("progPct");
  progMsg     = document.getElementById("progMsg");
  btnSimplify = document.getElementById("btnSimplify");
  btnTx       = document.getElementById("btnTx");
  btnUndo     = document.getElementById("btnUndo");
  modelLbl    = document.getElementById("modelLbl");
}


// ══════════════════════════════════════════════════════════════
//  LEVEL SELECTION
// ══════════════════════════════════════════════════════════════

function selectLevel(value) {
  document.querySelectorAll(".lvl-card").forEach(card => {
    const match = card.dataset.level === value;
    card.classList.toggle("sel", match);
    card.querySelector("input[type=radio]").checked = match;
  });
  chrome.storage.local.set({ araLevel: value });
}


// ══════════════════════════════════════════════════════════════
//  PROGRESS UI
// ══════════════════════════════════════════════════════════════

/**
 * @param {number} pct   0-100
 * @param {string} label Text left of the bar ("Ready", "Processing…")
 * @param {string} msg   Text below the bar
 * @param {"neutral"|"running"|"ok"|"error"} tone
 */
function setProgress(pct, label, msg, tone = "neutral") {
  if (!progFill) return;  // guard: called before refs are grabbed
  const c = Math.max(0, Math.min(100, pct));

  progFill.style.width = `${c}%`;
  progFill.classList.toggle("done",    c === 100 && tone !== "error" && tone !== "running");
  progFill.classList.toggle("running", tone === "running");

  progPct.textContent   = c > 0 ? `${c}%` : "—";
  progLabel.textContent = label || "Ready";
  progMsg.textContent   = msg   || "";
  progMsg.className     = "prog-msg"
    + (tone === "ok"    ? " ok"  : "")
    + (tone === "error" ? " err" : "");
}


// ══════════════════════════════════════════════════════════════
//  SERVER HEALTH CHECK
// ══════════════════════════════════════════════════════════════

async function checkServer() {
  if (!dotEl) return;
  dotEl.className = "m-dot checking";
  try {
    const res  = await fetch("http://localhost:5000/ping",
      { signal: AbortSignal.timeout(2500) });
    const data = await res.json();

    dotEl.className = "m-dot online";
    dotEl.title     = "Server online";

    if (data.model) {
      const short = data.model.includes("/")
        ? data.model.split("/").pop()
        : data.model;
      modelLbl.textContent = `Model: ${short}`;
    } else {
      modelLbl.textContent = "AI vocabulary simplifier";
    }
  } catch {
    dotEl.className      = "m-dot offline";
    dotEl.title          = "Server offline — run app.py";
    modelLbl.textContent = "Server offline";
    if (btnSimplify) btnSimplify.disabled = true;
    setProgress(0, "Offline",
      "⚠ Start app.py to enable simplification.", "error");
  }
}


// ══════════════════════════════════════════════════════════════
//  STATE RESTORATION  (called on every popup open)
// ══════════════════════════════════════════════════════════════

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

  if (btnUndo) btnUndo.disabled = !hasDone;
  if (running) setRunning(true);
}


// ══════════════════════════════════════════════════════════════
//  BUTTON STATE
// ══════════════════════════════════════════════════════════════

function setRunning(running) {
  if (!btnSimplify) return;
  btnSimplify.disabled = running;
  if (btnUndo) btnUndo.disabled = running;

  if (running) {
    if (!btnSimplify.querySelector(".spinner")) {
      const s = document.createElement("div");
      s.className = "spinner";
      btnSimplify.insertBefore(s, btnTx);
    }
    const ico = btnSimplify.querySelector(".m-ico");
    if (ico) ico.textContent = "";
    if (btnTx) btnTx.textContent = "Simplifying…";
  } else {
    btnSimplify.querySelector(".spinner")?.remove();
    const ico = btnSimplify.querySelector(".m-ico");
    if (ico) ico.textContent = "✨";
    if (btnTx) btnTx.textContent = "Simplify This Page";
  }
}


// ══════════════════════════════════════════════════════════════
//  LIVE STORAGE LISTENER
//  content.js writes progress to storage on every batch;
//  this listener updates the popup UI in real time.
// ══════════════════════════════════════════════════════════════

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  // Ignore updates when the main screen is not visible yet
  if (!progFill) return;

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
      if (btnUndo) btnUndo.disabled = !hasDone || running;
    }
  );
});


// ══════════════════════════════════════════════════════════════
//  EVENT BINDING (once only, called from initMain)
// ══════════════════════════════════════════════════════════════

function bindMainEvents() {
  // Level card clicks
  document.getElementById("levelGrid").addEventListener("click", e => {
    const card = e.target.closest(".lvl-card");
    if (card) selectLevel(card.dataset.level);
  });

  // Simplify
  document.getElementById("btnSimplify").addEventListener("click", handleSimplify);

  // Undo
  document.getElementById("btnUndo").addEventListener("click", async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) chrome.tabs.sendMessage(tab.id, { action: "undo" });
  });

  // About — slides back to the onboarding screen
  document.getElementById("btnAbout").addEventListener("click", showOnboard);
}


// ── Simplify handler ──────────────────────────────────────────

async function handleSimplify() {
  const selected  = document.querySelector(".lvl-card.sel");
  const userLevel = selected?.dataset.level || "intermediate";

  chrome.storage.local.set({
    araRunning:  true,
    araProgress: 0,
    araStatus:   "Sending paragraphs to the AI…",
    araTotal:    0,
    araDone:     0,
    araHasDone:  false,
    araLevel:    userLevel,
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

  // Re-inject content script if it's not already present
  // (this happens after extension updates or on first install).
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files:  ["content.js"],
    });
  } catch { /* already injected — safe to ignore */ }

  chrome.tabs.sendMessage(
    tab.id,
    { action: "simplify", userLevel },
    res => {
      if (chrome.runtime.lastError || res?.status === "error") {
        const msg = chrome.runtime.lastError?.message
                  || res?.error || "Unknown error.";
        setProgress(0, "Error", msg, "error");
        chrome.storage.local.set({ araRunning: false });
        setRunning(false);
      }
      // Success path: content.js updates storage on each batch,
      // and the storage.onChanged listener above updates the UI.
    }
  );
}


// ══════════════════════════════════════════════════════════════
//  BOOT  (runs on every popup open)
// ══════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  // Wire up the Get Started button on the onboarding screen
  document.getElementById("btn-start").addEventListener("click", () => {
    chrome.storage.local.set({ [ONBOARD_KEY]: true });
    showMain(true);   // animated transition
  });

  // Decide which screen to show
  chrome.storage.local.get([ONBOARD_KEY], result => {
    if (result[ONBOARD_KEY]) {
      // Returning user — skip the animation, go straight to main
      showMain(false);
    }
    // New user — onboarding is the default visible state
  });
});