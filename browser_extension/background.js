// Adaptive Reading Assistant — MV3 service worker.
// No relay needed: popup ↔ content script communicate directly.

chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.log(`[ARA] Installed (reason: ${reason}).`);
  if (reason === "install") {
    chrome.storage.local.clear();
  }
});