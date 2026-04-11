// Adaptive Reading Assistant — MV3 service worker.
//
// The popup communicates directly with the content script via
// chrome.tabs.sendMessage; the content script writes progress to
// chrome.storage.local which the popup reads via storage.onChanged.
// No message forwarding or relay is needed here.

chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.log(`[ARA] Installed (reason: ${reason}).`);

  // Clear any stale progress state from a previous install so the
  // popup doesn't show a ghost progress bar on first use.
  if (reason === "install") {
    chrome.storage.local.clear();
  }
});