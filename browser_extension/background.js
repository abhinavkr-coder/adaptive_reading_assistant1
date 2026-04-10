// Minimal MV3 service worker.
// The popup communicates directly with the content script,
// so no message forwarding is needed here.
chrome.runtime.onInstalled.addListener(() => {
  console.log("[ARA] Adaptive Reading Assistant installed.");
});