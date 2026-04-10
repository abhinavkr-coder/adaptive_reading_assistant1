document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("level");
  const btn    = document.getElementById("btn");
  const status = document.getElementById("status");

  // Restore the user's saved preference
  chrome.storage.local.get("userLevel", ({ userLevel }) => {
    if (userLevel) select.value = userLevel;
  });

  select.addEventListener("change", () => {
    chrome.storage.local.set({ userLevel: select.value });
  });

  btn.addEventListener("click", async () => {
    btn.disabled    = true;
    status.textContent = "Simplifying page…";

    const userLevel = select.value;
    chrome.storage.local.set({ userLevel });

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) {
      status.textContent = "No active tab found.";
      btn.disabled = false;
      return;
    }

    chrome.tabs.sendMessage(tab.id, { action: "simplify", userLevel }, (response) => {
      if (chrome.runtime.lastError) {
        status.textContent = "Content script not ready — reload the page.";
      } else if (response?.status === "done") {
        status.textContent = "Done! Hover blue text to see the original.";
      } else {
        status.textContent = "Error occurred. Is the server running?";
      }
      btn.disabled = false;
    });
  });
});