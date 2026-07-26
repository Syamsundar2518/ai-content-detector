/* =========================================================================
   history.js
   ----------
   "Recent Detection History" feature.

   We use the browser's built-in "localStorage" — a small storage box that
   belongs to your browser and stays saved even after you close the tab.
   No backend/database needed for this feature; it's simple and free.

   We store history as one JSON array under the key "ai-detector-history",
   like this:
   [
     { "type": "Text", "result": "Likely AI Generated", "confidence": 87, "date": "..." },
     ...
   ]
   ========================================================================= */

const HISTORY_KEY = 'ai-detector-history';
const MAX_HISTORY_ITEMS = 20;

/** Reads the history array from localStorage (or returns an empty array). */
function getHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Could not read history:', e);
    return [];
  }
}

/** Adds one new detection result to the top of the history list. */
function addToHistory(type, result, confidence) {
  const history = getHistory();

  history.unshift({
    type,
    result,
    confidence,
    date: new Date().toLocaleString(),
  });

  // Keep only the most recent MAX_HISTORY_ITEMS entries so storage doesn't grow forever
  const trimmed = history.slice(0, MAX_HISTORY_ITEMS);

  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch (e) {
    console.error('Could not save history:', e);
  }

  renderHistory();
}

/** Wipes all saved history. */
function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
}

/** Draws the history list onto the page, if a history list container exists. */
function renderHistory() {
  const listEl = document.getElementById('historyList');
  if (!listEl) return; // this page doesn't have a history section

  const history = getHistory();

  if (history.length === 0) {
    listEl.innerHTML = '<p class="history-empty">No detections yet. Your recent results will show up here.</p>';
    return;
  }

  listEl.innerHTML = history.map((item) => {
    const isAI = item.result.toLowerCase().includes('ai generated');
    const resultClass = isAI ? 'is-ai' : 'is-human';
    return `
      <div class="history-item">
        <div class="h-left">
          <span class="h-type">${item.type}</span>
          <span>${item.date}</span>
        </div>
        <div class="h-result ${resultClass}">${item.result} · ${item.confidence}%</div>
      </div>
    `;
  }).join('');
}

document.addEventListener('DOMContentLoaded', () => {
  renderHistory();

  const clearBtn = document.getElementById('clearHistoryBtn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (confirm('Clear all detection history? This cannot be undone.')) {
        clearHistory();
      }
    });
  }
});
