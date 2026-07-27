/* =========================================================================
   detector.js
   -----------
   All the logic for the Detector page:
     1. Switching between Image / Video / Text tabs
     2. Drag-and-drop + click-to-browse file uploads
     3. File validation (type + size)
     4. Sending data to the Flask backend and receiving the result
     5. Showing loading states / progress bar
     6. Rendering the result card
     7. Copy result + Download report buttons
     8. Saving each result into "Recent History" (see history.js)

   IMPORTANT: change BACKEND_URL below to match where your Flask server is
   running. While testing on your own computer, it's usually:
       http://127.0.0.1:5000
   When you deploy the backend to Render, change this to your Render URL,
   e.g. https://ai-content-detector-backend.onrender.com
   ========================================================================= */

const BACKEND_URL = "https://ai-content-detector-lixc.onrender.com";

document.addEventListener('DOMContentLoaded', () => {

  /* ============ TAB SWITCHING ============ */
  const tabs = document.querySelectorAll('.mode-tab');
  const panels = document.querySelectorAll('.detector-panel');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      panels.forEach((p) => p.classList.remove('active'));

      tab.classList.add('active');
      document.getElementById(tab.dataset.target).classList.add('active');
    });
  });

  /* ============ IMAGE DETECTION ============ */
  setupFileDetector({
    zoneId: 'imageUploadZone',
    inputId: 'imageInput',
    previewId: 'imagePreview',
    previewMediaId: 'imagePreviewMedia',
    metaId: 'imageFileMeta',
    analyzeBtnId: 'imageAnalyzeBtn',
    loadingId: 'imageLoading',
    resultId: 'imageResult',
    errorId: 'imageError',
    allowedExt: ['jpg', 'jpeg', 'png'],
    maxSizeMB: 10,
    endpoint: '/api/detect/image',
    historyType: 'Image',
    isVideo: false,
  });

  /* ============ VIDEO DETECTION ============ */
  setupFileDetector({
    zoneId: 'videoUploadZone',
    inputId: 'videoInput',
    previewId: 'videoPreview',
    previewMediaId: 'videoPreviewMedia',
    metaId: 'videoFileMeta',
    analyzeBtnId: 'videoAnalyzeBtn',
    loadingId: 'videoLoading',
    resultId: 'videoResult',
    errorId: 'videoError',
    allowedExt: ['mp4', 'mov', 'avi'],
    maxSizeMB: 100,
    endpoint: '/api/detect/video',
    historyType: 'Video',
    isVideo: true,
    progressId: 'videoProgress',
  });

  /* ============ TEXT DETECTION ============ */
  setupTextDetector();

});

/* -------------------------------------------------------------------------
   Reusable setup function for BOTH image and video detectors, since they
   work almost identically (upload a file -> preview it -> analyze it).
   ------------------------------------------------------------------------- */
function setupFileDetector(cfg) {
  const zone = document.getElementById(cfg.zoneId);
  const input = document.getElementById(cfg.inputId);
  if (!zone || !input) return;

  const previewCard = document.getElementById(cfg.previewId);
  const previewMedia = document.getElementById(cfg.previewMediaId);
  const meta = document.getElementById(cfg.metaId);
  const analyzeBtn = document.getElementById(cfg.analyzeBtnId);
  const loading = document.getElementById(cfg.loadingId);
  const resultCard = document.getElementById(cfg.resultId);
  const errorNote = document.getElementById(cfg.errorId);
  const progressWrap = cfg.progressId ? document.getElementById(cfg.progressId) : null;

  let selectedFile = null;

  // Click anywhere on the drop zone opens the file browser
  zone.addEventListener('click', () => input.click());

  // Drag-and-drop visual feedback + handling
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  input.addEventListener('change', () => {
    if (input.files.length) handleFile(input.files[0]);
  });

  function handleFile(file) {
    hideError();
    const ext = file.name.split('.').pop().toLowerCase();

    // --- Validation: file type ---
    if (!cfg.allowedExt.includes(ext)) {
      showError(`Unsupported file type ".${ext}". Allowed types: ${cfg.allowedExt.join(', ')}.`);
      return;
    }

    // --- Validation: file size ---
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > cfg.maxSizeMB) {
      showError(`File is too large (${sizeMB.toFixed(1)}MB). Maximum allowed is ${cfg.maxSizeMB}MB.`);
      return;
    }

    selectedFile = file;
    resultCard.classList.remove('show');

    // Show preview
    const url = URL.createObjectURL(file);
    previewMedia.src = url;
    meta.textContent = `${file.name} · ${sizeMB.toFixed(2)}MB`;
    previewCard.classList.add('show');
  }

  function hideError() { errorNote.classList.remove('show'); errorNote.textContent = ''; }
  function showError(msg) { errorNote.textContent = msg; errorNote.classList.add('show'); }

  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      showError('Please choose a file first.');
      return;
    }

    hideError();
    resultCard.classList.remove('show');
    loading.classList.add('show');
    analyzeBtn.disabled = true;

    // For video, show a simulated progress bar since a real upload-progress
    // event isn't available with plain fetch(). This gives the user useful
    // visual feedback while the analysis happens.
    let progressInterval = null;
    if (progressWrap) {
      progressWrap.classList.add('show');
      const fill = progressWrap.querySelector('.progress-fill');
      const label = progressWrap.querySelector('.progress-label');
      let pct = 0;
      progressInterval = setInterval(() => {
        pct = Math.min(pct + Math.random() * 12, 92);
        fill.style.width = pct + '%';
        label.textContent = `Analyzing frames... ${Math.round(pct)}%`;
      }, 400);
    }

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(BACKEND_URL + cfg.endpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (progressInterval) {
        clearInterval(progressInterval);
        const fill = progressWrap.querySelector('.progress-fill');
        const label = progressWrap.querySelector('.progress-label');
        fill.style.width = '100%';
        label.textContent = 'Done!';
      }

      if (!response.ok) {
        showError(data.error || 'Something went wrong. Please try again.');
        return;
      }

      renderResult(resultCard, data, cfg.historyType);
    } catch (err) {
      showError('Could not reach the server. Is the Flask backend running at ' + BACKEND_URL + '?');
    } finally {
      loading.classList.remove('show');
      analyzeBtn.disabled = false;
    }
  });
}

/* -------------------------------------------------------------------------
   TEXT DETECTOR
   ------------------------------------------------------------------------- */
function setupTextDetector() {
  const textArea = document.getElementById('textInput');
  const wordCountEl = document.getElementById('wordCount');
  const charCountEl = document.getElementById('charCount');
  const analyzeBtn = document.getElementById('textAnalyzeBtn');
  const loading = document.getElementById('textLoading');
  const resultCard = document.getElementById('textResult');
  const errorNote = document.getElementById('textError');

  if (!textArea) return;

  textArea.addEventListener('input', () => {
    const text = textArea.value;
    const words = text.trim().length ? text.trim().split(/\s+/).length : 0;
    wordCountEl.textContent = words;
    charCountEl.textContent = text.length;
  });

  analyzeBtn.addEventListener('click', async () => {
    const text = textArea.value.trim();
    errorNote.classList.remove('show');
    resultCard.classList.remove('show');

    if (text.length < 10) {
      errorNote.textContent = 'Please enter at least a short sentence to analyze.';
      errorNote.classList.add('show');
      return;
    }

    loading.classList.add('show');
    analyzeBtn.disabled = true;

    try {
      const response = await fetch(BACKEND_URL + '/api/detect/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();

      if (!response.ok) {
        errorNote.textContent = data.error || 'Something went wrong. Please try again.';
        errorNote.classList.add('show');
        return;
      }

      renderResult(resultCard, data, 'Text');
    } catch (err) {
      errorNote.textContent = 'Could not reach the server. Is the Flask backend running at ' + BACKEND_URL + '?';
      errorNote.classList.add('show');
    } finally {
      loading.classList.remove('show');
      analyzeBtn.disabled = false;
    }
  });
}

/* -------------------------------------------------------------------------
   SHARED: renders a result object { result, confidence, details } into a
   given result-card element, wires up Copy/Download buttons, and saves
   the result to local history.
   ------------------------------------------------------------------------- */
function renderResult(cardEl, data, historyType) {
  const isAI = data.result === 'Likely AI Generated';
  const flagClass = isAI ? 'flag-ai' : 'flag-human';
  const textClass = isAI ? 'is-ai' : 'is-human';

  const detailsHtml = data.details
    ? Object.entries(data.details).map(([key, value]) => `
        <div>${formatKey(key)}: <b>${value}</b></div>
      `).join('')
    : '';

  cardEl.className = `result-card show ${flagClass}`;
  cardEl.innerHTML = `
    <div class="result-top">
      <div class="result-label ${textClass}">${data.result}</div>
      <div class="result-tag">${historyType} scan</div>
    </div>
    <div class="confidence-readout">
      <span class="num ${textClass}">${data.confidence}</span>
      <span class="unit">% confidence</span>
    </div>
    <div class="meter-track">
      <div class="meter-fill" style="width:${data.confidence}%; background:${isAI ? 'var(--alert)' : 'var(--verified)'}"></div>
    </div>
    ${detailsHtml ? `<div class="result-details">${detailsHtml}</div>` : ''}
    <div class="result-actions">
      <button class="btn btn-ghost" id="copyResultBtn">Copy result</button>
      <button class="btn btn-ghost" id="downloadReportBtn">Download report</button>
    </div>
  `;

  // Wire up Copy button
  cardEl.querySelector('#copyResultBtn').addEventListener('click', () => {
    const text = `${data.result} (${data.confidence}% confidence)`;
    navigator.clipboard.writeText(text).then(() => {
      const btn = cardEl.querySelector('#copyResultBtn');
      const original = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = original; }, 1500);
    });
  });

  // Wire up Download report button (creates a simple .txt report)
  cardEl.querySelector('#downloadReportBtn').addEventListener('click', () => {
    const lines = [
      'AI CONTENT DETECTOR — REPORT',
      '-----------------------------',
      `Content type: ${historyType}`,
      `Result: ${data.result}`,
      `Confidence: ${data.confidence}%`,
      `Date: ${new Date().toLocaleString()}`,
      '',
      'Details:',
      ...(data.details ? Object.entries(data.details).map(([k, v]) => `- ${formatKey(k)}: ${v}`) : []),
      '',
      'Note: This is an estimate produced by an automated tool and is not',
      'proof of authorship. Free AI-detection methods can be inaccurate.',
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-detector-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Save to Recent History (function lives in history.js)
  if (typeof addToHistory === 'function') {
    addToHistory(historyType, data.result, data.confidence);
  }
}

function formatKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
