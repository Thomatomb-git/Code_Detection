/* ============================================================
   STATE
============================================================ */
let activeTab    = 'paste'; // 'paste' | 'upload'
let selectedFile = null;    // File object dari drag & drop / picker
let fileContent  = '';      // Isi file setelah dibaca

/* ============================================================
   REFERENSI ELEMEN DOM
============================================================ */
const elTabPaste     = document.getElementById('tab-paste');
const elTabUpload    = document.getElementById('tab-upload');
const elPanelPaste   = document.getElementById('panel-paste');
const elPanelUpload  = document.getElementById('panel-upload');
const elCodeInput    = document.getElementById('code-input');
const elModelSelect  = document.getElementById('model-select');
const elCodeMeta     = document.getElementById('code-meta');
const elBtnAnalyze   = document.getElementById('btn-analyze');

const elDropzone     = document.getElementById('dropzone');
const elBtnBrowse    = document.getElementById('btn-browse');
const elFileInput    = document.getElementById('file-input');
const elFileInfo     = document.getElementById('file-info');
const elFileName     = document.getElementById('file-name');
const elBtnRemove    = document.getElementById('btn-remove');

const elResultCard   = document.getElementById('result-card');
const elVerdictIcon  = document.getElementById('verdict-icon');
const elVerdictIconI = document.getElementById('verdict-icon-i');
const elVerdictText  = document.getElementById('verdict-text');
const elConfVal      = document.getElementById('confidence-value');
const elConfFill     = document.getElementById('confidence-fill');
const elMetaModel    = document.getElementById('meta-model');
const elMetaTime     = document.getElementById('meta-time');

/* ============================================================
   TAB SWITCHING
============================================================ */
function switchTab(to) {
  activeTab = to;

  const isPaste = to === 'paste';

  elTabPaste.classList.toggle('active', isPaste);
  elTabUpload.classList.toggle('active', !isPaste);
  elTabPaste.setAttribute('aria-selected', isPaste ? 'true' : 'false');
  elTabUpload.setAttribute('aria-selected', isPaste ? 'false' : 'true');

  elPanelPaste.classList.toggle('hidden', !isPaste);
  elPanelUpload.classList.toggle('hidden', isPaste);

  updateMeta();
}

elTabPaste.addEventListener('click', () => switchTab('paste'));
elTabUpload.addEventListener('click', () => switchTab('upload'));

/* ============================================================
   META INFO (bahasa, jumlah baris, karakter)
============================================================ */
function detectLanguage(code, filename) {
  if (filename) {
    if (filename.endsWith('.py'))                              return 'Python';
    if (filename.endsWith('.cpp') || filename.endsWith('.cc')) return 'C++';
  }
  // Deteksi sederhana dari isi kode
  if (code.includes('def ') || code.includes('import ') || code.includes('print(')) return 'Python';
  if (code.includes('#include') || code.includes('cout') || code.includes('int main')) return 'C++';
  return null;
}

function updateMeta() {
  const code     = activeTab === 'paste' ? elCodeInput.value : fileContent;
  const filename = selectedFile ? selectedFile.name : null;

  if (!code.trim() && !filename) {
    elCodeMeta.innerHTML = '';
    return;
  }

  const lang  = detectLanguage(code, filename);
  const lines = code.trim() ? code.trim().split('\n').length : 0;
  const chars = code.length;

  const iconClass = lang === 'Python' ? 'ti-brand-python' : 'ti-code';
  const parts = [];

  if (lang)  parts.push(`<i class="ti ${iconClass}" aria-hidden="true"></i>${lang}`);
  if (lines) parts.push(`${lines} baris`);
  if (chars) parts.push(`${chars} karakter`);

  elCodeMeta.innerHTML = parts.join(' &nbsp;·&nbsp; ');
}

elCodeInput.addEventListener('input', updateMeta);

/* ============================================================
   DRAG & DROP + FILE PICKER
============================================================ */
function handleFile(file) {
  if (!file) return;

  const ext = file.name.split('.').pop().toLowerCase();
  if (!['py', 'cpp', 'cc'].includes(ext)) {
    alert('Hanya file .py dan .cpp yang didukung.');
    return;
  }

  selectedFile = file;
  elFileName.textContent = file.name;

  // Sembunyikan dropzone, tampilkan info file
  elDropzone.classList.add('hidden');
  elFileInfo.classList.remove('hidden');

  // Baca isi file sebagai teks
  const reader = new FileReader();
  reader.onload = (e) => {
    fileContent = e.target.result;
    updateMeta();
  };
  reader.readAsText(file);
}

function clearFile() {
  selectedFile = null;
  fileContent  = '';
  elFileInput.value = '';

  elDropzone.classList.remove('hidden');
  elFileInfo.classList.add('hidden');

  updateMeta();
}

// Drag events pada dropzone
elDropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  elDropzone.classList.add('dragover');
});

elDropzone.addEventListener('dragleave', (e) => {
  // Hapus class hanya jika kursor benar-benar keluar dari dropzone
  if (!elDropzone.contains(e.relatedTarget)) {
    elDropzone.classList.remove('dragover');
  }
});

elDropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  elDropzone.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

// Klik area dropzone → buka file picker
elDropzone.addEventListener('click', () => elFileInput.click());

// Keyboard: Enter atau Space pada dropzone → buka file picker
elDropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    elFileInput.click();
  }
});

// Tombol "Pilih file" — stopPropagation agar tidak men-trigger click dropzone dua kali
elBtnBrowse.addEventListener('click', (e) => {
  e.stopPropagation();
  elFileInput.click();
});

elFileInput.addEventListener('change', () => handleFile(elFileInput.files[0]));
elBtnRemove.addEventListener('click', clearFile);

/* ============================================================
   ANALYZE
============================================================ */
function getCurrentCode() {
  return activeTab === 'paste' ? elCodeInput.value.trim() : fileContent.trim();
}

function showResult({ verdict, confidence, model, durationMs }) {
  const isHuman = verdict === 'human';

  // Ikon verdict
  elVerdictIcon.className = `verdict-icon ${verdict}`;
  elVerdictIconI.className = `ti ${isHuman ? 'ti-user-check' : 'ti-robot'}`;

  // Teks verdict
  elVerdictText.className = `verdict-text ${verdict}`;
  elVerdictText.textContent = isHuman ? 'Written by Human' : 'Generated by AI';

  // Confidence
  elConfVal.textContent = `${confidence.toFixed(1)}%`;
  elConfFill.className = `confidence-fill ${verdict}`;
  elConfFill.style.width = '0'; // Reset sebelum animasi

  // Tampilkan result card
  elResultCard.classList.remove('hidden');

  // Animasi confidence bar — double rAF agar transisi CSS berjalan setelah elemen visible
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      elConfFill.style.width = `${confidence}%`;
    });
  });

  // Meta info
  elMetaModel.textContent = model;
  elMetaTime.textContent  = `${durationMs}ms`;

  // Scroll ke result card jika perlu
  elResultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

elBtnAnalyze.addEventListener('click', async () => {
  const code = getCurrentCode();

  if (!code) {
    const msg = activeTab === 'paste'
      ? 'Masukkan kode terlebih dahulu.'
      : 'Pilih file terlebih dahulu.';
    alert(msg);
    return;
  }

  // Set tombol ke loading state
  elBtnAnalyze.disabled = true;
  elBtnAnalyze.innerHTML = '<span class="spinner"></span> Analyzing...';

  // Sembunyikan result lama & reset bar
  elResultCard.classList.add('hidden');
  elConfFill.style.width = '0';

  // Jalankan analisis (ganti dengan fetch ke API backend)
  await analyze(code);
});

const API_URL = 'https://thomatomb-aicodetrace-backend.hf.space/predict';
async function analyze(code) {
  const startTime = Date.now();

  // Validasi sebelum mengirim ke API
  if (code.length < 10) {
    resetButton();
    alert('Kode terlalu pendek untuk dianalisis. Minimal 10 karakter.');
    return;
  }

  let data;

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code:  code,
        model: elModelSelect.value,
      }),
    });

    if (!response.ok) {
      // Server merespons dengan status error (4xx / 5xx)
      const errorText = await response.text();
      throw new Error(`Server error ${response.status}: ${errorText}`);
    }

    data = await response.json();

  } catch (err) {
    resetButton();
    // alert(`Gagal menghubungi API.\n${err.message}`);
    showResult({
      verdict:    'human',
      confidence: 100,
      model:      'test',
      durationMs: 0.1,
    });
    return;
  }

  const durationMs = Date.now() - startTime;

  resetButton();

  showResult({
    verdict:    data.verdict,
    confidence: data.confidence,
    model:      data.model,
    durationMs,
  });
}

function resetButton() {
  elBtnAnalyze.disabled = false;
  elBtnAnalyze.textContent = 'Analyze';
}
