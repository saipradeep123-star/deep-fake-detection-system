const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const clearFile = document.getElementById('clearFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const frameStrip = document.getElementById('frameStrip');
const verdict = document.getElementById('verdict');
const verdictLabel = document.getElementById('verdictLabel');
const verdictConf = document.getElementById('verdictConf');
const realBar = document.getElementById('realBar');
const fakeBar = document.getElementById('fakeBar');
const realPct = document.getElementById('realPct');
const fakePct = document.getElementById('fakePct');
const metaFrames = document.getElementById('metaFrames');
const metaBlinks = document.getElementById('metaBlinks');
const errorMsg = document.getElementById('errorMsg');

let selectedFile = null;
let scanInterval = null;

dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  })
);
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileInfo.classList.remove('hidden');
  analyzeBtn.disabled = false;
  resetVerdict();
}

clearFile.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  fileInfo.classList.add('hidden');
  analyzeBtn.disabled = true;
  resetVerdict();
  buildIdleStrip();
});

function resetVerdict() {
  verdict.className = 'verdict idle';
  verdictLabel.textContent = '—';
  verdictConf.textContent = '';
  realBar.style.width = '0%';
  fakeBar.style.width = '0%';
  realPct.textContent = '0%';
  fakePct.textContent = '0%';
  metaFrames.textContent = '—';
  metaBlinks.textContent = '—';
  errorMsg.classList.add('hidden');
}

function buildIdleStrip() {
  frameStrip.innerHTML = '<div class="strip-idle">awaiting input</div>';
}

function buildScanningStrip(n = 24) {
  frameStrip.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const cell = document.createElement('div');
    cell.className = 'strip-cell';
    frameStrip.appendChild(cell);
  }
  const cells = frameStrip.querySelectorAll('.strip-cell');
  let i = 0;
  scanInterval = setInterval(() => {
    cells.forEach(c => c.classList.remove('scanning'));
    cells[i % cells.length].classList.add('scanning');
    i++;
  }, 80);
}

function finishStrip(label) {
  clearInterval(scanInterval);
  const cells = frameStrip.querySelectorAll('.strip-cell');
  cells.forEach((c, idx) => {
    c.classList.remove('scanning');
    setTimeout(() => {
      c.classList.add(label === 'FAKE' ? 'fake' : 'real');
    }, idx * 15);
  });
}

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  resetVerdict();
  buildScanningStrip();

  const formData = new FormData();
  formData.append('video', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      clearInterval(scanInterval);
      buildIdleStrip();
      errorMsg.textContent = data.error || 'Something went wrong during analysis.';
      errorMsg.classList.remove('hidden');
      return;
    }

    finishStrip(data.label);
    verdict.className = 'verdict ' + (data.label === 'FAKE' ? 'fake' : 'real');
    verdictLabel.textContent = data.label === 'FAKE' ? 'LIKELY FAKE' : 'LIKELY REAL';
    verdictConf.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

    const realP = (data.real_probability * 100).toFixed(1);
    const fakeP = (data.fake_probability * 100).toFixed(1);
    realBar.style.width = realP + '%';
    fakeBar.style.width = fakeP + '%';
    realPct.textContent = realP + '%';
    fakePct.textContent = fakeP + '%';

    metaFrames.textContent = data.num_frames_analyzed;
    metaBlinks.textContent = data.blink_count;
  } catch (err) {
    clearInterval(scanInterval);
    buildIdleStrip();
    errorMsg.textContent = 'Could not reach the analysis server.';
    errorMsg.classList.remove('hidden');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Run analysis';
  }
});

buildIdleStrip();
