const STORAGE_KEY = 'settings_v0';
const toggleEl = document.getElementById('toggle');
const toggleStatus = document.getElementById('toggle-status');
const segmented = document.querySelectorAll('.segmented button');
const segmented1 = document.querySelectorAll('.segmented1 button');
const saveBtn = document.getElementById('save-btn');
const resetBtn = document.getElementById('reset-btn');

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : { active: false, country: 'ko', gender: '2' };
  } catch {
    return { active: false, country: 'ko', gender: '2' };
  }
}

function saveSettings(s) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

function updateToggleUI(enabled) {
  if (enabled) {
    toggleEl.classList.add('on');
    toggleEl.setAttribute('aria-checked', 'true');
    toggleStatus.textContent = '활성';
  } else {
    toggleEl.classList.remove('on');
    toggleEl.setAttribute('aria-checked', 'false');
    toggleStatus.textContent = '비활성';
  }
  settings.active = !!enabled;
}

toggleEl.addEventListener('click', () => updateToggleUI(!settings.active));

function setCountryUI(country) {
  segmented.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.country === country);
  });
  settings.country = country;
}

function setGenderUI(gender) {
  segmented1.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.gender === gender);
  });
  settings.gender = gender;
}

segmented.forEach(btn => btn.addEventListener('click', () => setCountryUI(btn.dataset.country)));
segmented1.forEach(btn => btn.addEventListener('click', () => setGenderUI(btn.dataset.gender)));

saveBtn.addEventListener('click', () => {
  saveSettings(settings);
  saveBtn.textContent = '저장됨 ✓';
  setTimeout(() => (saveBtn.textContent = '저장'), 1200);
});

resetBtn.addEventListener('click', () => {
  settings = { active: false, country: 'ko', gender: '2' };
  updateToggleUI(settings.active);
  setCountryUI(settings.country);
  setGenderUI(settings.gender);
  saveSettings(settings);
});

let settings = loadSettings();
updateToggleUI(settings.active);
setCountryUI(settings.country);
setGenderUI(settings.gender);
