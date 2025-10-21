// ================================
// ☕ BMI Analysis Script
// ================================

const REGRESSION_URL = "/coffee/api/run_regression_bmi/";
const CLASSIFICATION_URL = "/coffee/api/run_classification_bmi/";
const CLUSTERING_URL = "/coffee/api/run_clustering_bmi/";
const DL_PREDICT_URL = "/coffee/api/predict_dl_bmi/";

function showLoader(id) {
  document.getElementById(id).style.display = 'block';
}
function hideLoader(id) {
  document.getElementById(id).style.display = 'none';
}
function showResults(id, content) {
  document.getElementById(id).innerHTML = content;
}

// --- 회귀 ---
document.getElementById('run-regression')?.addEventListener('click', () => {
  showLoader('regression-loader');
  fetch(REGRESSION_URL, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      let html = '<h3>회귀 모델 성능 비교</h3><table><tr><th>Model</th><th>RMSE</th><th>R²</th></tr>';
      for (const m in data.results) {
        const r2 = data.results[m];
        html += `<tr><td>${m}</td><td>${r2.RMSE}</td><td>${r2.R2}</td></tr>`;
      }
      html += '</table>';
      showResults('regression-results', html);
    })
    .finally(() => hideLoader('regression-loader'));
});

// --- 분류 ---
document.getElementById('run-classification')?.addEventListener('click', () => {
  showLoader('classification-loader');
  fetch(CLASSIFICATION_URL, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      let html = '<h3>분류 모델 성능 비교</h3><table><tr><th>Model</th><th>Accuracy</th><th>F1</th></tr>';
      for (const m in data.results) {
        const res = data.results[m];
        html += `<tr><td>${m}</td><td>${res.accuracy}</td><td>${res.f1_score}</td></tr>`;
      }
      html += '</table>';
      showResults('classification-results', html);
    })
    .finally(() => hideLoader('classification-loader'));
});

// --- 군집 ---
document.getElementById('run-clustering')?.addEventListener('click', () => {
  showLoader('clustering-loader');
  fetch(CLUSTERING_URL, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      let html = `<h3>군집 결과 (K=${data.optimal_k})</h3>`;
      html += `<img src="/static/${data.plot_filename}" style="max-width:100%;border-radius:5px;">`;
      showResults('clustering-results', html);
    })
    .finally(() => hideLoader('clustering-loader'));
});

// --- 딥러닝 예측 ---
document.getElementById('dl-form')?.addEventListener('submit', e => {
  e.preventDefault();
  showLoader('dl-loader');
  const d = {
    Age: +age.value,
    Country: +country.value,
    Coffee_Intake: +coffee_intake.value,
    Sleep_Hours: +sleep_hours.value,
    Physical_Activity_Hours: +activity.value,
    Smoking: +smoking.value,
    Alcohol_Consumption: +alcohol.value,
    Gender_Male: +gender.value,
    Occupation: +occupation.value
  };
  fetch(DL_PREDICT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(d)
  })
    .then(r => r.json())
    .then(data => {
      let html = `<h3>예측 결과</h3><p><strong>예상 BMI: ${data.predicted_class}</strong></p>`;
      html += '<table><tr><th>카테고리</th><th>확률</th></tr>';
      for (const c in data.probabilities) {
        html += `<tr><td>${c}</td><td>${data.probabilities[c]}</td></tr>`;
      }
      html += '</table>';
      showResults('dl-results', html);
    })
    .finally(() => hideLoader('dl-loader'));
});

console.log("✅ coffee.js loaded successfully");

// ☕ Coffee Analysis Script
console.log("✅ coffee.js loaded successfully");

const REGRESSION_URL = "/coffee/api/run_regression_cf/";
const CLASSIFICATION_URL = "/coffee/api/run_classification_cf/";
const CLUSTERING_URL = "/coffee/api/run_clustering_cf/";
const DL_PREDICT_URL = "/coffee/api/predict_dl_cf/";

function showLoader(id) { document.getElementById(id).style.display = 'block'; }
function hideLoader(id) { document.getElementById(id).style.display = 'none'; }
function showResults(id, html) { document.getElementById(id).innerHTML = html; }

// 1️⃣ 회귀 모델
document.getElementById('run-regression')?.addEventListener('click', () => {
  showLoader('regression-loader');
  fetch(REGRESSION_URL, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      let html = '<h3>회귀 모델 성능 비교</h3><table><tr><th>Model</th><th>RMSE</th><th>R²</th></tr>';
      for (const m in data.results) {
        const r2 = data.results[m];
        html += `<tr><td>${m}</td><td>${r2.RMSE}</td><td>${r2.R2}</td></tr>`;
      }
      html += '</table>';
      showResults('regression-results', html);
    })
    .finally(() => hideLoader('regression-loader'));
});

// 2️⃣ 분류 모델
document.getElementById('run-classification')?.addEventListener('click', () => {
  showLoader('classification-loader');
  fetch(CLASSIFICATION_URL, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      let html = '<h3>분류 모델 성능 비교</h3><table><tr><th>Model</th><th>Accuracy</th><th>F1</th></tr>';
      for (const m in data.results) {
        const res = data.results[m];
        html += `<tr><td>${m}</td><td>${res.accuracy}</td><td>${res.f1_score}</td></tr>`;
      }
      html += '</table>';
      showResults('classification-results', html);
    })
    .finally(() => hideLoader('classification-loader'));
});

// 3️⃣ 군집 분석
document.getElementById('run-clustering')?.addEventListener('click', () => {
  showLoader('clustering-loader');
  fetch(CLUSTERING_URL, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      let html = `<h3>군집 결과 (K=${data.optimal_k})</h3>`;
      html += `<img src="/static/${data.plot_filename}" style="max-width:100%;border-radius:5px;">`;
      showResults('clustering-results', html);
    })
    .finally(() => hideLoader('clustering-loader'));
});

// 4️⃣ 딥러닝 예측
document.getElementById('dl-form')?.addEventListener('submit', e => {
  e.preventDefault();
  showLoader('dl-loader');
  const d = {
    Age: +age.value,
    Country: +country.value,
    Sleep_Hours: +sleep_hours.value,
    BMI: +bmi.value,
    Gender_Male: +gender.value,
    Physical_Activity_Hours: +activity.value,
    Smoking: +smoking.value,
    Alcohol_Consumption: +alcohol.value,
    Stress_Level: +stress.value,
    Occupation: +occupation.value
  };
  fetch(DL_PREDICT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(d)
  })
    .then(r => r.json())
    .then(data => {
      let html = `<h3>예측 결과</h3><p><strong>예상 커피 섭취 카테고리: ${data.predicted_class}</strong></p>`;
      html += '<table><tr><th>카테고리</th><th>확률</th></tr>';
      for (const c in data.probabilities) {
        html += `<tr><td>${c}</td><td>${data.probabilities[c]}</td></tr>`;
      }
      html += '</table>';
      showResults('dl-results', html);
    })
    .finally(() => hideLoader('dl-loader'));
});

// ☕ 커피·BMI·수면·스트레스 페이지 공용 스크립트

function showLoader(id){ document.getElementById(id).style.display = 'block'; }
function hideLoader(id){ document.getElementById(id).style.display = 'none'; }
function showResults(id, html){ document.getElementById(id).innerHTML = html; }

// 회귀
function runRegression(url){
  showLoader('regression-loader');
  fetch(url,{method:'POST'})
  .then(r=>r.json())
  .then(d=>{
    let html='<h3>회귀 모델 성능 비교</h3><table><tr><th>Model</th><th>RMSE</th><th>R²</th></tr>';
    for(const m in d.results){
      html+=`<tr><td>${m}</td><td>${d.results[m].RMSE}</td><td>${d.results[m].R2}</td></tr>`;
    }
    html+='</table>';
    showResults('regression-results',html);
  }).finally(()=>hideLoader('regression-loader'));
}

// 분류
function runClassification(url){
  showLoader('classification-loader');
  fetch(url,{method:'POST'})
  .then(r=>r.json())
  .then(d=>{
    let html='<h3>분류 모델 성능 비교</h3><table><tr><th>Model</th><th>Accuracy</th><th>F1</th></tr>';
    for(const m in d.results){
      html+=`<tr><td>${m}</td><td>${d.results[m].accuracy}</td><td>${d.results[m].f1_score}</td></tr>`;
    }
    html+='</table>';
    showResults('classification-results',html);
  }).finally(()=>hideLoader('classification-loader'));
}

// 군집
function runClustering(url){
  showLoader('clustering-loader');
  fetch(url,{method:'POST'})
  .then(r=>r.json())
  .then(d=>{
    let html=`<h3>군집 분석 결과 (K=${d.optimal_k})</h3>`;
    html+=`<img src="/static/${d.plot_filename}" style="max-width:100%;border-radius:5px;">`;
    showResults('clustering-results',html);
  }).finally(()=>hideLoader('clustering-loader'));
}