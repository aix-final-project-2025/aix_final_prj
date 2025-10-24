console.log("♻️ recycle_upload.js loaded!");

// ==============================
// 공통 변수
// ==============================
const dropArea = document.getElementById("drop-area");
const fileInputAlbum = document.getElementById("fileElemAlbum");
const fileInputCamera = document.getElementById("fileElemCamera");
const dropMessage = document.getElementById("drop-message");
const loaderOverlay = document.getElementById("loaderOverlay");
const modal = document.getElementById("mobileModal");
const overlay = document.getElementById("modalOverlay");
const modalHandle = document.getElementById("modalHandle");
const modalContent = document.getElementById("modalContent");
const cameraButton = document.getElementById("cameraButton");
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

// ==============================
// 로더 컨트롤
// ==============================
function showLoader() { loaderOverlay.style.display = "flex"; }
function hideLoader() { loaderOverlay.style.display = "none"; }

// ==============================
// 파일 선택 및 미리보기
// ==============================
dropArea.addEventListener("click", () => fileInputAlbum.click());
dropArea.addEventListener("dragover", e => { e.preventDefault(); dropArea.classList.add("dragover"); });
dropArea.addEventListener("dragleave", () => dropArea.classList.remove("dragover"));
dropArea.addEventListener("drop", e => {
  e.preventDefault();
  dropArea.classList.remove("dragover");
  const files = e.dataTransfer.files;
  showPreview(files[0]);
  fileInputAlbum.files = files;
});

fileInputAlbum.addEventListener("change", () => {
  if(fileInputAlbum.files[0]) showPreview(fileInputAlbum.files[0]);
});
if(!isMobile) cameraButton.textContent = "파일선택";
cameraButton.addEventListener("click", () => fileInputAlbum.click());

function showPreview(file){
  const reader = new FileReader();
  reader.onload = (e) => {
    dropMessage.style.display = "none";
    let existingImg = dropArea.querySelector("img");
    if(existingImg) existingImg.remove();
    const img = document.createElement("img");
    img.src = e.target.result;
    dropArea.appendChild(img);
  }
  reader.readAsDataURL(file);
}

// ==============================
// 예측 요청
// ==============================
document.getElementById("uploadForm").addEventListener("submit", async (e)=>{
  e.preventDefault();
  const fileToUpload = fileInputAlbum.files[0];
  if (!fileToUpload) return alert("이미지를 선택하거나 드래그하여 넣어주세요.");
  showLoader();

  const formData = new FormData(e.target);
  try {
    const res = await fetch("/api/predict/", {
      method: "POST",
      headers: { "X-CSRFToken": e.target.querySelector("[name=csrfmiddlewaretoken]").value },
      body: formData
    });
    const data = await res.json();
    renderResult(data);
    openModal(data);
  } catch (err) {
    alert("서버 오류가 발생했습니다.");
  } finally {
    hideLoader();
  }
});

// ==============================
// 모달 컨트롤
// ==============================
function openModal(data) {
  modal.style.display = "flex";
  overlay.style.display = "block";
  document.body.classList.add("modal-open");
  modal.classList.add("slide-up");
  if (data.tts_able == 1) new Audio(data.tts_url).play();
}

function closeModal() {
  modal.classList.remove("slide-up");
  modal.classList.add("slide-down");
  modal.addEventListener("animationend", () => {
    modal.style.display = "none";
    overlay.style.display = "none";
    modal.classList.remove("slide-down");
    document.body.classList.remove("modal-open");
  }, { once: true });
}

overlay.addEventListener("click", closeModal);
modalHandle.addEventListener("click", closeModal);

// ==============================
// 결과 렌더링
// ==============================
function renderResult(data){
  let html = `<div class="result-container">`;
  if(data.image_data_uri){
    html += `
      <div class="result-card-img"><img src="${data.image_data_uri}"></div>
      <div class="result-right">
        <div class="processing-card">
          <h3>📌 분리처리 안내</h3>
          <p>${data.result_message}</p>
        </div>
        <div class="result-card">
          <h3>Top 3 예측결과</h3>
          <ul>${data.top_3.map(item=>`<li>${item[0]} — ${(item[1]*100).toFixed(2)}%</li>`).join('')}</ul>
        </div>`;
    if(data.recycling_guide){
      html += `
        <div class="guide-card">
          <h3>♻️ 분리수거 가이드</h3>
          <p><strong>카테고리:</strong> ${data.recycling_guide.category}</p>
          <p><strong>방법:</strong> ${data.recycling_guide.action}</p>
        </div>`;
    }
    html += `</div>`; // .result-right 닫기
  }
  html += `</div>`; // .result-container 닫기
  modalContent.innerHTML = html;
}