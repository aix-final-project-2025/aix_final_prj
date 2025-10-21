// 간단한 드래그 인터랙션
document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById('drop-area');
  const fileElem = document.getElementById('fileElem');
  const dropMsg = document.getElementById('drop-message');

  dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('dragover');
    dropMsg.textContent = "이곳에 파일을 놓으세요";
  });

  dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
    dropMsg.textContent = "여기에 PDF 파일을 클릭하거나 드래그하세요";
  });

  dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileElem.files = files;
      dropMsg.textContent = `${files.length}개의 파일 선택됨`;
    }
  });
});