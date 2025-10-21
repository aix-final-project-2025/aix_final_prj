// =============================
// ⚙️ Global Sidebar Script (최종 안정 버전)
// =============================

document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.getElementById("mySidebar");
  const toggleButton = document.querySelector(".togglebtn");

  if (!sidebar || !toggleButton) {
    console.warn("⚠️ Sidebar or toggle button not found in DOM");
    return;
  }

  // 🌈 토글 동작 함수
  function toggleNav() {
    const isExpanded = sidebar.classList.toggle("expanded");
    toggleButton.textContent = isExpanded ? "✕" : "☰";
  }

  // 🧩 햄버거 버튼 클릭 시 토글
  toggleButton.addEventListener("click", toggleNav);

  // 🔒 사이드바 내 링크 클릭 시 자동 닫기
  sidebar.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", () => {
      if (sidebar.classList.contains("expanded")) {
        sidebar.classList.remove("expanded");
        toggleButton.textContent = "☰";
      }
    });
  });

  // 🧠 ESC 키 눌렀을 때 닫기 (UX 향상)
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && sidebar.classList.contains("expanded")) {
      sidebar.classList.remove("expanded");
      toggleButton.textContent = "☰";
    }
  });

  // ✅ 초기 확인용 로그 (개발 중만 사용)
  console.log("✅ base.js fully loaded and sidebar initialized");
});