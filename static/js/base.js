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

        document.addEventListener('DOMContentLoaded', () => {
            const sidebar = document.getElementById('mySidebar');
            const toggleButton = document.querySelector('.togglebtn');
            // 사이드바 내부의 모든 <a> 링크를 선택합니다.
            const allLinks = sidebar ? sidebar.querySelectorAll('a') : []; 
            const menuToggles = document.querySelectorAll(".main-menu-toggle");

            if (!sidebar || !toggleButton) {
                console.warn("⚠️ Sidebar or toggle button not found in DOM");
                return;
            }

            // 1. 사이드바 열기/닫기 로직 (토글 버튼)
            toggleButton.addEventListener('click', () => {
                // 'open' 클래스로 사이드바 상태를 토글합니다.
                sidebar.classList.toggle('open');
            });

            // 2. 일반 링크 클릭 시 사이드바 닫기 (아코디언 제외)
            allLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    // 아코디언 토글(.main-menu-toggle)일 경우 닫지 않습니다.
                    if (e.currentTarget.classList.contains('main-menu-toggle')) {
                        return; 
                    }
                    
                    // 일반 링크(Brand Button, 서브메뉴 아이템, 기타 메뉴) 클릭 시 사이드바 닫기
                    if (sidebar.classList.contains('open')) {
                        sidebar.classList.remove('open');
                    }
                });
            });

            // 3. 아코디언 서브메뉴 토글 로직
            menuToggles.forEach(toggle => {
                toggle.addEventListener('click', (e) => {
                    e.preventDefault(); 
                    e.stopPropagation(); 
                    
                    const targetId = toggle.dataset.target;
                    const targetSubmenu = document.getElementById(targetId);

                    if (!targetSubmenu) return;

                    const isOpened = targetSubmenu.classList.contains('active');
                    const arrow = toggle.querySelector('.toggle-arrow');

                    if (isOpened) {
                        // 닫기
                        targetSubmenu.classList.remove('active');
                        targetSubmenu.style.maxHeight = '0';
                        arrow.style.transform = 'rotate(0deg)';
                    } else {
                        // 열기
                        targetSubmenu.classList.add('active');
                        targetSubmenu.style.maxHeight = targetSubmenu.scrollHeight + "px";
                        arrow.style.transform = 'rotate(180deg)';
                    }
                });
            });
            
            // 4. ESC 키로 사이드바 닫기
            document.addEventListener("keydown", (event) => {
                if (event.key === "Escape" && sidebar.classList.contains("open")) {
                    sidebar.classList.remove('open');
                }
            });

            console.log("✅ Sidebar Script fully loaded and initialized.");
        });