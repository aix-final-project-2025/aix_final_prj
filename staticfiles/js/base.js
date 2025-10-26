// =================================================================
// ⚙️ Global Sidebar Script (사이드바 및 메뉴 아코디언 로직)
// - 중복 코드 제거 및 클래스명 통일 (open)
// =================================================================

document.addEventListener("DOMContentLoaded", () => {
    // 🔍 DOM 요소 선택
    const sidebar = document.getElementById("mySidebar");
    const toggleButton = document.querySelector(".togglebtn");
    const allLinks = sidebar ? sidebar.querySelectorAll('a') : []; 
    const menuToggles = document.querySelectorAll(".main-menu-toggle");

    if (!sidebar || !toggleButton) {
        console.warn("⚠️ Sidebar or toggle button not found in DOM.");
        return;
    }

    /**
     * @function updateToggleButton
     * @description 토글 버튼의 텍스트와 상태를 현재 사이드바 상태에 따라 업데이트합니다.
     */
    function updateToggleButton() {
        const isOpen = sidebar.classList.contains('open');
        toggleButton.textContent = isOpen ? "✕" : "☰";
    }

    /**
     * @function toggleSidebar
     * @description 사이드바의 열림/닫힘 상태를 토글합니다.
     */
    function toggleSidebar() {
        sidebar.classList.toggle('open');
        updateToggleButton();
    }
    
    // ===========================================================
    // 1. 사이드바 열기/닫기 로직 (토글 버튼 클릭)
    // ===========================================================
    toggleButton.addEventListener('click', toggleSidebar);

    // ===========================================================
    // 2. 일반 링크 클릭 시 사이드바 닫기 (모바일 UX)
    // ===========================================================
    allLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            // 아코디언 토글 버튼(.main-menu-toggle)은 닫지 않습니다.
            if (e.currentTarget.classList.contains('main-menu-toggle')) {
                return; 
            }
            
            // 사이드바가 열려 있을 경우에만 닫습니다.
            if (sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
                updateToggleButton();
            }
        });
    });

    // ===========================================================
    // 3. 아코디언 서브메뉴 토글 로직
    // ===========================================================
    menuToggles.forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            e.preventDefault(); 
            e.stopPropagation(); 
            
            const targetId = toggle.dataset.target; // base.html의 data-target 속성 사용
            const targetSubmenu = document.getElementById(targetId);

            if (!targetSubmenu) return;

            const isOpened = targetSubmenu.classList.toggle('open'); // active 대신 open 사용
            const arrow = toggle.querySelector('.toggle-arrow');

            if (isOpened) {
                // 열기: max-height를 scrollHeight로 설정하여 부드럽게 열립니다.
                targetSubmenu.style.maxHeight = targetSubmenu.scrollHeight + "px";
                if (arrow) arrow.style.transform = 'rotate(180deg)';
            } else {
                // 닫기
                targetSubmenu.style.maxHeight = '0';
                if (arrow) arrow.style.transform = 'rotate(0deg)';
            }
        });
    });
    
    // ===========================================================
    // 4. ESC 키로 사이드바 닫기 (접근성 및 UX 향상)
    // ===========================================================
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && sidebar.classList.contains("open")) {
            sidebar.classList.remove('open');
            updateToggleButton();
        }
    });

    // 5. 초기 상태 설정 (미디어 쿼리가 처리하므로, 여기서는 토글 버튼 텍스트만 초기화)
    updateToggleButton(); 
});
