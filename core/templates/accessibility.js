// 동적 알림 업데이트용 live region
function announce(message) {
    let liveRegion = document.getElementById('aria-live-region');
    if (!liveRegion) {
        liveRegion = document.createElement('div');
        liveRegion.id = 'aria-live-region';
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.style.position = 'absolute';
        liveRegion.style.left = '-9999px';
        document.body.appendChild(liveRegion);
    }
    liveRegion.textContent = message;
}

// 예시: 페이지 로드 시 안내
window.addEventListener('load', () => {
    announce("페이지 로딩이 완료되었습니다.");
});
