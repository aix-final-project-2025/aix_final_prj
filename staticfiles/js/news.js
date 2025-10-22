// 📰 BBC Asia 뉴스 RSS 불러오기
async function loadNews() {
  const newsList = document.getElementById("news-list");
  newsList.innerHTML = "<li>BBC Asia 최신 뉴스를 불러오는 중...</li>";

  try {
    const response = await fetch(
      "https://api.rss2json.com/v1/api.json?rss_url=https://feeds.bbci.co.uk/news/world/asia/rss.xml"
    );
    const data = await response.json();

    if (!data.items) {
      newsList.innerHTML = "<li>뉴스를 불러오지 못했습니다 😢</li>";
      return;
    }

    newsList.innerHTML = data.items
      .slice(0, 10)
      .map(
        (item) => `
        <li>
          <a href="${item.link}" target="_blank" rel="noopener noreferrer">${item.title}</a>
          <p>${item.description.slice(0, 200)}...</p>
        </li>`
      )
      .join("");
  } catch (error) {
    newsList.innerHTML = "<li>❌ 데이터를 가져오는 중 오류가 발생했습니다.</li>";
    console.error(error);
  }
}

// 페이지 로드시 자동 실행
document.addEventListener("DOMContentLoaded", loadNews);