let categories = [];
let page = 1;
let selectedCategoryId = null;
let loading = false;

function renderCard(item){
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.id = item.id;

  const img = document.createElement('img');
  img.src = item.image_url;
  card.appendChild(img);

  const content = document.createElement('div');
  content.className = 'card-content';
  content.innerHTML = `
    <div class="content-box"><strong>예측결과:</strong> ${item.predicted_result}</div>
    <div class="content-box mobile-hide"><strong>분리처리 안내:</strong> ${item.recycling_guide}</div>
    <div class="content-box mobile-hide"><strong>Top 3:</strong> ${item.top3}</div>
    <div class="card-row mobile-hide">
      <select id='predicted_class_${item.id}'>
        ${categories.map(c=>`<option value="${c.id}" ${c.id===item.group_code_id?'selected':''}>${c.name}</option>`).join('')}
      </select>
      <button class="update-btn" onclick="predicChange(${item.id})">수정</button>
    </div>
  `;
  card.appendChild(content);
  return card;
}

async function codeLoad(){
  try {
    const res = await fetch("/dev/api/api/code/");
    const data = await res.json();
    categories = data.codelist;
  } catch(err){ console.error(err); }
}

async function predicChange(itemId){
  try {
    const res = await fetch(`/dev/api/api/class_change/?id=${itemId}&group_code_id=${selectedCategoryId}`);
    const data = await res.json();
    if(!data.success) alert("수정 실패");
    else {
      alert("수정 성공");
      closeModal();
    }
  } catch(err){ console.error(err); }
}

async function loadData(){
  const container = document.getElementById('waste-container');
  const emptyMessage = document.getElementById('empty-message');
  if(loading) return;
  loading = true;
  try {
    const res = await fetch(`/dev/api/api/predict_list_page/?page=${page}`);
    const data = await res.json();
    if(page === 1 && data.items.length === 0) emptyMessage.style.display = 'block';
    else {
      emptyMessage.style.display = 'none';
      data.items.forEach(item => container.appendChild(renderCard(item)));
      page++;
    }
    if(!data.has_more) window.removeEventListener('scroll', scrollHandler);
  } catch(err){ console.error(err); }
  loading = false;
}

function scrollHandler(){
  if((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 300) loadData();
}
window.addEventListener('scroll', scrollHandler);

function openModal(card){
  if(window.innerWidth >= 768) return;
  const imgSrc = card.querySelector('img').src;
  const predicted = card.querySelector('.content-box strong').nextSibling.textContent;
  const guide = card.querySelectorAll('.content-box')[1].textContent;
  const top3 = card.querySelectorAll('.content-box')[2].textContent;
  const categoryId = parseInt(card.querySelector('select')?.value || 0);
  selectedCategoryId = categoryId;

  document.getElementById('modal-image').src = imgSrc;
  document.getElementById('modal-predicted').innerText = predicted;
  document.getElementById('modal-guide').innerText = guide;
  document.getElementById('modal-top3').innerText = top3;

  const buttonGroup = document.getElementById('modal-category-buttons');
  buttonGroup.innerHTML = '';
  categories.forEach(c => {
    const btn = document.createElement('button');
    btn.innerText = c.name;
    btn.dataset.codeId = c.id;
    btn.className = (c.id === categoryId ? 'active' : '');
    btn.addEventListener('click', ()=>{
      selectedCategoryId = c.id;
      Array.from(buttonGroup.children).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
    buttonGroup.appendChild(btn);
  });

  document.getElementById('modal-sheet').classList.add('show');
  document.body.style.overflow = 'hidden';
  document.body.classList.add('modal-open');
}

function closeModal(){
  const modal = document.getElementById('modal-sheet');
  modal.classList.remove('show');
  document.body.style.overflow = '';
  document.body.classList.remove('modal-open');
}

document.getElementById('waste-container').addEventListener('click', e=>{
  const card = e.target.closest('.card');
  if(card) openModal(card);
});

document.querySelector('.modal-close').addEventListener('click', closeModal);

document.getElementById('modal-update-btn').addEventListener('click', ()=>{
  if(selectedCategoryId === null) {
    alert("항목을 선택해주세요.");
    return;
  }
  predicChange(selectedCategoryId);
});

async function init(){
  await codeLoad();
  loadData();
}
init();

// ♻️ recyclables.html 전용 스크립트
document.addEventListener('DOMContentLoaded', () => {
  const albumGrid = document.querySelector('.album-grid');

  albumGrid.addEventListener('click', (event) => {
    const target = event.target.closest('.dropdown-toggle');

    // 1️⃣ 메뉴 열기
    if (target) {
      event.preventDefault();
      const menuContent = target.closest('.card-dropdown-menu').querySelector('.menu-content');
      document.querySelectorAll('.menu-content').forEach(menu => {
        if (menu !== menuContent) menu.classList.add('hidden');
      });
      menuContent.classList.toggle('hidden');
      return;
    }

    // 2️⃣ 삭제 클릭
    const deleteTarget = event.target.closest('.delete-action');
    if (deleteTarget) {
      event.preventDefault();
      const itemId = deleteTarget.dataset.id;
      deleteTarget.closest('.menu-content').classList.add('hidden');

      if (confirm(`항목 ID ${itemId}를 삭제하시겠습니까?`)) {
        // TODO: Django DELETE 요청 구현 예정
        alert(`항목 ID ${itemId} 삭제 로직 실행됨 (미구현 상태)`);
      }
    }

    // 3️⃣ 외부 클릭 시 메뉴 닫기
    if (!event.target.closest('.card-dropdown-menu')) {
      document.querySelectorAll('.menu-content').forEach(menu => menu.classList.add('hidden'));
    }
  });
});