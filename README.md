# 📌 AiX Final Project (Django)

## 📖 프로젝트 개요
이 레포는 **Django 입문 & 최종 프로젝트 준비**용입니다.  
누구나 쉽게 실행할 수 있도록 기본 뼈대만 세팅했습니다.  

팀 이름(예시): **AiX Final Project**

---

## ⚙️ 프로젝트 구조
```

aix\_final\_prj/
├── aix\_final\_prj/       # 프로젝트 설정
│   └── urls.py          # 전역 URL 관리
├── core/                # 앱
│   ├── views.py         # 화면 처리
│   ├── urls.py          # core 앱 URL
│   └── templates/core/  # core 앱 템플릿
│       └── home.html
├── static/              # 전역 CSS/JS/이미지
│   ├── css/global.css
│   ├── js/global.js
└── templates/           # 전역 템플릿
└── base.html

````

---

## ▶ 실행 방법

1. **레포 클론**
```bash
git clone https://github.com/kopynara/aix_final_prj.git
cd aix_final_prj
````

2. **가상환경 만들기 & 실행**

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

3. **Django 설치**

```bash
pip install django
```

4. **서버 실행**

```bash
python manage.py runserver
```

5. **브라우저에서 확인**

```
http://127.0.0.1:8000/
```

---

## 📝 Git 사용 (간단 버전)

* **작업 저장**

```bash
git add .
git commit -m "메시지"
```

* **GitHub(dev 브랜치)로 올리기**

```bash
git push origin dev
```

---

## 🌿 Git 브랜치 규칙

* **main**

  * 깔끔한 기준 브랜치 (팀 확정 전까지는 비워둠)

* **dev**

  * 모든 개발 작업은 여기서 진행
  * GitHub에 올릴 때도 `dev`만 push

👉 정리: **작업은 dev, main은 나중에**

---

## 🖼️ Django 요청 흐름 (아이콘 버전)

```
🌐 사용자 요청 (브라우저)
          ↓
🛣️  aix_final_prj/urls.py   → 프로젝트 전체 URL 총괄
          ↓
📂 core/urls.py              → 앱별 URL 담당
          ↓
🖥️ views.py                  → 화면/데이터 처리
          ↓
📄 templates/core/home.html  → HTML 렌더링
```

👉 흐름: **🌐 → 🛣️ → 📂 → 🖥️ → 📄**
