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

---



---

# 📌 AiX Final Project — 협업을 위한 Django 기본 규칙 안내

팀원 여러분 반갑습니다 🙌
이번 프로젝트는 **여러 명이 함께 개발**하는 협업이므로,
작업 초기에 **폴더 구조와 네임스페이스 규칙**을 정해두는 것이 중요합니다.

👉 조금 번거롭더라도, 장기적으로는 **충돌 없는 안정적인 협업**을 위해 꼭 필요한 방법입니다.

## 1. 네임스페이스란?

- **이름 충돌을 피하기 위해 영역을 구분하는 방법**  
- 같은 이름의 파일이나 URL이 여러 곳에 있어도, 네임스페이스를 붙이면 안전하게 호출 가능  

예시:  
- `core/templates/core/home.html` → `render(request, "core/home.html")`  
- `users/templates/users/home.html` → `render(request, "users/home.html")`  

✅ **전역 템플릿(`templates/base.html`)은 경로 없이 파일명 `"base.html"`만 호출하면 됩니다.**  

---

## 2. 경로가 왜 이렇게 길어야 할까?

예: `core/templates/core/home.html`

- **앞의 `core/templates/`**  
  → Django가 `core` 앱의 템플릿을 탐색하는 위치  

- **뒤의 `core/home.html`**  
  → 호출할 때 구분자 역할 (앱 이름 네임스페이스)  

👉 표면적으로는 `core`가 두 번 반복돼 보이지만,  
하나라도 빠지면 Django는 **어느 앱의 home.html인지 구분하지 못해 충돌**이 발생합니다.  

즉,  
- `templates/base.html` → 호출: `"base.html"` (전역은 경로 필요 없음)  
- `core/templates/core/home.html` → 호출: `"core/home.html"` (앱 전용은 앱 이름/파일명 구조 필수)  

---

## 3. URL 네임스페이스 (미리 알아두기)

- 각 앱의 `urls.py`에 `app_name = "core"` 지정  
- 뷰 이름 지정 후:  
  ```python
  path("", views.home, name="home")
````

* 템플릿에서 호출:

  ```html
  <a href="{% url 'core:home' %}">홈</a>
  ```

👉 URL도 템플릿과 동일한 원리로 **“앱이름:라우트이름”** 구조를 씁니다.

---

## 4. 정적 파일 (Static files)

### Django 설정

```python
STATIC_URL = '/static/'

STATICFILES_DIRS = [
    BASE_DIR / "static",          # 전역 static 폴더
    BASE_DIR / "core" / "static", # 앱별 static 폴더
]
```

* **STATIC\_URL**: 브라우저에서 접근할 때 쓰는 접두사 (`/static/...`)
* **STATICFILES\_DIRS**: Django가 실제 파일을 찾는 경로 (위에서 아래 순서대로 탐색)

### 협업 전략

* **전역 static**: 공용 리소스 (예: `global.css`, `main.js`)
* **앱별 static**: 각 앱 전용 리소스 (예: `core_style.css`, `crawling_chart.js`)

---

## 5. 구조 예시

```
aix_final_prj/
 ├─ templates/                # 전역 공통 템플릿
 │   └─ base.html             # 호출 시 → "base.html"
 ├─ core/
 │   ├─ templates/core/       # core 앱 전용 템플릿
 │   │   ├─ home.html         # 호출 시 → "core/home.html"
 │   │   ├─ about.html        # 호출 시 → "core/about.html"
 │   │   └─ contact.html      # 호출 시 → "core/contact.html"
 │   └─ static/core/          # core 앱 전용 정적 파일
 │       ├─ css/core_style.css
 │       └─ js/core_script.js
 ├─ crawling/
 │   ├─ templates/crawling/   # crawling 앱 전용 템플릿
 │   │   └─ list.html         # 호출 시 → "crawling/list.html"
 │   └─ static/crawling/
 └─ ...
```

---

## 6. 장점

* **명확성**

  * 앱 이름이 경로에 포함 → 파일이 어느 앱 소속인지 바로 알 수 있음
* **협업 편의성**

  * 충돌 최소화, 각자 맡은 앱 내부에서만 작업 가능
* **실무 표준**

  * Django 공식 권장 방식, 새 팀원이 와도 구조만 보고 이해 가능

---

## ✅ 결론

* **HTML (템플릿)**

  * 앱 전용: `"앱이름/파일명.html"`
  * 전역: `"base.html"`

* **URL**

  * `"앱이름:라우트이름"`

* **정적 파일**

  * 전역 → 앱별 순서로 탐색

👉 **경로가 다소 길어도 충돌 없는 안정적인 협업을 위해 필요한 규칙**입니다.

---

🙌 이 규칙만 지켜도 **팀 프로젝트가 훨씬 수월해집니다!**
처음엔 불편해 보여도, 나중에는 “이 구조 덕분에 살았다!” 하는 순간이 올 거예요 😉

```




