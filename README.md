# 📌 AiX Final Project — 협업을 위한 Django 기본 규칙 안내

팀원 여러분 반갑습니다 🙌
이번 프로젝트는 **여러 명이 함께 개발**하는 협업이므로,
작업 초기에 **폴더 구조와 네임스페이스 규칙**을 정해두는 것이 중요합니다.

👉 경로가 다소 길어지는 수고스러움이 있지만, **장기적으로 충돌 없는 안정적인 협업**을 위해 꼭 필요한 방식입니다.

---

## 1. 네임스페이스란?

* **이름 충돌을 피하기 위해 영역을 구분하는 방법**
* 같은 이름의 파일이나 URL이 여러 곳에 있어도, 네임스페이스를 붙이면 안전하게 호출 가능

예시:

* `core/templates/core/home.html` → `render(request, "core/home.html")`
* `users/templates/users/home.html` → `render(request, "users/home.html")`

✅ **전역 템플릿(`templates/base.html`)은 경로 없이 파일명 `"base.html"`만 호출하면 됩니다.**

---

## 2. URL 네임스페이스 (미리 알아두기)

* 각 앱의 `urls.py`에 `app_name = "core"` 지정
* 뷰 이름 지정 후:

  ```python
  path("", views.home, name="home")
  ```

  템플릿에서는 이렇게 사용합니다:

  ```html
  <a href="{% url 'core:home' %}">홈</a>
  ```

👉 원리: **URL도 이름 충돌 방지를 위해 “앱이름:라우트이름” 구조 사용**

---

## 3. 정적 파일 (Static files)

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

👉 전역 → 앱별 순서로 관리하면 충돌을 피할 수 있습니다.

---

## 4. 왜 앱별 분리가 필요할까?

지난번 프로젝트에서는 모든 HTML을 한 폴더에 몰아넣어서

* `login.html` 같은 이름이 겹치고,
* "이게 어느 기능이지?" 혼동되는 경우가 많았습니다.

👉 이번에는 앱별로 `templates` / `static`을 분리해서 관리합니다.

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

  * `core/templates/core/home.html` → 호출: `"core/home.html"`
  * `crawling/templates/crawling/list.html` → 호출: `"crawling/list.html"`
    → 이름 충돌 없음, 소속이 바로 보임

* **협업 편의성**

  * 각자 맡은 앱 안에서만 작업 → 충돌 최소화
  * 공통 레이아웃은 `templates/base.html` → 호출: `"base.html"`

* **실무 표준**

  * Django 공식 권장 패턴
  * 새로운 팀원이 와도 구조만 보면 바로 이해 가능

---

# 📌 왜 `core/templates/core/home.html` 구조여야 할까요?

여러분, Django에서는 템플릿을 이렇게 두 번 `core`가 반복된 경로로 관리합니다.
👉 **초반 개발시에는 경로가 다소 길어져도, 협업 시 이름 충돌 없이 안정적으로 관리할 수 있습니다.**

* `core/templates/` → Django가 탐색하는 위치
* `core/home.html` → 앱 이름 네임스페이스
* 최종 구조: **`core/templates/core/home.html` → 호출: `"core/home.html"`**

---

🙌 이 규칙만 지켜도 **팀 프로젝트가 훨씬 수월해집니다!**
처음엔 불편해 보여도, 나중에는 “와 이 구조 덕분에 살았다!” 하는 순간이 올 거예요 😉

---

### 📌 고정된 Header/Footer와 본문 여백 관리

현재 프로젝트에서는 `header`와 `footer`가 `position: fixed`로 화면 상/하단에 고정되어 있습니다.  
따라서 본문(`main`)은 header/footer 높이만큼 `margin-top` / `margin-bottom`을 주어 겹침을 방지합니다.

👉 협업 시 header/footer의 **높이(padding, font-size 등)가 변경되면**  
`main`의 margin 값도 반드시 같이 수정해주어야 합니다.  
(예: header 높이를 80px로 늘리면 → `main { margin-top: 80px; }` 로 변경)

