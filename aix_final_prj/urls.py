from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # 🧭 관리자 페이지
    path("admin/", admin.site.urls),

    # 🏠 홈
    path("", TemplateView.as_view(template_name="home.html"), name="home"),

    # ♻️ 재활용 관련 (Recycle App)
    path("recycle/", TemplateView.as_view(template_name="recycle/recyclables.html"), name="recyclables"),
    path("recycle/upload/", TemplateView.as_view(template_name="recycle/upload.html"), name="upload"),
    path("recycle/predict/", TemplateView.as_view(template_name="recycle/predict_list.html"), name="predict_list"),

    # ☕ 커피 분석 (Coffee App)
    # 👉 탭 순서상 첫 번째: 리포트
    path("coffee/coffee_report/", TemplateView.as_view(template_name="coffee/coffee_report.html"), name="coffee_report"),
    # 👉 두 번째: 메인 탭 (딥러닝, 회귀, 분류, 군집 통합)
    path("coffee/", TemplateView.as_view(template_name="coffee/coffee.html"), name="coffee"),
    # 👉 세 번째: BMI 분석
    path("coffee/bmi/", TemplateView.as_view(template_name="coffee/bmi.html"), name="bmi"),
    # 👉 네 번째: 수면 분석
    path("coffee/sleep/", TemplateView.as_view(template_name="coffee/sleep.html"), name="sleep"),
    # 👉 다섯 번째: 스트레스 분석
    path("coffee/stress/", TemplateView.as_view(template_name="coffee/stress.html"), name="stress"),

    # 🧠 PDF-RAG (문서 기반 챗봇)
    path("rag/", TemplateView.as_view(template_name="rag/pdf_rag.html"), name="rag"),

    # 📰 뉴스 / ⚙️ 설정 (기타 페이지)
    path("news/", TemplateView.as_view(template_name="etc/news.html"), name="news"),
    path("settings/", TemplateView.as_view(template_name="etc/settings.html"), name="settings"),

    # 🔧 개발용 (AI API / ML 실험)
    path("dev/api/", include("aix_final_prj.dev.urls1")),  # 예측, 업로드, 코드변경 등
    path("dev/ml/", include("aix_final_prj.dev.urls2")),   # 머신러닝 / 데이터 분석 관련
]

# ✅ 미디어 파일 접근 (업로드 파일 표시)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)