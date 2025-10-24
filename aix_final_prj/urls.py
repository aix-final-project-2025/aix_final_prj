from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core import views # 세희추가
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),

    path("", TemplateView.as_view(template_name="home.html"), name="home"),
    # path('', include('core.urls')),  # core 앱의 URL 포함
    path("dev/", include("aix_final_prj.dev.urls1")),  # 개발자 1
    path("dev/", include("aix_final_prj.dev.urls2")),  # 개발자 2
    # path('', include('core.urls')), # 세희
    # path('recyclables/', views.recyclables_view, name='recyclables'), # 세희
    # path("settings_with_nav/", TemplateView.as_view(template_name="settings_with_nav.html"), name="settings_with_nav"),
    # path("dev/api/settings/", TemplateView.as_view(template_name="settings_api_with_nav.html"), name="settings_api_with_nav"), # ⚙️ 추가 — API 기반 페이지에 메뉴 포함
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)