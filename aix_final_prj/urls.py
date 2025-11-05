from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", TemplateView.as_view(template_name="home.html"), name="home"),
    path("dev1/", include(("aix_final_prj.dev.urls1","dev1") , namespace="dev1")),  # 개발자 1
    path("dev/", include("aix_final_prj.dev.urls2")),  # 개발자 2
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)