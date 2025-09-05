from django.urls import path
from aix_final_prj.service.news import NewsPage

# Dev1 개발자용
urlpatterns = [
    # http://127.0.0.1:8000/dev/news
    path('news', NewsPage.as_view(), name='urls1'),
]