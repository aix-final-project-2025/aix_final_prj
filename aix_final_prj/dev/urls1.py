from django.urls import path
from aix_final_prj.service.news import NewsPage
# from .views import ArticleListView, ArticleDetailView
from core.views import ArticleListView, ArticleDetailView


# Dev1 개발자용
urlpatterns = [
    # http://127.0.0.1:8000/dev/news
    path('news', NewsPage.as_view(), name='urls1'),
    path('', ArticleListView.as_view(), name='article_list'),
    path('article/<int:pk>/', ArticleDetailView.as_view(), name='article_detail'),
]


