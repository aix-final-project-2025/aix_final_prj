from django.urls import path
#from aix_final_prj.service.news_views import NewsPage
from aix_final_prj.service.recycleable_views import UploadView, PredictApiView
from django.http import JsonResponse

def dummy_wellknown(_):
    return JsonResponse({}, status=200)
# Dev1 개발자용
urlpatterns = [
    # http://127.0.0.1:8000/dev/news
    # path('news', NewsPage.as_view(), name='urls1'),
    path('api/predict/', PredictApiView.as_view(), name='api_predict'),
    path('api/upload/', UploadView.as_view(), name='upload'),
    path('.well-known/<path:subpath>', dummy_wellknown),
]