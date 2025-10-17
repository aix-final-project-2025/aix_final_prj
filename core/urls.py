from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_view, name='upload'), # 세희 추가
    path('recyclables/', views.recyclables_view, name='recyclables'), # 세희 추가
    path('coffee/', views.coffee_view, name='coffee'), # 세희 추가
    path('api/predict/', views.api_predict_view, name='api_predict'), # 세희추가
]
