from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('upload/', views.upload_view, name='upload'), # 세희 추가
    path('api/predict/', views.api_predict_view, name='api_predict'), # 세희추가
]
