from django.urls import path
from django.views import View
from django.shortcuts import render

class TestPage(View):
    template_name = "test.html"

    def get(self, request):
        context = {"title": "Dev1", "message": "클래스 기반 뷰"}
        return render(request, self.template_name, context)

# Dev2 개발자용
urlpatterns = [
    # http://127.0.0.1:8000/dev/test
    path('test', TestPage.as_view(), name='urls2'),
]