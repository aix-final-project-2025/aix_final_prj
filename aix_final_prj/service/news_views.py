from django.shortcuts import render
from django.views import View

class NewsPage(View):
    template_name = "news.html"

    def get(self, request):
        context = {"title": "Dev1", "message": "클래스 기반 뷰"}
        return render(request, self.template_name, context)