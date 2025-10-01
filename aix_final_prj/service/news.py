from django.shortcuts import render
from django.views import View
from django.views.generic import ListView, DetailView

class NewsPage(View):
    template_name = "news.html"

    def get(self, request):
        context = {"title": "Dev1", "message": "클래스 기반 뷰"}
        return render(request, self.template_name, context)
    
#from django.db import models

#class Article(models.Model):
#    title = models.CharField(max_length=200, verbose_name="기사 제목")
#    content = models.TextField(verbose_name="본문 내용")
#    created_at = models.DateTimeField(auto_now_add=True)

#    def __str__(self):
#        return self.title


