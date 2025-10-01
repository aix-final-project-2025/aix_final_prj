from django.shortcuts import render
from django.views.generic import ListView, DetailView
def home(request):
    return render(request, 'core/home.html')

def about(request):
    return render(request, 'core/about.html')

def contact(request):
    return render(request, 'core/contact.html')

class ArticleListView(ListView):
    #model = Article/
    template_name = 'article_list.html'
    context_object_name = 'articles'

class ArticleDetailView(DetailView):
    #model = Article
    template_name = 'article_detail.html'
    context_object_name = 'article'