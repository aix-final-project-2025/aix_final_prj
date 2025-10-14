from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.views.generic import ListView, DetailView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import RecyclingItem  # 세희

def home(request):
    return render(request, 'core/home.html')

def about(request):
    return render(request, 'core/about.html')

def contact(request):
    return render(request, 'core/contact.html')

def upload_view(request):     # 세희
    return render(request, 'upload.html') 

def recyclables_view(request):    # 세희
    return render(request, 'recyclables.html')

class ArticleListView(ListView):
    #model = Article/
    template_name = 'article_list.html'
    context_object_name = 'articles'

class ArticleDetailView(DetailView):
    #model = Article
    template_name = 'article_detail.html'
    context_object_name = 'article'

# 4. API 뷰 함수 (추가하신 내용) 세희
@csrf_exempt
@require_http_methods(["POST"])
def api_predict_view(request):
    if 'image' not in request.FILES:
        return JsonResponse({"error": "이미지 파일이 전송되지 않았습니다."}, status=400)

    uploaded_file = request.FILES['image']
    
def recyclables_view(request):
    # 최신순으로 정렬된 모든 항목을 가져옵니다. (models.py의 ordering에 의해 자동 정렬됨)
    items = RecyclingItem.objects.all() 
    
    context = {
        'recycling_items': items
    }
    return render(request, 'recyclables.html', context)
    
    # 임시 응답 (예시)
    result_data = {
        "result_message": "임시 예측 결과: 플라스틱",
        "confidence": 0.95,
        "top_3": [
            ["플라스틱", 0.95],
            ["캔", 0.03],
            ["종이", 0.01]
        ],
        "recycling_guide": {
            "category": "플라스틱",
            "action": "내용물을 비우고 깨끗하게 헹군 뒤 압착하여 배출"
        }
    }
    
    return JsonResponse(result_data)