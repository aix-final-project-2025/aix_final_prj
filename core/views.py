
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.views.generic import ListView, DetailView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import RecyclingItem  # 세희

def home(request):    #세희
    return render(request, 'home.html')    

def upload_view(request):     # 세희
    return render(request, 'upload.html') 

def recyclables_view(request):    # 세희
    return render(request, 'recyclables.html')

def coffee_view(request):    # 세희
    return render(request, 'coffee.html', {})

class ArticleListView(ListView):
    #model = Article
    template_name = 'article_list.html'
    context_object_name = 'articles'

class ArticleDetailView(DetailView):
    #model = Article
    template_name = 'article_detail.html'
    context_object_name = 'article'

# core/views.py의 수정된 api_predict_view 함수 부분

# 4. API 뷰 함수 (추가하신 내용) 세희
@csrf_exempt
@require_http_methods(["POST"])
def api_predict_view(request):
    # 1. 파일이 없는지 확인
    if 'image' not in request.FILES:
        return JsonResponse({"error": "이미지 파일이 전송되지 않았습니다."}, status=400)

    uploaded_file = request.FILES['image']
    

    result_data = {
        "result_message": "임시 예측 결과: 플라스틱", # 예측 결과에 따라 동적으로 변경
        "confidence": 0.95, # 예측 신뢰도 (예: 모델 결과)
        "top_3": [
            ["플라스틱", 0.95],
            ["캔", 0.03],
            ["종이", 0.01]
        ],
        "recycling_guide": { # RecyclingItem 모델 등을 이용해 정보를 가져와야 함
            "category": "플라스틱",
            "action": "내용물을 비우고 깨끗하게 헹군 뒤 압착하여 배출"
        },
        # image_data_uri는 필요에 따라 Base64 인코딩 후 추가할 수 있습니다.
        # "image_data_uri": "data:image/jpeg;base64,..."
    }
    
    # 2. 결과 반환 (가장 중요)
    return JsonResponse(result_data)
    
