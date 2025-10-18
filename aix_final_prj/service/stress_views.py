import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
#from aix_final_prj.service.bmi_views import preprocess_data, run_regression_analysis, run_classification_analysis, run_clustering_analysis, predict_bmi_category
import aix_final_prj.service.stress_logic as stress_logic
# 딥러닝 모델을 서버 시작 시 미리 로드
stress_logic.get_trained_model()

# 메인 페이지를 렌더링하는 뷰
def index(request):
    return render(request, 'bmi.html')

# --- API 역할을 하는 View 함수들 ---

@csrf_exempt
def run_classification_st(request):
    if request.method == 'POST':
        results = stress_logic.run_classification_analysis()
        return JsonResponse({"results": results})
    return JsonResponse({"error": "Invalid request method"}, status=405)

@csrf_exempt
def predict_dl_st(request):
    if request.method == 'POST':
        try:
            # request.body에서 JSON 데이터를 파싱
            input_data = json.loads(request.body)
            result = stress_logic.predict_stress_category(input_data)
            return JsonResponse(result)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)