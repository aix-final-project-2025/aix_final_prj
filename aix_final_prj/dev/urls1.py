from django.urls import path
from django.http import JsonResponse
from aix_final_prj.service.recycleable_views import (
    Settings, ClassChange, CodeList,
    PredictListView, PredictListPageView,
    UploadView, PredictApiView
)

# 🔹 간단한 테스트 응답 (헬스체크용)
def dummy_wellknown(_):
    return JsonResponse({"status": "ok"}, status=200)

urlpatterns = [
    # ♻️ AI 재활용 분류 / 업로드 / 코드 관련
    path("api/predict/", PredictApiView.as_view(), name="api_predict"),
    path("api/upload/", UploadView.as_view(), name="api_upload"),
    path("api/predict_request_list/", PredictListPageView.as_view(), name="api_predict_request_list"),
    path("api/predict_list_page/", PredictListView.as_view(), name="api_predict_list"),

    # ✅ 여기 이름 통일!
    path("api/class_change/", ClassChange.as_view(), name="predict_class_change"),
    path("api/code/", CodeList.as_view(), name="code_list"),
    path("api/settings/", Settings.as_view(), name="api_settings"),

    # ✅ 헬스체크용 (Cloud Run .well-known)
    path(".well-known/<path:subpath>", dummy_wellknown),
]