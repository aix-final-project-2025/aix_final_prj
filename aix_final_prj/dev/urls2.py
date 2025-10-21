from django.urls import path
from aix_final_prj.service import (
    ds_views as dsv,
    bmi_views as bmiview,
    coffee_views as coffeeview,
    sleep_views as sleepview,
    stress_views as stressview,
)
from aix_final_prj.service.ragview import RagView

urlpatterns = [
    # ☕ 커피 관련 페이지
    path("coffee/", coffeeview.run_regression_cf, name="run_regression_cf"),
    path("bmi/", bmiview.run_regression_bmi, name="run_regression_bmi"),
    path("sleep/", sleepview.run_regression_sl, name="run_regression_sl"),
    path("stress/", stressview.run_classification_st, name="run_classification_st"),

    # 📊 분석 페이지
    path("ds/", dsv.coffee_analysis_view, name="ds"),

    # 🧠 RAG 뷰어
    path("ragview/", RagView.as_view(), name="ragview"),

    # ✅ 커피 딥러닝 예측 엔드포인트
    path("predict_dl_bmi/", bmiview.predict_dl_bmi, name="predict_dl_bmi"),
    path("predict_dl_cf/", coffeeview.predict_dl_cf, name="predict_dl_cf"),
    path("predict_dl_sl/", sleepview.predict_dl_sl, name="predict_dl_sl"),
    path("predict_dl_st/", stressview.predict_dl_st, name="predict_dl_st"),
]