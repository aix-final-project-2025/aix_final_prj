from django.urls import path
from django.views import View
from django.shortcuts import render
from aix_final_prj.service.ragview import RagView
import aix_final_prj.service.ds_views as dsv
import aix_final_prj.service.bmi_views as bmiview
import aix_final_prj.service.coffee_views as coffeeview
import aix_final_prj.service.sleep_views as sleepview
import aix_final_prj.service.stress_views as stressview

#class dsPage(View):
 #   template_name = "ds.html"
    
#    def get(self, request):
#       context = {}
#        return render(request, self.template_name, context)
    
class ds_views(View):
    template_name = "ds.html"
        
    def get(self, request):
        context = {}
        return (dsv.coffee_analysis_view)
    
class bmiPage(View):
    template_name = "bmi.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context)
    
    
class bmi_views(View):
    template_name = "bmi.html"
    
    def get(self, request):
        context = {}
        return (bmiview.run_regression_bmi, bmiview.run_classification_bmi,  bmiview.run_clustering_bmi, bmiview.predict_dl_bmi)
    
class coffeePage(View):
    template_name = "coffee.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context)

class coffee_views(View):
    template_name = "coffee.html"
        
    def get(self, request):
        context = {}
        return (coffeeview.run_regression_cf, coffeeview.run_classification_cf, coffeeview.run_clustering_cf, coffeeview.predict_dl_cf) 
    
class sleepPage(View):
    template_name = "sleep.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context)
    
class sleep_views(View):
    template_name = "sleep.html"
    
    def get(self, request):
        context = {}
        return (sleepview.run_regression_sl, sleepview.run_classification_sl, sleepview.run_clustering_sl, sleepview.predict_dl_sl)

class stressPage(View):
    template_name = "stress.html"

    def get(self, request):
        context = {}
        return render(request, self.template_name, context)

class stress_views(View):
    template_name = "stress.html"
     
    def get(self, request):
        context = {}
        return (stressview.run_classification_st, stressview.predict_dl_st)

# Dev2 개발자용
urlpatterns = [
    # http://127.0.0.1:8000/dev/test
#    path('ds', dsPage.as_view(), name='ds'),
    path('ds/', dsv.coffee_analysis_view, name='ds'),
    path('bmi', bmiPage.as_view(), name='bmi'),
    path('coffee', coffeePage.as_view(), name='coffee'),
    path('sleep', sleepPage.as_view(), name='sleep'),
    path('stress', stressPage.as_view(), name='stress'),
    path("ragview/", RagView.as_view(), name="ragview"),
    path('run_regression_bmi/', bmiview.run_regression_bmi, name='run_regression_bmi'),
    path('run_classification_bmi/',  bmiview.run_classification_bmi, name='run_classification_bmi'),
    path('run_clustering_bmi/',  bmiview.run_clustering_bmi, name='run_clustering_bmi'),
    path('predict_dl_bmi/',  bmiview.predict_dl_bmi, name='predict_dl_bmi'),
    path('run_regression_cf/', coffeeview.run_regression_cf, name='run_regression_cf'),
    path('run_classification_cf/', coffeeview.run_classification_cf, name='run_classification_cf'),
    path('run_clustering_cf/', coffeeview.run_clustering_cf, name='run_clustering_cf'),
    path('predict_dl_cf/', coffeeview.predict_dl_cf, name='predict_dl_cf'),
    path('run_regression_sl/', sleepview.run_regression_sl, name='run_regression_sl'),
    path('run_classification_sl/', sleepview.run_classification_sl, name='run_classification_sl'),
    path('run_clustering_sl/', sleepview.run_clustering_sl, name='run_clustering_sl'),
    path('predict_dl_sl/', sleepview.predict_dl_sl, name='predict_dl_sl'),
    path('run_classification_st/', stressview.run_classification_st, name='run_classification_st'),
    path('predict_dl_st/',stressview.predict_dl_st, name='predict_dl_st'),
]