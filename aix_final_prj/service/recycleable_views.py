# trashapp/views.py
import os
from django.views.generic import TemplateView, FormView
from django import forms
from django.shortcuts import render
from django.conf import settings
from django.views import View
from django.http import JsonResponse
import io
from PIL import Image
import base64
from io import BytesIO

from aix_final_prj.service.keras_utils import pil_to_base64,fix_image_orientation
from .efficient_net_v2m import predict_from_pil

# 간단 업로드 폼
class ImageUploadForm(forms.Form):
    image = forms.ImageField(required=True)

# REST API 엔드포인트: POST multipart/form-data 'image'
IMAGE_SIZE = (224, 224)  # 모델 입력 사이즈

class UploadView(FormView):
    template_name = "upload.html"
    form_class = ImageUploadForm
    
    def form_valid(self, form):
        image   = form.cleaned_data["image"]
        pil_img = Image.open(image).convert("RGB")
        result = predict_from_pil(pil_img)

        context = self.get_context_data(form=form, result=result, image_url=image.url)
        return self.render_to_response(context)
    


# ver 1 #############################
class PredictApiView(View):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return JsonResponse({"error": "image file missing (field name 'image')"}, status=400)

        file = request.FILES['image']
        # 파일을 PIL 이미지로 로드 및 base64 변환
        try:
            image = Image.open(file).convert("RGB")
            image = fix_image_orientation(image)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data_uri = f"data:image/png;base64,{img_str}"
        except Exception as e:
            return JsonResponse({"error": "cannot open image: " + str(e)}, status=400)

        # predict 호출
        try:
            res = predict_from_pil(image)
            res["result_image"] = pil_to_base64(res["result_image"])
            res["image_data_uri"] = "data:image/png;base64," + res["result_image"]
        except Exception as e:
            return JsonResponse({"error": "prediction error: " + str(e)}, status=500)

        # # JSON에 base64 이미지 포함
        # res["image_data_uri"] = image_data_uri

        return JsonResponse(res)
