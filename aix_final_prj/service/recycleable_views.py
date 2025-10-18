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
from .tts_utils import translate_and_tts
from .models import RecyclableResult, GroupCode


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

        #DB에 결과 등록
        try:
            # 예측 결과에서 group_code_name 가져오기 (예: "steel_can1")
            predicted_class = res.get("predicted_class")  # predict_from_pil에서 반환되도록 수정 필요
            # predicted_code_name = res.get("predicted_code")  # predict_from_pil에서 반환되도록 수정 필요
            group_code = None
            if predicted_class:
                group_code = GroupCode.objects.filter(code=predicted_class).first()
                # if group_code:
                #    numeric_code = group_code.id  # numeric_code 가져오기


            result_message = res.get('result_message', '')
            print(f"predicted_class {res.get('predicted_class', '')}")
            print(f"confidence {res.get('confidence', '')}")
            print(f"confidence_level {res.get('confidence_level', '')}")
            print(f"result_message {result_message}")
            print(f"top3 {res.get('top3', '')}")
            print(f"category {res.get('category', '')}")
            print(f"recycling_guide {group_code.id}")
            # RecyclableResult 저장
            RecyclableResult.objects.create(
                PREDICTED_CLASS=res.get("predicted_class", ""),
                CONFIDENCE=res.get("confidence", 0.0),
                CONFIDENCE_LEVEL=res.get("confidence_level", ""),
                RESULT_MESSAGE=result_message,
                TOP_3=res.get("top3", ""),
                RECYCLING_GUIDE=res.get("recycling_guide", ""),
                RESULT_IMAGE=file,  # 실제 업로드된 이미지 그대로 저장
                group_code_id=group_code.id
            )
        except Exception as e:
            # DB 등록 실패는 로그만 남기고, 예측 결과는 반환
            print("DB save error:", e)

        enable = os.getenv('ENABLE')
        res["tts_able"] = enable
        if(enable == 1):
            #JSON 반환
            tts_name = translate_and_tts(f'{result_message}','en')
            host = request.scheme + "://" + request.get_host()
            res["tts_url"] = host + settings.MEDIA_URL +  tts_name['tts_name']
     
        return JsonResponse(res)


