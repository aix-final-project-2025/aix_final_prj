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
import json
from .models import CountryPf

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
            print(" ==================== ")
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

        # enable = os.getenv('ENABLE')

        rsEnable = CountryPf.objects.all().order_by('-created_at').first()
        enable = 0
        if rsEnable:
            enable = rsEnable.active
            print(f"TTs 사용여부 : {enable}")
      
        print(f" enable data {enable}")
        res["tts_able"] = enable
        if(enable == 1):
            #JSON 반환
            print(f"  TTS called ======{rsEnable.country} {rsEnable.gender}")
            
            tts_name = translate_and_tts(f'{result_message}',rsEnable.country,rsEnable.gender)
            host = request.scheme + "://" + request.get_host()
            res["tts_url"] = host + settings.MEDIA_URL +  tts_name['tts_name']
     
        return JsonResponse(res)





"""
생활쓰레기 리스트화면 호출
"""    
class PredictListPageView(TemplateView):
    print('PredictListView called')
    template_name = "predict_list.html"


"""
생활쓰레기 리스트조회
"""
class PredictListView(View):
    def get(self, request):
        page = int(request.GET.get("page", 1))
        per_page = 20
        start = (page - 1) * per_page
        end = page * per_page

        # QuerySet 슬라이싱
        qs = RecyclableResult.objects.order_by('-id')[start:end]

        items = [
            {
                "id": w.id,
                "image_url": w.RESULT_IMAGE.url if w.RESULT_IMAGE else "",
                "predicted_result": w.RESULT_MESSAGE or "",
                "predicted_class": w.PREDICTED_CLASS or "",
                "recycling_guide": w.RECYCLING_GUIDE or "",
                "group_code_id": w.group_code_id or ""
            }
            for w in qs
        ]

        has_more = RecyclableResult.objects.count() > end

        return JsonResponse({"items": items, "has_more": has_more})


"""
코드목록 가져오기
"""
class CodeList(View):   
    def get(self, request):
            print("===========CodeList=============")
            # 전체 목록 조회
            qs = GroupCode.objects.order_by('-id')
            print(qs.first().__dict__)  # 첫 번째 객체의 내부 dict 확인
            # JSON으로 변환
            items = [
                {"id": w.id, "name": w.name}  # 실제 존재하는 필드로 수정
                for w in qs
            ]

            return JsonResponse({"codelist": items})
    

class ClassChange(View):   
    def get(self, request):
            
            record_id = int(request.GET.get("id", 1))
            group_code_id = int(request.GET.get("group_code_id", 1))
            print(f" request group_code_id ={group_code_id} ")
            print("===========ClassChange=============")
            # 전체 목록 조회
            try:
                print(f" record_id {record_id}")
                record = RecyclableResult.objects.get(id=record_id)
                print(f" record {record}")
            except RecyclableResult.DoesNotExist:
                return JsonResponse({"success": False, "error": "해당 레코드를 찾을 수 없습니다."})

            # 새로운 group_code 존재 여부 확인
            try:
                print(f" group_code_id {group_code_id}")
                new_group_code = GroupCode.objects.get(id=group_code_id)
                print(f" new_group_code {new_group_code}")
            except GroupCode.DoesNotExist:
                return JsonResponse({"success": False, "error": "해당 GroupCode가 존재하지 않습니다."})

            print(" 1 change class. =========================")
            # group_code 변경 후 저장
            record.group_code = new_group_code
            record.save()

            print(" 2 change class. =========================")
            updated_item = {
                "id": record.id,
                "group_code_id": record.group_code.id,
                "group_code_name": record.group_code.name,
                "predicted_result": record.RESULT_MESSAGE or "",
                "category_id": record.PREDICTED_CLASS or "",
                "guide": record.RECYCLING_GUIDE or ""
            }
            print(f" response group_code_id ={updated_item['group_code_id']} ")
            return JsonResponse({"success": True, "updated_item": updated_item})
    


class Settings(TemplateView):
    template_name = "settings.html"


from .models_utils import get_profile_setting

class getSettingInfo(View):
    template_name = "settings.html"
    # ... (생략: post 메서드 시작 및 JSON 파싱)
    def post(self, request, *args, **kwargs):

        settingInfo = CountryPf.objects.all().first() 
        if settingInfo is not None:
            print(" 1 is not None")
            sdict = {
                'country': settingInfo.country,
                'gender': settingInfo.gender,
                'active': settingInfo.active
                # 필요한 필드만 선택
            }

        else :
            print(" 2 is not None")
            sdict = {
                'country': 'ko',
                'gender': 2,
                'active': False
                # 필요한 필드만 선택
             }
        print(f" setting called =={settingInfo}")
        return JsonResponse({
                'status': 'success', 
                'message': 'Existing settings updated successfully.',
                'action': 'UPDATED',
                'settingInfo': sdict
        }, status=200)




class SettingDetailView(View):
    template_name = "settings.html"
    # ... (생략: post 메서드 시작 및 JSON 파싱)
    def post(self, request, *args, **kwargs):

        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)

        # 2. 필수 필드 추출
        query_country = data.get('country')
        query_gender = data.get('gender')
        query_active = data.get('active', None) # active 값 추출 (없으면 None)
        print(f" query_country {query_country},query_gender {query_gender},query_active {query_active}")
        
        if query_country is None and query_gender is None and query_active is None:
            return JsonResponse({'status': 'error', 'message': 'Country and gender fields are required in JSON body.'}, status=400)

        # 3. 공통 모듈을 사용하여 DB 조회: active 값까지 전달
        db_profile = get_profile_setting(query_country, query_gender, query_active) # active 전달

        # ... (생략: 조회 성공/실패 로직)
        if db_profile:
            # 조회 성공: 객체를 JSON 응답 형태로 변환
            response_data = {
                'country': db_profile.country,
                'gender': db_profile.gender,
                'active': db_profile.active, 
                'created_at': db_profile.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'exists': True
            }
            return JsonResponse(response_data, status=200)
        else:

            # 조회 실패 (신규 등록이 필요한 상태 또는 해당 active 상태의 레코드가 없는 상태)
            new_profile = CountryPf.objects.create(
                    country=query_country,
                    gender=query_gender,
                    active=query_active if query_active is not None else True, # 요청된 active 값이 있으면 사용, 없으면 기본값 True 
            )
            print(f" new_profile.id {new_profile.id}")
            return JsonResponse({
                'status': 'success', 'message': 'New settings registered successfully.',
                'action': 'CREATED', 'id': new_profile.id
            }, status=200)




from django.shortcuts import get_object_or_404
from django.db import transaction
from .models import CountryPf 
from datetime import datetime

class SettingUpdateView(View):
    """
    단일 환경설정 레코드 관리: 신규 등록 또는 기존 데이터 수정.
    country, gender, active 세 항목 모두 변경이 없을 시 DB 저장을 건너뜁니다.
    """
    
    def post(self, request, *args, **kwargs):
        # 1. JSON 데이터 파싱 및 검증
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON format'}, status=400)

        request_country = data.get('country')
        request_gender = data.get('gender')
        request_active = data.get('active') 

        if not all([request_country, request_gender, request_active is not None]):
            return JsonResponse({'status': 'error', 'message': 'Missing required fields (country, gender, active)'}, status=400)
        
        if not isinstance(request_active, bool):
             return JsonResponse({'status': 'error', 'message': 'Field "active" must be a boolean value.'}, status=400)


        # 2. DB에서 기존 레코드 조회 (단일 레코드 관리 목적)
        existing_profile = CountryPf.objects.all().first() 

        # 3. 신규 등록 로직 (DB에 레코드가 없는 경우)
        if existing_profile is None:
            with transaction.atomic():
                new_profile = CountryPf.objects.create(
                    country=request_country,
                    gender=request_gender,
                    active=request_active 
                )
            return JsonResponse({
                'status': 'success', 'message': 'New settings registered successfully.',
                'action': 'CREATED', 'id': new_profile.id
            }, status=201)

        # 4. 수정 로직 (기존 데이터가 있는 경우)
        is_changed = False
        update_fields = []
        
        # 4-1. country 필드 비교
        if existing_profile.country != request_country:
            existing_profile.country = request_country
            update_fields.append('country')
            is_changed = True
            
        # 4-2. gender 필드 비교
        if existing_profile.gender != request_gender:
            existing_profile.gender = request_gender
            update_fields.append('gender')
            is_changed = True
            
        # 4-3. active 필드 비교
        if existing_profile.active != request_active:
            existing_profile.active = request_active
            update_fields.append('active')
            is_changed = True

        # 5. 변경 사항 확인 및 저장
        if is_changed:
            # 변경 사항이 있을 경우에만 save() 호출 
            
            # updated_at을 명시적으로 추가하지 않아도, 
            # save(update_fields=...) 호출 시 Django가 알아서 auto_now 필드를 갱신합니다.
            existing_profile.save(update_fields=update_fields) 
            
            return JsonResponse({
                'status': 'success', 
                'message': 'Existing settings updated successfully.',
                'action': 'UPDATED',
                'updated_fields': update_fields
            }, status=200)
        
        # 6. 변경 사항이 없는 경우 (is_changed == False)
        # 3개 항목 모두 변경이 없으므로 save()가 호출되지 않고 이 응답을 반환합니다.
        return JsonResponse({
            'status': 'info', 
            'message': 'No changes detected.'
        }, status=200)