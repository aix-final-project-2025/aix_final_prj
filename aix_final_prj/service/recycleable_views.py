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
from .efficient_net_v2m_v2 import predict_with_tta_file
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
    


class PredictApiView_v1(View):
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



# ====== 분리수거 가이드 (모든 클래스 매핑) ======
TRASH_GUIDE_MAP = {
    # --- 캔류 ---
    "steel_can1": {"category": "캔류", "action": "내용물 비우고, 물로 헹군 후 압착하여 배출"},
    "steel_can2": {"category": "캔류", "action": "내용물 비우고, 물로 헹군 후 압착하여 배출"},
    "steel_can3": {"category": "캔류", "action": "내용물 비우고, 물로 헹군 후 압착하여 배출"},
    "aluminum_can1": {"category": "캔류", "action": "내용물 비우고, 물로 헹군 후 압착하여 배출"},
    "aluminum_can2": {"category": "캔류", "action": "내용물 비우고, 물로 헹군 후 압착하여 배출"},

    # --- 종이류 ---
    "paper1": {"category": "종이류", "action": "물기에 젖지 않도록 모아서 끈으로 묶어 배출"},
    "paper2": {"category": "종이류", "action": "스프링·코팅 제거 후 묶어서 배출"},

    # --- 플라스틱 (PET 투명) ---
    "pet_clear_single1": {"category": "플라스틱(PET 투명)", "action": "내용물 비우고 라벨·뚜껑 제거 후 압착하여 배출"},
    "pet_clear_single2": {"category": "플라스틱(PET 투명)", "action": "내용물 비우고 라벨·뚜껑 제거 후 압착하여 배출"},
    "pet_clear_single3": {"category": "플라스틱(PET 투명)", "action": "내용물 비우고 라벨·뚜껑 제거 후 압착하여 배출"},

    # --- 플라스틱 (PET 유색) ---
    "pet_colored_single1": {"category": "플라스틱(PET 유색)", "action": "내용물 비우고 라벨·뚜껑 제거 후 배출"},
    "pet_colored_single2": {"category": "플라스틱(PET 유색)", "action": "내용물 비우고 라벨·뚜껑 제거 후 배출"},
    "pet_colored_single3": {"category": "플라스틱(PET 유색)", "action": "내용물 비우고 라벨·뚜껑 제거 후 배출"},

    # --- 플라스틱 (기타 재질) ---
    "plastic_pe1": {"category": "플라스틱(PE)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_pe2": {"category": "플라스틱(PE)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_pp1": {"category": "플라스틱(PP)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_pp2": {"category": "플라스틱(PP)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_pp3": {"category": "플라스틱(PP)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_ps1": {"category": "플라스틱(PS)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_ps2": {"category": "플라스틱(PS)", "action": "내용물 비우고 이물질 제거 후 배출"},
    "plastic_ps3": {"category": "플라스틱(PS)", "action": "내용물 비우고 이물질 제거 후 배출"},

    # --- 유리류 ---
    "glass_clear": {"category": "유리류", "action": "뚜껑 제거 후 색상별로 배출"},
    "glass_brown": {"category": "유리류", "action": "뚜껑 제거 후 색상별로 배출"},
    "glass_green": {"category": "유리류", "action": "뚜껑 제거 후 색상별로 배출"},

    # --- 스티로폼 ---
    "styrofoam1": {"category": "스티로폼", "action": "내용물 제거 후 깨끗이 세척하여 배출"},
    "styrofoam2": {"category": "스티로폼", "action": "이물질 제거 후 배출 (오염 심하면 종량제 봉투)"},

    # --- 비닐류 ---
    "vinyl": {"category": "비닐류", "action": "깨끗이 세척 후 건조하여 배출 (오염 심하면 종량제 봉투)"},

    # --- 특수 폐기물 ---
    "battery": {"category": "특수 폐기물", "action": "폐건전지 수거함에 배출 (분리수거 아님)"},
    "fluorescent_lamp": {"category": "특수 폐기물", "action": "전용 수거함에 배출 (분리수거 아님)"}
}

def get_recycling_guidance(predicted_class):
    """
    예측된 클래스 이름을 기반으로 분리수거 가이드를 반환
    """
    if predicted_class in TRASH_GUIDE_MAP:
        return TRASH_GUIDE_MAP[predicted_class]
    else:
        # 기본 가이드 제공 (혹시 누락된 클래스가 있을 경우)
        return {"category": "일반 분류", "action": "라벨/뚜껑 제거 후 깨끗이 세척하여 배출"}


def get_top_result_value(result_list, key_name):
    """
    결과 리스트(result_list)의 첫 번째 항목(가장 확률 높은 딕셔너리)에서 
    특정 키(key_name)에 해당하는 값을 추출하여 반환합니다.
    
    Args:
        result_list (list): 분석 결과 딕셔너리들의 리스트. 
                            예: [{'class': 'battery', 'prob': 0.991...}, ...]
        key_name (str): 찾고자 하는 키 이름. 예: 'class', 'prob', 'idx'.
        
    Returns:
        any: 해당 키의 값. 리스트가 비어있거나 키가 존재하지 않으면 None을 반환합니다.
    """
    # 1. 리스트가 비어있는지 확인
    if not result_list:
        return None
        
    # 2. 가장 확률이 높은 첫 번째 딕셔너리 선택
    top_result_dict = result_list[0]
    print(f"result {top_result_dict}")
    # 3. 딕셔너리에서 해당 키의 값을 추출 (키가 없으면 None 반환)
    return top_result_dict.get(key_name)

def load_image_from_file(file_obj) -> Image.Image:
    img = Image.open(file_obj).convert("RGB")
    return img
# ver 1 #############################
class PredictApiView(View):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return JsonResponse({"error": "image file missing (field name 'image')"}, status=400)

        file = request.FILES['image']
        
        try:
            res = predict_with_tta_file(file)
            res["result_image"] = pil_to_base64(load_image_from_file(file))
            res["image_data_uri"] = "data:image/png;base64," + res["result_image"]
        except Exception as e:
            return JsonResponse({"error": "prediction error: " + str(e)}, status=500)
       
        try:
            predicted_class = res.get("top_k")  # predict_from_pil에서 반환되도록 수정 필요
            code = get_top_result_value(predicted_class,'class')
            print(f" predict : {code}")
            group_code = None
            if predicted_class:
                group_code = GroupCode.objects.filter(code=code).first()
                if group_code:
                   numeric_code = group_code.id  # numeric_code 가져오기

            print(f"group_code {numeric_code}")

            # top3_idx = np.argsort(preds)[::-1][:3]
            # top_3 = [(_CLASS_NAMES[i], float(preds[i])) for i in top3_idx]
            top_3 = res.get("top_k")
            top_3_list_of_tuples = [
                (d['class'], d['prob']) 
                for d in top_3
            ]
            res["top_3"] = top_3_list_of_tuples
            
            
            result_message = f"{group_code}로 분류 되었습니다."
            recycling_guide = get_recycling_guidance(code)
            res["result_message"] = result_message
            res["recycling_guide"] = recycling_guide
            print(f"1 top_3 {res.get('top_3')}")

            print(f"2 result_message {result_message}")
            # print(f"group_code.id {group_code.id}")
            print(f"3 predicted_class {code}")
            print(f"4 confidence {res.get('confidence', '')}")
            print(f"5 confidence_level {res.get('confidence_level', '')}")
            print(f"6 recycling_guide {recycling_guide}")
            print(f"7 recycling_guide {recycling_guide['action']}")
            
        #     # RecyclableResult 저장
            RecyclableResult.objects.create(
                PREDICTED_CLASS=code,
                CONFIDENCE=res.get("confidence", 0.0),
                CONFIDENCE_LEVEL=res.get("confidence_level", ""),
                RESULT_MESSAGE=result_message,
                TOP_3=top_3_list_of_tuples,
                RECYCLING_GUIDE= recycling_guide,
                RESULT_IMAGE=file,  # 실제 업로드된 이미지 그대로 저장
                group_code_id=numeric_code
            )
        except Exception as e:
            # DB 등록 실패는 로그만 남기고, 예측 결과는 반환
            print("DB save error:", e)


        enable = os.getenv('ENABLE')

        rsEnable = CountryPf.objects.all().order_by('-created_at').first()
        # enable = 0
        if rsEnable:
            enable = rsEnable.active
            print(f"TTs 사용여부 : {enable}")
      
        print(f" enable data {enable}")
        res["tts_able"] = enable
        if(enable == 1):
            #JSON 반환
            print(f"  TTS called ======{rsEnable.country} {rsEnable.gender}")
            tts_name = translate_and_tts(f'{result_message}\n{recycling_guide["action"]}',rsEnable.country,rsEnable.gender)
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
    template_name = "predict_list.html"


    def get(self, request):

        print("called ----------")
        page = int(request.GET.get("page", 1))
        per_page = 20
        start = (page - 1) * per_page
        end = page * per_page

        # QuerySet 슬라이싱
        # qs = RecyclableResult.objects.order_by('-id')[start:end]

        qs_items = RecyclableResult.objects.select_related('group_code').order_by('-id')[start:end]

        # items_list = []
        # for item in qs_items:
        #     items_list.append({
        #         'id': item.id,
        #         'image_url': item.RESULT_IMAGE.url if item.RESULT_IMAGE else '',
        #         'predicted_result': item.PREDICTED_CLASS,
        #         'top3': item.TOP_3,
        #         'recycling_guide': item.RECYCLING_GUIDE,
        #         'group_code_id': item.group_code.id,  # group_code의 icode를 그대로 사용
        #         'group_code_name': item.group_code.name,
        #     })


        # items = [
        #     {
        #         "id": w.id,
        #         "image_url": w.RESULT_IMAGE.url if w.RESULT_IMAGE else "",
        #         "predicted_result": w.RESULT_MESSAGE or "",
        #         "predicted_class": w.PREDICTED_CLASS or "",
        #         "recycling_guide": w.RECYCLING_GUIDE or "",
        #         "group_code_id": w.group_code_id or ""
        #     }
        #     for w in qs_items
        # ]

        items = [
            {
                'id': w.id,
                'image_url': w.RESULT_IMAGE.url if w.RESULT_IMAGE else '',
                'predicted_result': w.RESULT_MESSAGE,
                'top3': w.TOP_3,
                "predicted_class": w.PREDICTED_CLASS or "",
                'recycling_guide': w.RECYCLING_GUIDE,
                'group_code_id': w.group_code.code,  # group_code의 icode를 그대로 사용
                'group_code_name': w.group_code.name,
            }
            for w in qs_items
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
                'active': settingInfo.active,
                'result': 0
            }

        else :
            print(" 2 is not None")
            sdict = {
                'country': 'ko',
                'gender': 2,
                'active': False,
                'result': 1
                # 필요한 필드만 선택
             }
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
            'status': 'none', 
            'message': 'No changes detected.'
        }, status=200)