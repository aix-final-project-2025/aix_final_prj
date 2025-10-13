import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from django.conf import settings
from aix_final_prj.service.keras_utils import WarmUpCosine

# ====== 설정 ======
MODEL_PATH = getattr(settings, "TRASH_MODEL_PATH", "trash_classifier_efficientnetv2_best_final.keras")
CLASS_NAMES_PATH = getattr(settings, "TRASH_CLASS_NAMES_PATH", "class_names.json")
IMAGE_SIZE = (224, 224)
THRESHOLD_DEFAULT = 0.9

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

# ====== 전역 캐시 ======
_MODEL = None
_CLASS_NAMES = None


def load_model_and_classes(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    """
    모델과 클래스 이름 목록을 메모리에 로드
    """
    global _MODEL, _CLASS_NAMES
    if _MODEL is None:
        # _MODEL = tf.keras.models.load_model(model_path) 
        _MODEL = tf.keras.models.load_model(model_path,custom_objects={"WarmUpCosine": WarmUpCosine})
        
        print("Model loaded from:", model_path)

    if _CLASS_NAMES is None:
        if os.path.exists(class_names_path):
            with open(class_names_path, "r", encoding="utf-8") as f:
                _CLASS_NAMES = json.load(f)
            print("Class names loaded from:", class_names_path)
        else:
            # fallback
            _CLASS_NAMES = [f"class_{i}" for i in range(_MODEL.output_shape[-1])]
            print("class_names.json not found. Using index-based class names.")

    return _MODEL, _CLASS_NAMES


def set_class_names(class_names):
    global _CLASS_NAMES
    _CLASS_NAMES = list(class_names)


def preprocess_image(image: Image.Image, target_size=IMAGE_SIZE):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.BILINEAR)
    arr = np.array(image).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return arr


def get_recycling_guidance(predicted_class):
    """
    예측된 클래스 이름을 기반으로 분리수거 가이드를 반환
    """
    if predicted_class in TRASH_GUIDE_MAP:
        return TRASH_GUIDE_MAP[predicted_class]
    else:
        # 기본 가이드 제공 (혹시 누락된 클래스가 있을 경우)
        return {"category": "일반 분류", "action": "라벨/뚜껑 제거 후 깨끗이 세척하여 배출"}



def classify_image(model, image_path, classes, threshold=0.5):
    """
    학습용 코드(옵션1) 기반으로 안전하게 이미지 분류
    Args:
        model: 학습된 Keras 모델
        image_path: 분류할 이미지 경로
        classes: 학습 시 사용된 클래스 리스트
        threshold: confidence 임계값
    Returns:
        class_name 또는 "일반쓰레기"
    """
    # 이미지 읽기 및 디코딩 (JPEG, PNG 모두 가능)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    
    # 학습용 코드 기준: 이미 224x224이므로 resize 불필요
    img.set_shape([224, 224, 3])
    
    # 정규화
    img = tf.cast(img, tf.float32) / 255.0
    
    # 배치 차원 추가
    img = tf.expand_dims(img, axis=0)
    
    # 예측
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    
    # threshold 기준 판정
    if confidence < threshold:
        return "일반쓰레기"
    else:
        return classes[class_id], float(confidence)

# ver 1
def predict_from_pil(image: Image.Image, threshold=THRESHOLD_DEFAULT):
    """
    PIL 이미지 → 예측 수행 → 결과 반환
    """
  
    global _MODEL, _CLASS_NAMES
    if _MODEL is None or _CLASS_NAMES is None:
        raise RuntimeError("Model or class names not loaded. Call load_model_and_classes() first.")

    
    x = preprocess_image(image)
    print(x)
    preds = _MODEL.predict(x, verbose=1)[0]

    max_idx = int(np.argmax(preds))
    print(max_idx)
    print(f"====={_CLASS_NAMES[max_idx]}=")    
  
    max_prob = float(preds[max_idx])
    predicted_class = _CLASS_NAMES[max_idx]
    
    print(" 111111111111 ---")
    if max_prob >= threshold:
        result_message = f"🟢 [확정]: {predicted_class}로 분류되었습니다."
        confidence_level = "높음"
    else:
        result_message = f"🟡 [불확실]: {predicted_class}로 예측되지만, 신뢰도({max_prob*100:.2f}%)가 낮아 재확인이 필요합니다."
        confidence_level = "낮음"
    print(" 22222222222222 ")
    top3_idx = np.argsort(preds)[::-1][:3]
    top_3 = [( _CLASS_NAMES[int(i)], float(preds[int(i)]) ) for i in top3_idx]

    guide = get_recycling_guidance(predicted_class)

    return {
        "predicted_class": predicted_class,
        "confidence": max_prob,
        "result_message": result_message,
        "confidence_level": confidence_level,
        "top_3": top_3,
        "recycling_guide": guide
    }


def predict_from_pil_v2(image: Image.Image, threshold=THRESHOLD_DEFAULT):
    """
    PIL 이미지 → 예측 수행 → 결과 반환
    """
    global _MODEL, _CLASS_NAMES
    if _MODEL is None or _CLASS_NAMES is None:
        raise RuntimeError("Model or class names not loaded. Call load_model_and_classes() first.")

    # x = preprocess_image(image)
    # print(x)
    # preds = _MODEL.predict(x, verbose=1)[0]

    # ✅ 이미 PIL.Image 객체이므로 open() 불필요
    img = image.convert('RGB').resize((224, 224))

    # ✅ 올바른 전처리
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = _MODEL.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]


    max_idx = int(np.argmax(preds))
    print(max_idx)
    print(f"====={_CLASS_NAMES[max_idx]}=")    

    max_prob = float(preds[max_idx])
    predicted_class = _CLASS_NAMES[max_idx]



    if class_idx in TRASH_GUIDE_MAP:
        label = "♻️ 재활용 가능 쓰레기"
    else:
        label = "🚮 일반 쓰레기"
    confidence = float(np.max(preds))

    if max_prob >= threshold:
        result_message = f"🟢 [확정]: {predicted_class}로 분류되었습니다."
        confidence_level = "높음"
    else:
        result_message = f"🟡 [불확실]: {predicted_class}로 예측되지만, 신뢰도({max_prob*100:.2f}%)가 낮아 재확인이 필요합니다."
        confidence_level = "낮음"
  

    top3_idx = np.argsort(preds)[::-1][:3]
    top_3 = [( _CLASS_NAMES[int(i)], float(preds[int(i)]) ) for i in top3_idx]

    guide = get_recycling_guidance(predicted_class)

    return {
        "predicted_class": predicted_class,
        "confidence": max_prob,
        "result_message": result_message,
        "confidence_level": confidence_level,
        "top_3": top_3,
        "recycling_guide": guide
    }
