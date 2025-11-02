import os
import io
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# 경로 설정 (프로젝트 루트에서 상대경로)
# KERAS_DIR = os.path.join(os.getcwd(), "keras")
# MODEL_PATH = os.path.join(KERAS_DIR, "trash_classifier_efficientnetv2_best_final.keras")
# CLASS_JSON = os.path.join(KERAS_DIR, "class_names.json")


KERAS_DIR = Path(__file__).resolve().parent.parent
# model_path = BASE_DIR / "keras" / "trash_classifier_efficientnetv2_best_final.keras"
# class_json = BASE_DIR / "keras" / "class_names.json"
MODEL_PATH = str(KERAS_DIR / "keras" / "trash_classifier_efficientnetv2_best_final.keras")
CLASS_JSON = str(KERAS_DIR / "keras" / "class_names.json")

IMG_SIZE = (224, 224)
TTA_TRANSFORMS = [
    "orig", "flip_lr", "flip_ud",
    "rot_10", "rot_-10",
    "bright_up", "bright_down",
    "contrast_up", "contrast_down"
]

# 안전: GPU 메모리 growth 설정 (있을 때만)
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        for g in _gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# Load class names
if not os.path.exists(CLASS_JSON):
    raise FileNotFoundError(f"class_indices.json not found at {CLASS_JSON}")
with open(CLASS_JSON, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

# Load model (raise clear error if not found)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and place model there.")
MODEL = load_model(MODEL_PATH)
NUM_CLASSES = len(CLASS_NAMES)

# validation: model output dim vs class count
if MODEL.output_shape[-1] != NUM_CLASSES:
    raise RuntimeError(f"Model output dim ({MODEL.output_shape[-1]}) != class count ({NUM_CLASSES}).")

def load_image_from_file(file_obj) -> Image.Image:
    img = Image.open(file_obj).convert("RGB")
    return img

def resize_and_center_crop(img: Image.Image, target_size=IMG_SIZE) -> Image.Image:
    img.thumbnail((max(target_size)*2, max(target_size)*2), Image.LANCZOS)
    w, h = img.size
    tw, th = target_size
    left = max((w - tw)//2, 0)
    top = max((h - th)//2, 0)
    right = left + tw
    bottom = top + th
    img = img.crop((left, top, right, bottom)).resize(target_size, Image.LANCZOS)
    return img

def pil_to_model_input(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32)
    arr = preprocess_input(arr)  # EfficientNetV2 preprocessing
    return arr

def apply_tta(img: Image.Image, tname: str) -> Image.Image:
    if tname == "orig":
        return img.copy()
    if tname == "flip_lr":
        return ImageOps.mirror(img)
    if tname == "flip_ud":
        return ImageOps.flip(img)
    if tname == "rot_10":
        return img.rotate(10, resample=Image.BILINEAR, expand=False)
    if tname == "rot_-10":
        return img.rotate(-10, resample=Image.BILINEAR, expand=False)
    if tname == "bright_up":
        return ImageEnhance.Brightness(img).enhance(1.2)
    if tname == "bright_down":
        return ImageEnhance.Brightness(img).enhance(0.8)
    if tname == "contrast_up":
        return ImageEnhance.Contrast(img).enhance(1.2)
    if tname == "contrast_down":
        return ImageEnhance.Contrast(img).enhance(0.8)
    return img.copy()

def predict_with_tta_file(file_obj, tta_list=TTA_TRANSFORMS, top_k=3):
    img = load_image_from_file(file_obj)
    imgs = []
    imgt = None
    for t in tta_list:
        timg = apply_tta(img, t)
        timg = resize_and_center_crop(timg, IMG_SIZE)
        arr = pil_to_model_input(timg)
        imgs.append(arr)

    batch = np.stack(imgs, axis=0)  # shape (TTA, H, W, C)
    preds = MODEL.predict(batch, verbose=0)  # (TTA, num_classes)
    avg = np.mean(preds, axis=0)
    top_idx = np.argsort(avg)[::-1][:top_k]
    results = [{"class": CLASS_NAMES[i], "prob": float(avg[i]), "idx": int(i)} for i in top_idx]
    print(f" result : {results}")
    # # --- Base64 변환 옵션 시작 ---
    # resized_img = resize_and_center_crop(img.copy(), IMG_SIZE)
    
    # # 1. 메모리 버퍼 생성 (바이트 스트림)
    # buffer = io.BytesIO()
    
    # # 2. 이미지를 PNG 형식으로 버퍼에 저장
    # # 여기서 'PNG'는 이미지 포맷 옵션입니다. data URI의 image/png와 일치해야 합니다.
    # resized_img.save(buffer, format="PNG") 
    
    # # 3. 바이트 데이터를 Base64로 인코딩
    # img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # --- Base64 변환 옵션 끝 ---

    # return {"top_k": results, "avg_prob_vector": avg.tolist(),"result_image":img_base64}
    return {"top_k": results, "avg_prob_vector": avg.tolist()}
