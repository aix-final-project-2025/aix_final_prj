# ================================================
# ☁️ Google Cloud Storage 모델 로더 (안정형 v2025.10)
# Author: Chang-Do (kopynara)
# ================================================

import os
from tensorflow import keras
from google.cloud import storage

# ✅ 환경 변수에서 GCS 버킷명 & 모델명 가져오기 (없을 시 기본값)
BUCKET_NAME = os.getenv("GCS_MODEL_BUCKET", "aix-final-prj-models")
BLOB_NAME = os.getenv("GCS_MODEL_FILE", "trash_classifier_efficientnetv2_best_final.keras")

# ✅ Cloud Run의 임시 저장 경로 (/tmp → RAM Disk)
TMP_PATH = f"/tmp/{BLOB_NAME}"

# ✅ 로컬 fallback (Docker 이미지 내 포함될 경로)
LOCAL_PATH = f"aix_final_prj/keras/{BLOB_NAME}"

def load_model_from_gcs():
    """GCS에서 모델 다운로드 후 로드, 실패 시 로컬 fallback"""
    try:
        # 1️⃣ 캐시된 모델 존재 시 재사용
        if os.path.exists(TMP_PATH):
            print(f"✅ Cached model found → {TMP_PATH}")
            return keras.models.load_model(TMP_PATH)

        # 2️⃣ GCS 모델 다운로드
        print(f"📥 Downloading model from GCS bucket '{BUCKET_NAME}' ...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        blob.download_to_filename(TMP_PATH)
        print(f"✅ Model downloaded → {TMP_PATH}")

        return keras.models.load_model(TMP_PATH)

    except Exception as e:
        print(f"⚠️ GCS download failed: {e}")
        print(f"➡️ Using local fallback model → {LOCAL_PATH}")

        return keras.models.load_model(LOCAL_PATH)