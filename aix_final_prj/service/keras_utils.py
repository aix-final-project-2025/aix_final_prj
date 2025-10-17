import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.utils import register_keras_serializable


# =========================================================
# 🧠 Custom WarmUpCosine Scheduler
# =========================================================
@register_keras_serializable(package="custom")
class WarmUpCosine(LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, **kwargs):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # Warmup linear + cosine decay
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        cosine_lr = 0.5 * self.base_lr * (1 + tf.cos(
            3.14159265359 * (step - self.warmup_steps)
            / (self.total_steps - self.warmup_steps)
        ))
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


# =========================================================
# 🧩 Model & Class Names Loader (절대경로 안전 버전)
# =========================================================
def load_model_and_classes():
    """
    절대경로 기반 모델 + 클래스 이름 로더
    - 실행 위치와 무관하게 정상 작동 (Mac mini, WSL, Hugging Face 공용)
    """

    # 현재 파일 기준으로 절대경로 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    KERAS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "keras")

    MODEL_PATH = os.path.join(KERAS_DIR, "trash_classifier_efficientnetv2_best_final.keras")
    CLASS_PATH = os.path.join(KERAS_DIR, "class_names.json")

    print("🔍 [경로 디버그] CURRENT_DIR:", CURRENT_DIR)
    print("🔍 [경로 디버그] MODEL_PATH:", MODEL_PATH)
    print("🔍 [경로 디버그] CLASS_PATH:", CLASS_PATH)

    # ✅ 모델 로드
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model from {MODEL_PATH}: {e}")
        raise e

    # ✅ 클래스 이름 로드
    if os.path.exists(CLASS_PATH):
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                class_names = json.load(f)
            print(f"✅ Class names loaded successfully from: {CLASS_PATH}")
        except Exception as e:
            print(f"⚠️ Error reading class_names.json: {e}")
            class_names = [str(i) for i in range(model.output_shape[-1])]
    else:
        print("⚠️ class_names.json not found. Using index-based labels.")
        class_names = [str(i) for i in range(model.output_shape[-1])]

    return model, class_names