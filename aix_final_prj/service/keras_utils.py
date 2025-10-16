from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ExifTags
import base64
from io import BytesIO
@register_keras_serializable(package="custom")
class WarmUpCosine(LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, **kwargs):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        import tensorflow as tf
        # Warmup linear + cosine decay
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        cosine_lr = 0.5 * self.base_lr * (1 + tf.cos(
            3.14159265359 * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        ))
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


def pil_to_base64(image: Image.Image) -> str:
    """
    PIL 이미지를 base64 문자열로 변환
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def fix_image_orientation(image: Image.Image) -> Image.Image:
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                key = ExifTags.TAGS.get(tag, tag)
                if key == "Orientation":
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
                    break
    except Exception as e:
        print("EXIF 처리 오류:", e)
    return image