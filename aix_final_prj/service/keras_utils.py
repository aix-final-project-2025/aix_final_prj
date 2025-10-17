from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.utils import register_keras_serializable

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
