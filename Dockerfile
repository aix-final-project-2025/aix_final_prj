# ===============================
# 🧠 AIX Final Project — Dockerfile
# Hugging Face Spaces optimized (app.py entry)
# ===============================

FROM python:3.11-slim

RUN apt update && apt install -y \
    procps \
    curl \
    nano \
    net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ 모델과 클래스 JSON을 미리 복사 (캐시 누락 방지)
COPY ./aix_final_prj/keras/trash_classifier_efficientnetv2_best_final.keras /app/aix_final_prj/keras/
COPY ./aix_final_prj/keras/class_names.json /app/aix_final_prj/keras/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

EXPOSE 7860

# ✅ Hugging Face는 app.py를 entrypoint로 사용
CMD ["python", "app.py"]
