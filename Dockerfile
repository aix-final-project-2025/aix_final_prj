# ===============================
# 🧠 AIX Final Project — Dockerfile
# Hugging Face Spaces optimized (fixed COPY order)
# ===============================

# 1️⃣ Python 환경 (HF 권장 3.11)
FROM python:3.11-slim

# 2️⃣ 필수 유틸 설치 (procps, curl 등)
RUN apt update && apt install -y \
    procps \
    curl \
    nano \
    net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 3️⃣ 작업 디렉토리 설정
WORKDIR /app

# ✅ 4️⃣ 모델 및 클래스 JSON 파일 먼저 복사 (캐시 누락 방지)
COPY ./aix_final_prj/keras/trash_classifier_efficientnetv2_best_final.keras /app/aix_final_prj/keras/
COPY ./aix_final_prj/keras/class_names.json /app/aix_final_prj/keras/

# 5️⃣ 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ 나머지 소스 전체 복사
COPY . .

# 7️⃣ 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# ✅ 8️⃣ Hugging Face 기본 포트 설정
EXPOSE 7860

# ✅ 9️⃣ Hugging Face용 진입점 (app.py 프록시 실행)
CMD ["python", "app.py"]
