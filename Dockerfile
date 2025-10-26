# ============================
# 🧠 AIX Final Project (Cloud Run 안정형 배포용)
# ============================

FROM python:3.11-slim

# 1️⃣ 필수 시스템 패키지 설치 (TensorFlow CPU & Django 실행 유틸 포함)
RUN apt update && apt install -y \
    curl \
    procps \
    nano \
    net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ 작업 디렉토리 설정
WORKDIR /app

# 3️⃣ 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir google-cloud-storage

# 4️⃣ 소스 복사
COPY . .

# ✅ 4.5️⃣ 업로드 폴더 자동 생성 (Cloud Run에서도 누락 방지)
RUN mkdir -p /app/uploads/recyclable_results

# 5️⃣ 정적 파일 수집 (CSS, JS, 이미지 등)
RUN python manage.py collectstatic --noinput

# 6️⃣ 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# 7️⃣ Cloud Run 기본 포트 (자동 주입: $PORT → 기본 8080)
ENV PORT=8080
EXPOSE 8080

# 8️⃣ Gunicorn 실행 (운영 표준 방식 - 마이그레이션 제거)
CMD ["bash", "-c", "gunicorn aix_final_prj.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 0"]