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

# 4️⃣ 소스 복사
COPY . .

# 5️⃣ 정적 파일 수집 (CSS, JS, 이미지 등)
RUN python manage.py collectstatic --noinput

# 6️⃣ 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# 7️⃣ Cloud Run 기본 포트 (자동 주입: $PORT → 기본 8080)
ENV PORT=8080
EXPOSE 8080

# 8️⃣ Gunicorn 실행 (운영 표준 방식)
#    - runserver 대신 안정적 WSGI 서버 gunicorn 사용
#    - :application 생략 → Cloud Run과 완벽 호환
#    - workers=2, timeout=0 → 모델 연산에도 대응
CMD ["bash", "-c", "python manage.py migrate --noinput && gunicorn aix_final_prj.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 0"]