# ============================
# 🧠 AIX Final Project (Cloud Run 배포용 최종본)
# ============================

FROM python:3.11-slim

# 1️⃣ 필수 시스템 패키지 설치 (TensorFlow CPU & Django 실행 필수 유틸)
RUN apt update && apt install -y \
    curl \
    procps \
    nano \
    net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ 작업 디렉토리 설정
WORKDIR /app

# 3️⃣ 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ 소스 코드 복사
COPY . .

# 5️⃣ 정적 파일 수집 (CSS, JS 등)
RUN python manage.py collectstatic --noinput

# 6️⃣ 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# 7️⃣ Cloud Run용 포트 노출 ($PORT 자동 제공됨, 기본 8080)
EXPOSE 8080

# 8️⃣ Gunicorn 서버 실행 (Cloud Run 전용)
#    - $PORT: Cloud Run이 자동으로 주입하는 포트
#    - --workers 2: 요청 병렬 처리
#    - --timeout 0: 장시간 연산 모델 대응
CMD ["bash", "-c", "python manage.py migrate --noinput && gunicorn aix_final_prj.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 0"]