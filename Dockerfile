# ============================
# 🧠 AIX Final Project (Render 배포용)
# ============================

FROM python:3.11-slim

# 1️⃣ 필수 패키지 설치 (TensorFlow CPU, Django 실행 최소 유틸)
RUN apt update && apt install -y \
    curl \
    procps \
    nano \
    net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ 작업 디렉토리
WORKDIR /app

# 3️⃣ 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ 앱 파일 복사
COPY . .

# 5️⃣ 정적 파일 수집
RUN python manage.py collectstatic --noinput

# 6️⃣ 환경 변수 (Render 기본 세팅)
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings
ENV PORT=10000

# 7️⃣ Render에서 감지할 포트
EXPOSE 10000

# 8️⃣ 실행 명령 
CMD ["bash", "-c", "python manage.py migrate --noinput && gunicorn aix_final_prj.wsgi:application --bind 0.0.0.0:$PORT"]