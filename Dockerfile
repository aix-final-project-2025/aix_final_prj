# ============================
# 🧠 AIX Final Project (Cloud Run 배포용)
# ============================

FROM python:3.11-slim

# 1️⃣ 필수 패키지 설치
RUN apt update && apt install -y \
    curl procps nano net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 2️⃣ 작업 디렉토리
WORKDIR /app

# 3️⃣ 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ 소스 복사
COPY . .

# 5️⃣ 정적 파일
RUN python manage.py collectstatic --noinput

# 6️⃣ 환경 변수
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# 7️⃣ 포트 (Cloud Run은 기본적으로 $PORT 제공)
EXPOSE 8080

# 8️⃣ 실행 명령 (gunicorn 사용)
CMD exec gunicorn aix_final_prj.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 0