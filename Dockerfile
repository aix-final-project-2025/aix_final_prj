# ===============================
# 🧠 AIX Final Project — Dockerfile
# Hugging Face Spaces optimized (manage.py entry)
# ===============================

FROM python:3.11-slim

# ---------------------------------
# 🧰 기본 패키지 설치 (디버깅 및 모니터링용)
# ---------------------------------
RUN apt update && apt install -y \
    procps \
    curl \
    nano \
    net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------
# 📁 작업 디렉토리 설정
# ---------------------------------
WORKDIR /app

# ---------------------------------
# 📦 전체 프로젝트 복사 (keras 포함)
# ---------------------------------
# 👉 이 한 줄로 모든 폴더가 복사되므로 keras_utils.py 절대경로도 자동 정합됨
COPY . .

# ---------------------------------
# 📜 Python 의존성 설치
# ---------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------
# ⚙️ 환경 변수 설정
# ---------------------------------
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# ---------------------------------
# 🔥 포트 노출
# ---------------------------------
EXPOSE 7860

# ---------------------------------
# 🚀 실행 명령 (manage.py 기준 실행)
# ---------------------------------
CMD ["bash", "-c", "python manage.py migrate --noinput && python app.py"]