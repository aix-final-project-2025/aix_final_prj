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

# 4️⃣ 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ 소스 코드 복사
COPY . .

# 6️⃣ 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings

# ✅ 7️⃣ Hugging Face 기본 포트 설정
EXPOSE 7860

# ✅ 8️⃣ runserver 포트를 7860으로 변경
CMD ["python", "-u", "manage.py", "runserver", "0.0.0.0:7860"]
