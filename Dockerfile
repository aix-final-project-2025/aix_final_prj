# ============================
# 🧠 AIX Final Project (Render 배포용)
# ============================

FROM python:3.11-slim

RUN apt update && apt install -y \
    curl procps nano net-tools \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ static 수집 (WhiteNoise가 서빙할 파일 모음)
RUN python manage.py collectstatic --noinput

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=aix_final_prj.settings
ENV PORT=10000

EXPOSE 10000

# ✅ gunicorn으로 실행
CMD ["bash", "-c", "python manage.py migrate --noinput && gunicorn aix_final_prj.wsgi:application --bind 0.0.0.0:$PORT"]