import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


TRASH_MODEL_PATH = os.path.join(BASE_DIR, "keras", "trash_classifier_efficientnetv2_best_final.keras")
TRASH_CLASS_NAMES_PATH = os.path.join(BASE_DIR, "keras", "class_names.json")

SECRET_KEY = 'django-insecure-j#blyrg*)(8ml$)7zxozibkb61#64cx&3ztg^6*mrs^u7_dr^o'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'aix_final_prj.service.apps.ServiceConfig', # 선언은 service/apps.py를 구동시킴
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # CORS
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

]

CORS_ALLOW_ALL_ORIGINS = True

ROOT_URLCONF = 'aix_final_prj.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],   # 전역 templates 폴더
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'aix_final_prj.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'ko-kr'

TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_TZ = True

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    )
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ✅ 미디어 파일 (사용자 업로드용)
MEDIA_URL = '/media/'  # 브라우저 접근 경로 (예: /media/filename.jpg)
MEDIA_ROOT = os.path.join(BASE_DIR, 'uploads')  # 실제 저장 폴더
os.makedirs(MEDIA_ROOT, exist_ok=True)  # 폴더 없으면 자동 생성

# ✅ 정적 파일 (CSS, JS, 이미지 등)
STATIC_URL = '/static/'  # 정적파일 접근 경로
STATICFILES_DIRS = [
    BASE_DIR / "static",  # 개발 중 사용할 정적 폴더
]
STATIC_ROOT = BASE_DIR / "staticfiles"  # collectstatic 시 모이는 폴더