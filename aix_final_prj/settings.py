import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


TRASH_MODEL_PATH = os.path.join(BASE_DIR, "keras", "trash_classifier_efficientnetv2_best_final.keras")
TRASH_CLASS_NAMES_PATH = os.path.join(BASE_DIR, "keras", "class_names.json")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-j#blyrg*)(8ml$)7zxozibkb61#64cx&3ztg^6*mrs^u7_dr^o'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
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

# CORS - 앱/모바일 테스트 시 임시 허용
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


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

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


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'ko-kr'

TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

MEDIA_URL = '/media/' # 추가함
MEDIA_ROOT = os.path.join(BASE_DIR, 'uploads') # 추가함
os.makedirs(MEDIA_ROOT, exist_ok=True)
# REST Framework (기본 설정)
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    )
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "static", # 전역 static 폴더
    BASE_DIR / "core" / "static",
]

# ✅ Docker, Cloud Run 등 배포 환경에서는 필요
STATIC_ROOT = BASE_DIR / 'staticfiles'

# ✅ WhiteNoise 설정 (Dockerfile에서 collectstatic 후 서빙)
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"


# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db_new"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "my_notes")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", GOOGLE_API_KEY) 
GEMINI_EMBED_MODELS = os.environ.get("GEMINI_EMBED_MODELS", "text-embedding-004").split(",")
GEMINI_TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
WEB_INGEST_TO_CHROMA = os.environ.get("WEB_INGEST_TO_CHROMA", "1").lower() not in ("0", "false", "no")
CRAWL_ANSWER_LINKS = os.environ.get("CRAWL_ANSWER_LINKS", "1").lower() not in ("0", "false", "no")
NEWS_TOPK = os.environ.get("NEWS_TOPK", "5")
RAG_QUERY_TOPK = os.environ.get("RAG_QUERY_TOPK", "5")
RAG_FALLBACK_TOPK = os.environ.get("RAG_FALLBACK_TOPK", "12")
RAG_AUTO_SEED_IF_EMPTY = os.environ.get("RAG_AUTO_SEED_IF_EMPTY", "1").lower() not in ("0", "false", "no")

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
