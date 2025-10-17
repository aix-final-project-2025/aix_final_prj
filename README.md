---
title: AIX Final Project
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 AIX Final Project (Recycling Image Classifier)

> ♻️ 재활용품 이미지 분류 및 친환경 행동 유도 AI 웹서비스  
> Django + EfficientNetV2M + Hugging Face Spaces + Docker 기반  

---

## 🚀 프로젝트 개요

이 프로젝트는 **재활용품 분류 이미지 AI 모델**을 웹서비스 형태로 제공합니다.  
사용자는 이미지를 업로드하여 쓰레기 종류(플라스틱, 종이, 캔 등)를 예측받을 수 있습니다.  
모델은 `EfficientNetV2-M` 기반으로 학습되었으며,  
TensorFlow + Django를 이용해 백엔드와 프론트엔드를 통합했습니다.

---

## 🧩 기술 스택

| 분류 | 사용 기술 |
|------|------------|
| 🧱 백엔드 | Django 5.2.7 |
| 🎨 프론트엔드 | HTML, CSS, JS (Django Template) |
| 🧠 모델 | TensorFlow 2.x (`EfficientNetV2M`) |
| 🐳 배포 | Docker + Hugging Face Spaces |
| ☁️ 저장소 | GitHub / Hugging Face |

---

## 🧠 모델 구조

- 모델 경로:  
  `aix_final_prj/keras/trash_classifier_efficientnetv2_best_final.keras`

- 클래스 이름 파일:  
  `aix_final_prj/keras/class_names.json`  
  (없을 경우 인덱스 기반 자동 분류)

---

## 🧾 실행 방식

이 Space는 Docker 기반으로 구성되어 있으며,  
빌드 후 아래 명령어로 Django 서버를 실행합니다 👇

```bash
python manage.py runserver 0.0.0.0:7860

aix_final_prj/
 ├── keras/                      # 학습된 모델 및 클래스 JSON
 ├── service/                    # 모델 로더 및 inference 코드
 ├── dev/                        # 실험용 노트북 및 분석 스크립트
 ├── settings.py                 # Django 설정
 └── urls.py                     # 라우팅 설정
core/
 ├── templates/                  # HTML 템플릿
 ├── static/                     # CSS/JS/이미지
 └── views.py                    # 페이지 뷰
Dockerfile
requirements.txt
manage.py

