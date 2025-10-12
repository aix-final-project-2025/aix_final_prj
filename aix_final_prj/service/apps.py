# service/apps.py
from django.apps import AppConfig
import os
from pathlib import Path
class ServiceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = "aix_final_prj.service"

    def ready(self):
        # 앱 시작 시 모델 및 클래스 로드
        from .efficient_net_v2m import load_model_and_classes, set_class_names
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        model_path = BASE_DIR / "keras" / "trash_classifier_efficientnetv2_best_final.keras"
        class_json = BASE_DIR / "keras" / "class_names.json"
        # model load
        try:
            load_model_and_classes(model_path=model_path)
            print(f'loader load_model_and_classes {model_path} ')
        except Exception as e:
            # 로컬 개발 시 모델 없으면 경고만 출력
            print("Warning: model load failed in AppConfig.ready():", e)

        # 클래스 파일이 있다면 설정
        try:
            import json, pathlib
            p = pathlib.Path(class_json)
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    class_names = json.load(f)
                set_class_names(class_names)

            print(f' loader class_json {p} ')    
        except Exception as e:
            print("Warning: class_names.json load failed:", e)
