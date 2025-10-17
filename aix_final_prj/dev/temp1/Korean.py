import pandas as pd
import os
import glob
from pydub import AudioSegment
import time

# =================================================================
# 1. 경로 설정 (고객님 경로 반영)
# =================================================================
# 다운로드한 AI Hub 데이터셋 파일들이 들어있는 최상위 폴더 경로
# (KsponSpeech_05 폴더 안의 모든 PCM/TXT 파일을 찾습니다.)
DATA_ROOT_DIR = 'C:\\Users\\Admin\\F프로젝트\\데이터\\KsponSpeech_05'

# 변환된 WAV 파일을 저장할 폴더 경로 (자동 생성됩니다.)
OUTPUT_WAV_DIR = 'C:\\Users\\Admin\\F프로젝트\\데이터\\re'

# 최종 학습용 CSV 파일을 저장할 경로 및 파일명
OUTPUT_CSV_PATH = 'C:\\Users\\Admin\\F프로젝트\\데이터\\reCSV\\stt_training_dataset.csv'

# WAV 저장 폴더가 없으면 생성
if not os.path.exists(OUTPUT_WAV_DIR):
    os.makedirs(OUTPUT_WAV_DIR)
    print(f"WAV 저장 폴더 생성: {OUTPUT_WAV_DIR}")

# CSV 저장 폴더가 없으면 생성
if not os.path.exists(os.path.dirname(OUTPUT_CSV_PATH)):
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH))
    print(f"CSV 저장 폴더 생성: {os.path.dirname(OUTPUT_CSV_PATH)}")


# =================================================================
# 2. PCM 파일 속성 설정 (KsponSpeech 기준, 데이터 문서 확인 필수)
# =================================================================
# KsponSpeech는 보통 16kHz, 16비트, 모노입니다.
SAMPLE_RATE = 16000  # 16000 Hz
SAMPLE_WIDTH = 2     # 16비트 (2바이트)
CHANNELS = 1         # 모노 (Mono)
FORMAT = "s16le"     # 16비트 리틀엔디언 포맷

# --- (이하 나머지 코드는 이전과 동일하게 유지됩니다.) ---
# 3. 데이터 처리 함수 (process_data_directory)
# 4. 실행 및 저장 로직
# -----------------------------------------------------------------