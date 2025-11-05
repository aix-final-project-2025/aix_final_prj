# langchain_tts_utils.py
# ==========================================
# 기존 기능 유지 + 문자내용/국가/성별 옵션 추가
# 설치: pip install langchain openai gTTS pydub python-dotenv playsound (LangChain v0.x 버전 설치 필요)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 탐색 비활성화 > import os 바로 밑에 코드 위치해야
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 불필요한 로그 줄이기 > import os 바로 밑에 코드 위치해야

# ----------------------------------------------------
# 2 표준 라이브러리 & 외부 라이브러리
# ----------------------------------------------------

import json
import re
from typing import Dict
from django.conf import settings
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# ----------------------------------------------------
#  v0.x 호환성을 위해 import 경로 수정됨 
# ----------------------------------------------------
from langchain.chains import LLMChain             
from langchain.prompts import PromptTemplate      
from langchain_openai import ChatOpenAI
# ----------------------------------------------------

import platform

# ------------------------
# 플랫폼별로 playsound import
# ------------------------
try:
    if platform.system() in ["Darwin", "Linux"]:
        from playsound import playsound 
    else:
        from playsound import playsound
except ImportError:
    playsound = None  # fallback


# ------------------------
# 1. OpenAI API Key 로드
# ------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    # Django settings를 사용하는 환경일 경우, settings.OPENAI_API_KEY로 대체하거나,
    # .env 파일에 키를 설정해야 합니다.
    # raise ValueError('OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일 또는 환경변수 필요')
    pass 

# ------------------------
# 2. LLMChain 초기화
# ------------------------
prompt_template = PromptTemplate(
    input_variables=['input_text'],
    template=(
        '다음 한글 문장을 받아서 세 가지 항목을 포함하는 JSON을 출력하세요.\n'
        '1) "ko": 한국어(원문 자연스럽게 다듬기)\n'
        '2) "en": 영어 번역(의미 유지, 자연스럽게)\n'
        '3) "es": 스페인어 번역(의미 유지, 자연스럽게)\n'
        'JSON 형식만 출력하고 추가 설명하지 마세요.\n'
        'Input: {input_text}'
    )
)

llm = ChatOpenAI(
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo"
)

# LLMChain은 v0.x의 langchain.chains에서 가져옵니다.
chain = LLMChain(llm=llm, prompt=prompt_template)  

# ------------------------
# 3. JSON 파싱 유틸
# ------------------------
def extract_json_substring(s: str) -> str:
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f'응답에서 JSON을 찾을 수 없습니다:\n{s}')
    return s[start:end+1]

def fallback_parse(s: str) -> Dict[str, str]:
    out = {'ko': '', 'en': '', 'es': ''}
    # 정규식 패턴 수정 (문자열에 "나 '가 있을 경우 처리)
    patterns = {
        'ko': r'(?s)(?:"?ko"?|한국어)\s*[:\-]\s*(?:"?([^"]+)"?|([^,}\]]+))',
        'en': r'(?s)(?:"?en"?|영어)\s*[:\-]\s*(?:"?([^"]+)"?|([^,}\]]+))',
        'es': r'(?s)(?:"?es"?|스페인어)\s*[:\-]\s*(?:"?([^"]+)"?|([^,}\]]+))'
    }
    for k, pat in patterns.items():
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            # 그룹 1 (따옴표 있는 값) 또는 그룹 2 (따옴표 없는 값) 사용
            text = m.group(1) if m.group(1) else m.group(2)
            if text:
                text = text.split('\n')[0].strip()
                out[k] = text.strip(' \"')
    return out


def get_translations(input_text: str) -> Dict[str, str]:
    """LLMChain으로 한국어, 영어, 스페인어 번역 결과를 JSON으로 반환"""
    try:
        # v0.x에서 딕셔너리 {'text': '...'} 반환을 보장하는 안전한 호출 방식으로 수정
        resp = chain({"input_text": input_text}) 
    except Exception as e:
        print(f"chain error {e}")    
    # 응답에서 텍스트 추출
    # v0.3.x에서는 주로 딕셔너리 {'text': '...'} 형태로 반환됩니다.
    if isinstance(resp, dict) and "text" in resp:
        resp_text = resp["text"]
    else:
        resp_text = str(resp)
    
    try:
        parsed = json.loads(extract_json_substring(resp_text))
    except Exception:
        # JSON 파싱 실패 시 fallback 파싱 시도
        parsed = fallback_parse(resp_text)
        
    # 빈 값 검사 및 기본값으로 대체
    if not parsed.get('ko'): parsed['ko'] = input_text
    
    return parsed

# ------------------------
# 4. TTS 유틸
# ------------------------
# 변경: gender_key 파라미터를 추가하고 gTTS 대신 다른 TTS를 사용할 때를 대비하여 주석 처리
def tts_generate_play(text, lang_code, gender_key=None, filename=None, use_pydub=False):
    if not filename:
        filename = f"tts_{lang_code}_{gender_key or 'default'}.mp3"
    
    # G-TTS는 성별 옵션을 지원하지 않습니다. 이 키는 파일명에만 사용됩니다.
    #    실제 성별 출력을 원하시면 Google Cloud TTS, Naver Clova Voice 등으로 전환해야 합니다.
    tts = gTTS(text=text, lang=lang_code)
    tts.save(filename)
    
    try:
        if use_pydub or not playsound:
            sound = AudioSegment.from_file(filename, format="mp3")
            play(sound)
        else:
            playsound(filename)
    except Exception as e:
        print(f"[TTS 재생 오류] {e}")
    
    return filename

# 변경: gender_key 파라미터를 추가
def tts_generate_save(text, lang_code, gender_key=None, filename=None, use_pydub=False):
    if not filename:
        # 파일명에 gender_key를 포함
        filename = f"tts_{lang_code}_{gender_key or 'default'}.mp3"
        
    # settings.MEDIA_ROOT가 정의되어 있지 않을 경우를 대비하여 현재 디렉토리를 사용
    base_dir = getattr(settings, 'MEDIA_ROOT', os.getcwd())
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    filepath = os.path.join(base_dir, filename)
    # G-TTS는 성별 옵션을 지원하지 않습니다. 
    tts = gTTS(text=text, lang=lang_code)
    tts.save(filepath)
    return filename

# ------------------------
# 5. 번역 + TTS (옵션 기반)
# ------------------------
def translate_and_tts(
    text: str,
    country: str = "ko",
    gender: str = "female", # 성별 키값 기본값 유지
    translate_first: bool = True,
    use_pydub: bool = False
) -> Dict[str, str]:
    print(f"[INFO] translate_and_tts 실행 (country={country}, gender={gender})")

    if translate_first:
        try:
            translations = get_translations(text)
            text_to_read = translations.get(country, text)
        except Exception as e:
            print(f"[ERROR] 번역 중 오류 발생: {e}. 원본 텍스트로 TTS를 시도합니다.")
            text_to_read = text
    else:
        text_to_read = text

    # tts_generate_save 함수에 gender 키값 전달
    tts_name = tts_generate_save(
        text_to_read, 
        lang_code=country, 
        gender_key=gender, # gender_key 전달
        filename=None, 
        use_pydub=use_pydub
    )
    print(f"[INFO] TTS 저장 완료: {tts_name}")
    return {country: tts_name, "tts_name": tts_name} # tts_name을 country 키에도 반환

# ------------------------
# 테스트 실행
# ------------------------
if __name__ == "__main__":
    # Django settings가 없다고 가정하고 mock 처리
    class MockSettings:
        MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")
    settings.configure(MEDIA_ROOT=MockSettings.MEDIA_ROOT)
    
    if not os.path.exists(settings.MEDIA_ROOT):
        os.makedirs(settings.MEDIA_ROOT)
        
    print(f"저장 경로: {settings.MEDIA_ROOT}")

    msg = input("문자 입력: ").strip()
    if not msg:
        print("문장이 비어 있습니다. 종료합니다.")
        exit()

    country = input("언어코드 (ko/en/es): ").strip() or "ko"
    gender = input("성별 (female/male): ").strip() or "female"
    tf = input("번역 먼저 할까요? (y/n): ").strip().lower() != 'n'

    result = translate_and_tts(msg, country=country, gender=gender, translate_first=tf)
    print(f"\n최종 결과: {result}")
    
    # playsound 테스트는 주석 처리, 필요 시 주석 해제하여 사용
    # playsound(os.path.join(settings.MEDIA_ROOT, result['tts_name']))