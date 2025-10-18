# langchain_tts_utils.py
# ==========================================
# 기존 기능 유지 + 문자내용/국가/성별 옵션 추가
# 설치: pip install langchain openai gTTS pydub python-dotenv playsound

import os
import json
import re
from typing import Dict, Optional
from django.conf import settings
from dotenv import load_dotenv
from gtts import gTTS
# from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from langchain import LLMChain, PromptTemplate
from langchain_openai import OpenAI
import platform


# 플랫폼별로 playsound import
try:
    if platform.system() in ["Darwin", "Linux"]:
        from playsound2 import playsound
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
    raise ValueError('OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일 또는 환경변수 필요')

# ------------------------
# 2. LLMChain 초기화
# ------------------------
llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

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
    patterns = {
        'ko': r'(?s)(?:"?ko"?|한국어)\s*[:\-]\s*(?:"([^"]+)")?(.+)',
        'en': r'(?s)(?:"?en"?|영어)\s*[:\-]\s*(?:"([^"]+)")?(.+)',
        'es': r'(?s)(?:"?es"?|스페인어)\s*[:\-]\s*(?:"([^"]+)")?(.+)'
    }
    for k, pat in patterns.items():
        m = re.search(pat, s)
        if m:
            text = m.group(1) if m.group(1) else m.group(2)
            text = text.strip()
            text = re.split(r"\n(?=[A-Za-z\"']|한국어|영어|스페인어)", text)[0].strip()
            out[k] = text.strip(' \"')
    return out

def get_translations(input_text: str) -> Dict[str, str]:
    """LLMChain으로 한국어, 영어, 스페인어 번역 결과를 JSON으로 반환"""
    resp = chain.invoke({'input_text': input_text})
    
    # LangChain 응답이 문자열이 아닐 경우 문자열로 변환
    if not isinstance(resp, str):
        if hasattr(resp, "content"):  # ChatMessage 유형
            resp = resp.content
        elif isinstance(resp, dict) and "text" in resp:
            resp = resp["text"]
        else:
            resp = str(resp)

    try:
        parsed = json.loads(extract_json_substring(resp))
    except Exception:
        parsed = fallback_parse(resp)
    return parsed

# ------------------------
# 4. TTS 유틸
# ------------------------
def tts_generate_play(text, lang_code, filename=None, use_pydub=False):
    if not filename:
        filename = f"tts_{lang_code}.mp3"
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


# ------------------------
# 4. TTS 유틸
# ------------------------
def tts_generate_save(text, lang_code, filename=None, use_pydub=False):
    if not filename:
        filename = f"tts_{lang_code}.mp3"
    tts = gTTS(text=text, lang=lang_code)
    filepath = os.path.join(settings.MEDIA_ROOT, filename)
    tts.save(filepath)
  
    return filename

# ------------------------
# 5. 번역 + TTS (옵션 기반)
# ------------------------
def translate_and_tts(
    text: str,               # 문자내용
    country: str = "ko",     # 국가 (언어코드)
    gender: str = "female",  # 성별
    translate_first: bool = True,
    use_pydub: bool = False
) -> Dict[str, str]:
    """
    문자내용, 국가, 성별 옵션으로 TTS 수행
    - text: 읽을 문장
    - country: 언어코드 ('ko', 'en', 'es', 'ja' 등)
    - gender: female/male (현재 gTTS는 female만 지원)
    - translate_first: True 시 LLMChain 번역 수행, False 시 원문 그대로
    """
    print(f"[INFO] translate_and_tts 실행 (country={country}, gender={gender})")

    # ✅ 번역 여부에 따라 문장 선택
    if translate_first:
        translations = get_translations(text)
        text_to_read = translations.get(country, text)
    else:
        text_to_read = text

    # ✅ 파일 이름 지정
    fname = f"tts_{country}_{gender}.mp3"

    # ✅ TTS 생성
    # tts_generate_play(text_to_read, lang_code=country, filename=fname, use_pydub=use_pydub)
    tts_name = tts_generate_save(text_to_read, lang_code=country, filename=fname, use_pydub=use_pydub)
    print(f"tts_url save {tts_name}")
    return {country: fname,"tts_name":tts_name}

# ------------------------
# 테스트 실행
# ------------------------
if __name__ == "__main__":
    msg = input("문자 입력: ").strip()
    if not msg:
        print("문장이 비어 있습니다. 종료합니다.")
        exit()

    country = input("언어코드 (ko/en/es): ").strip() or "ko"
    gender = input("성별 (female/male): ").strip() or "female"
    tf = input("번역 먼저 할까요? (y/n): ").strip().lower() != 'n'

    translate_and_tts(msg, country=country, gender=gender, translate_first=tf)
