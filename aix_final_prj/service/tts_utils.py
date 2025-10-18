# langchain_tts_utils.py
# =========================
# 한글 입력 → 다국어 번역 → TTS 생성/재생
# gTTS + LangChain(OpenAI) 통합 공통 모듈
# 설치: pip install langchain openai gTTS pydub python-dotenv playsound

import os
import json
import re
from typing import Dict, Optional
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from langchain import LLMChain, PromptTemplate
from langchain_openai import OpenAI

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
    resp = chain.run({'input_text': input_text})
    try:
        parsed = json.loads(extract_json_substring(resp))
    except Exception:
        parsed = fallback_parse(resp)
    return parsed

# ------------------------
# 4. TTS 유틸
# ------------------------
def tts_generate_play(
    text: str,
    lang_code: str,
    filename: Optional[str] = None,
    use_pydub: bool = False
) -> str:
    """
    text: 읽을 텍스트
    lang_code: 'ko','en','es' 등 ISO 639-1 코드
    filename: 저장 파일명. 미지정 시 tts_{lang_code}.mp3 자동 생성
    use_pydub: True -> pydub 재생, False -> playsound 재생
    """
    if not filename:
        filename = f'tts_{lang_code}.mp3'
    tts = gTTS(text=text, lang=lang_code)
    tts.save(filename)
    
    try:
        if use_pydub:
            sound = AudioSegment.from_file(filename, format='mp3')
            play(sound)
        else:
            playsound(filename)
    except Exception as e:
        print(f'[TTS 재생 오류] lang={lang_code}, file={filename}, error={e}')
    
    return filename

# ------------------------
# 5. 전체 프로세스 (입력 → 번역 → TTS)
# ------------------------
def translate_and_tts(input_text: str, use_pydub: bool = False) -> Dict[str, str]:
    """
    input_text: 한국어 입력
    use_pydub: TTS 재생 방식
    return: 언어별 생성 파일명 dict
    """
    translations = get_translations(input_text)
    files = {}
    for lang_code, text in translations.items():
        if not text:
            text = input_text  # fallback
        fname = f'tts_{lang_code}.mp3'
        tts_generate_play(text, lang_code, filename=fname, use_pydub=use_pydub)
        files[lang_code] = fname
    return files

# ------------------------
# 단독 실행 테스트
# ------------------------
if __name__ == "__main__":
    user_input = input("한글 문장 입력: ").strip()
    if not user_input:
        print("문장이 비어있습니다. 종료합니다.")
        exit()
    
    print("\n== 번역 + TTS 생성 중 ==")
    files = translate_and_tts(user_input)
    print("\n생성 완료:")
    print(files)
