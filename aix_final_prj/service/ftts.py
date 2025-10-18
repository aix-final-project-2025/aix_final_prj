from googletrans import Translator
from gtts import gTTS
import os

# 1. 한국어 텍스트 입력
korean_text = "안녕하세요! 이것은 번역 후 TTS 예제입니다."

# 2. 번역
translator = Translator()

translations = {
    "en": translator.translate(korean_text, src='ko', dest='en').text,
    "es": translator.translate(korean_text, src='ko', dest='es').text
}

# 3. TTS 변환 및 재생
for lang, text in translations.items():
    filename = f"tts_{lang}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f"{filename} 생성 완료! 텍스트: {text}")

    # OS별 재생
    if os.name == 'nt':  # Windows
        os.system(f"start {filename}")
    elif os.uname().sysname == 'Darwin':  # macOS
        os.system(f"afplay {filename}")
    else:  # Linux
        os.system(f"mpg123 {filename}")
