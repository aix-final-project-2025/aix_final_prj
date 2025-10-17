import os
import subprocess
import sys
import time

# ==========================
# 🧠 AIX Final Project — app.py
# Hugging Face용 Django 실행 엔트리포인트
# ==========================

if __name__ == "__main__":
    print("🚀 Starting AIX Final Project on Hugging Face...")
    print(f"🔧 Working Directory: {os.getcwd()}")
    print(f"🐍 Python Executable: {sys.executable}")
    sys.stdout.flush()

    # 환경 변수 설정 (혹시 누락 방지용)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aix_final_prj.settings")

    # manage.py runserver 실행
    try:
        process = subprocess.Popen(
            ["python", "manage.py", "runserver", "0.0.0.0:7860"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True
        )

        # 실시간 로그 안정화
        while True:
            retcode = process.poll()
            if retcode is not None:
                print(f"❌ Django process exited with code {retcode}")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("🛑 Manual stop received. Shutting down gracefully...")
    except Exception as e:
        print(f"⚠️ Error while starting Django: {e}")
    finally:
        print("👋 AIX Final Project stopped.")
