import pandas as pd

file_path = r"C:\Users\Admin\aix_final_prj\aix_final_prj\dev\temp1\synthetic_coffee_health_10000.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f" 오류: '{file_path}' 파일을 찾을 수 없습니다.")

    
print(df.head())
df.info()

# ----------------- 한국 확인
df_korea = df[df['Country'] == 'South Korea']
df_korea
print(df_korea.shape[0])
# ------------------------

