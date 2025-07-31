import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

print("📂 현재 디렉토리:", os.getcwd())
print("📄 파일 목록:", os.listdir())

# 파일명 정확히 확인
STATIC_FILE = "ML_7_8월_2021_2025_dataset.xlsx"
DYNAMIC_FILE = "ML_asos_dataset.csv"

if not os.path.exists(STATIC_FILE):
    print(f"❌ {STATIC_FILE} 파일이 없습니다.")
    exit(1)

df_static = pd.read_excel(STATIC_FILE)
print("✅ 정적 데이터 로드 완료:", df_static.shape)

if os.path.exists(DYNAMIC_FILE):
    df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="utf-8-sig")
    print("✅ 동적 데이터 로드 완료:", df_dynamic.shape)
    df = pd.concat([df_static, df_dynamic], ignore_index=True)
else:
    print("⚠️ 동적 데이터 없음 → 정적 데이터만 사용")
    df = df_static.copy()

print("📊 결합 후 전체 행 수:", len(df))
df = df.dropna()
print("🧹 dropna 후 행 수:", len(df))

features = ['최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)']
target = '환자수'

if len(df) == 0 or not all(col in df.columns for col in features + [target]):
    print("❌ 학습 가능한 데이터가 없습니다.")
    exit(1)

X = df[features]
y = df[target]

model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X, y)

joblib.dump(model, "trained_model.pkl")
joblib.dump(features, "feature_names.pkl")
print("✅ 모델 학습 및 저장 완료")
