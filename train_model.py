import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ✅ 파일 경로
STATIC_FILE = "ML_7_8월_2021_2025_dataset.xlsx"
DYNAMIC_FILE = "ML_asos_dataset.csv"
MODEL_FILE = "trained_model.pkl"
FEATURE_FILE = "feature_names.pkl"

# ✅ 1. 정적 데이터 불러오기
df_static = pd.read_excel(STATIC_FILE)

# ✅ 2. 동적 데이터 불러오기 (있을 경우)
if os.path.exists(DYNAMIC_FILE):
    try:
        df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="cp949")
    df = pd.concat([df_static, df_dynamic], ignore_index=True)
else:
    df = df_static.copy()

# ✅ 3. 결측치 제거
df = df.dropna()

# ✅ 4. 특성 및 타깃 정의
features = ['최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)']
X = df[features]
y = df['환자수']

# ✅ 5. 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ✅ 6. 모델 및 피처 저장
joblib.dump(model, MODEL_FILE)
joblib.dump(features, FEATURE_FILE)

print("🎉 학습 완료: 모델이 저장되었습니다.")
