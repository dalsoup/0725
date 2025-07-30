import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_excel("ML_7_8월_2021_2025_dataset.xlsx")

# 결측값 제거
df = df.dropna()

# ✅ 컬럼명 통일 (예측 코드와 일치시킴)
df = df.rename(columns={
    "습도(%)": "평균상대습도(%)"
})

# 입력 변수와 타겟 정의
features = ['최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)']
X = df[features]
y = df['환자수']

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 모델 및 피처 저장
joblib.dump(model, "trained_model.pkl")
joblib.dump(features, "feature_names.pkl")
