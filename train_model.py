import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# 📁 파일 경로 설정
DATA_FILE = "ML_asos_dataset.csv"
MODEL_FILE = "trained_model.pkl"
FEATURE_FILE = "feature_names.pkl"

# ✅ 1. 데이터 불러오기
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

# ✅ 2. 전처리
X = df[["최고체감온도(°C)", "최고기온(°C)", "평균기온(°C)", "최저기온(°C)", "평균상대습도(%)"]]
y = df["환자수"]

# ✅ 3. 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 4. 모델 학습
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ✅ 5. 예측 및 성능 출력 (선택적)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance:")
print(f"MAE: {mae:.2f} / RMSE: {rmse:.2f} / R²: {r2:.4f}")

# ✅ 6. 모델 및 피처 저장
joblib.dump(model, MODEL_FILE)
joblib.dump(X.columns.tolist(), FEATURE_FILE)

print("✅ 모델 저장 완료:", MODEL_FILE, FEATURE_FILE)
