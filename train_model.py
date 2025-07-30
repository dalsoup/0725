import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# 파일 불러오기
df = pd.read_csv("ML_asos_dataset.csv")

# 피처와 타겟 분리
X = df.drop(columns=["일자", "지역", "환자수"])
y = df["환자수"]

# 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ 버전 이슈 방지
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 4))

# 모델 및 피처 저장
joblib.dump(model, "trained_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
