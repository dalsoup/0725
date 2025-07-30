import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. 기존 학습 데이터 로드 (엑셀)
base_df = pd.read_excel("ML_7_8월_2021_2025_dataset.xlsx")

# 2. 추가된 최신 ASOS 기반 학습 데이터 로드 (CSV)
try:
    asos_df = pd.read_csv("ML_asos_dataset.csv")
    asos_df = asos_df.rename(columns={
        "max_temp": "최고기온(°C)",
        "min_temp": "최저기온(°C)",
        "avg_temp": "평균기온(°C)",
        "avg_rh": "평균상대습도(%)",
        "환자수": "환자수"
    })
    asos_df["최고체감온도(°C)"] = asos_df["최고기온(°C)"] + 1.5
    combined_df = pd.concat([base_df, asos_df], ignore_index=True)
except FileNotFoundError:
    combined_df = base_df  # 추가 파일 없으면 기존 데이터만 사용

# 3. 결측 제거
combined_df = combined_df.dropna()

# 4. 입력 변수 및 타겟 설정
features = ['최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)']
X = combined_df[features]
y = combined_df['환자수']

# 5. 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. 모델 저장
joblib.dump(model, "trained_model.pkl")
joblib.dump(features, "feature_names.pkl")

print("✅ 모델 재학습 완료. 총 학습 데이터 수:", len(combined_df))
