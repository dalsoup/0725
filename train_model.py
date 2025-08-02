import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ✅ 파일 경로
STATIC_FILE = "ML_7_8월_2021_2025_dataset.xlsx"
DYNAMIC_FILE = "ML_asos_dataset.csv"
MODEL_FILE = "trained_model.pkl"
FEATURE_FILE = "feature_names.pkl"

print("📂 현재 디렉토리:", os.getcwd())
print("📄 파일 목록:", os.listdir())

# ✅ 정적 데이터 로드
if not os.path.exists(STATIC_FILE):
    print(f"❌ {STATIC_FILE} 파일이 없습니다.")
    exit(1)
df_static = pd.read_excel(STATIC_FILE)
print("✅ 정적 데이터 로드 완료:", df_static.shape)

# ✅ 동적 데이터 로드 (있는 경우만)
if os.path.exists(DYNAMIC_FILE):
    try:
        df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="cp949")
    print("✅ 동적 데이터 로드 완료:", df_dynamic.shape)
    df = pd.concat([df_static, df_dynamic], ignore_index=True)
else:
    print("⚠️ 동적 데이터 없음 → 정적 데이터만 사용")
    df = df_static.copy()

print("📊 결합 후 전체 행 수:", len(df))

# ✅ 열 이름 정제
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace(' ', '')

# ✅ 결측치 제거 대상 열만 지정
required_columns = ['일자', '지역', '최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)', '환자수']
print("\n📌 결측치 개수:")
print(df[required_columns].isna().sum())

df = df.dropna(subset=required_columns)
print("🧹 dropna 후 행 수:", len(df))

# ✅ 일자 + 지역 단위로 평균/합계 집계
grouped = df.groupby(['일자', '지역']).agg({
    '최고체감온도(°C)': 'mean',
    '최고기온(°C)': 'mean',
    '평균기온(°C)': 'mean',
    '최저기온(°C)': 'mean',
    '평균상대습도(%)': 'mean',
    '환자수': 'sum'  # 🔥 핵심: 자치구 환자수를 광역시 단위로 합산
}).reset_index()

# ✅ 피처 및 타겟 정의
features = ['최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)']
target = '환자수'

# ✅ 학습 가능성 검사
if len(grouped) == 0 or not all(col in grouped.columns for col in features + [target]):
    print("❌ 학습 가능한 데이터가 없습니다.")
    exit(1)

X = grouped[features]
y = grouped[target]

# ✅ 모델 학습 (XGBoost 사용)
model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X, y)

# ✅ 저장
joblib.dump(model, MODEL_FILE)
joblib.dump(features, FEATURE_FILE)
print("✅ 모델 학습 및 저장 완료")
