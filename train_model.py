import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 추론 함수 불러오기
from model_utils import predict_from_weather

# ✅ 파일 경로
STATIC_FILE = "ML_7_8월_2021_2025_dataset.xlsx"
DYNAMIC_FILE = "ML_asos_dataset.csv"
MODEL_FILE = "trained_model.pkl"
FEATURE_FILE = "feature_names.pkl"

print("📂 현재 디렉토리:", os.getcwd())
print("📄 파일 목록:", os.listdir())

# ✅ 정적 데이터 로드
if not os.path.exists(STATIC_FILE):
    print(f"❌ 정적 데이터 파일이 없습니다: {STATIC_FILE}")
    exit(1)

df_static = pd.read_excel(STATIC_FILE)
print(f"✅ 정적 데이터 로드 완료: {df_static.shape}")

# ✅ 동적 데이터 로드 (선택적)
if os.path.exists(DYNAMIC_FILE):
    try:
        df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_dynamic = pd.read_csv(DYNAMIC_FILE, encoding="cp949")
    print(f"✅ 동적 데이터 로드 완료: {df_dynamic.shape}")
    df = pd.concat([df_static, df_dynamic], ignore_index=True)
else:
    print("⚠️ 동적 데이터 없음 → 정적 데이터만 사용")
    df = df_static.copy()

print(f"📊 결합 후 전체 행 수: {len(df)}")

# ✅ 열 이름 정제
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace(' ', '')

# ✅ 결측치 제거 대상 열 지정
required_columns = [
    '일자', '지역', '최고체감온도(°C)', '최고기온(°C)', 
    '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)', '환자수'
]
print("\n📌 결측치 개수:")
print(df[required_columns].isna().sum())

df = df.dropna(subset=required_columns)
print("🧹 결측치 제거 후 행 수:", len(df))

# ✅ 집계 (일자+지역 단위)
grouped = df.groupby(['일자', '지역']).agg({
    '최고체감온도(°C)': 'mean',
    '최고기온(°C)': 'mean',
    '평균기온(°C)': 'mean',
    '최저기온(°C)': 'mean',
    '평균상대습도(%)': 'mean',
    '환자수': 'sum'
}).reset_index()
print(f"📊 집계 완료: {grouped.shape}")

# ✅ 피처 및 타겟 정의
features = [
    '최고체감온도(°C)', '최고기온(°C)', '평균기온(°C)', 
    '최저기온(°C)', '평균상대습도(%)'
]
target = '환자수'

# ✅ 학습 가능성 검사
if len(grouped) == 0 or not all(col in grouped.columns for col in features + [target]):
    print("❌ 학습 가능한 데이터가 없습니다.")
    exit(1)

X = grouped[features]
y = grouped[target]

# ✅ 모델 학습
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X, y)

# ✅ 성능 평가
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print("\n📈 모델 성능 평가")
print(f"  - R²: {r2:.4f}")
print(f"  - RMSE: {rmse:.4f}")

# ✅ 저장
joblib.dump(model, MODEL_FILE)
joblib.dump(features, FEATURE_FILE)
print(f"\n✅ 모델 및 피처 저장 완료 → '{MODEL_FILE}', '{FEATURE_FILE}'")
print(f"🧠 사용된 피처: {features}")

# ✅ 추론 함수 테스트
print("\n🧪 예측 함수 연동 테스트 (predict_from_weather)")
sample_tmx = 34.0
sample_tmn = 26.0
sample_reh = 70.0

pred, avg_temp, heat_index, input_df = predict_from_weather(sample_tmx, sample_tmn, sample_reh)

print(f"  - 입력: TMX={sample_tmx}, TMN={sample_tmn}, REH={sample_reh}")
print(f"  - 평균기온: {avg_temp:.2f}°C")
print(f"  - 체감온도: {heat_index:.2f}°C")
print(f"  - 예측 환자 수: {pred:.2f}명")
print(f"  - 모델 입력 벡터:")
print(input_df)
