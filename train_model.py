import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 추론 함수
from model_utils import predict_from_weather

# ✅ 파일 경로
STATIC_FILE = "ML_static_dataset.csv"  
DYNAMIC_FILE = "ML_asos_dataset.csv"
MODEL_FILE = "trained_model.pkl"
FEATURE_FILE = "feature_names.pkl"

print("📂 현재 디렉토리:", os.getcwd())
print("📄 파일 목록:", os.listdir())

# ✅ 정적 데이터 로드 (CSV)
if not os.path.exists(STATIC_FILE):
    print(f"❌ 정적 데이터 파일이 없습니다: {STATIC_FILE}")
    exit(1)

df_static = pd.read_csv(STATIC_FILE, encoding="cp949")
print(f"✅ 정적 데이터 로드 완료: {df_static.shape}")

# 🔧 열 이름 정제
df_static.columns = df_static.columns.str.strip().str.replace('\n', '').str.replace(' ', '')

# 🔧 '일자' 생성
if '일시' in df_static.columns and pd.api.types.is_numeric_dtype(df_static['일시']):
    df_static['일자'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df_static['일시'], unit='D')
    df_static['일자'] = df_static['일자'].dt.strftime('%Y-%m-%d')
elif '일시' in df_static.columns:
    df_static['일자'] = pd.to_datetime(df_static['일시'], errors='coerce').dt.strftime('%Y-%m-%d')

# 🔧 '지역' 통일
for col in ['광역자치단체', '지역', '시도']:
    if col in df_static.columns:
        df_static['지역'] = df_static[col]
        break

# 🔧 불필요한 열 제거
df_static = df_static.drop(columns=[col for col in ['일시', '광역자치단체', '시도'] if col in df_static.columns])

# ✅ 동적 데이터 로드
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

# ✅ 열 이름 다시 정제 (전체 통일)
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace(' ', '')

# ✅ 결측치 제거 대상
required_columns = [
    '일자', '지역', '최고체감온도(°C)', '최고기온(°C)',
    '평균기온(°C)', '최저기온(°C)', '평균상대습도(%)', '환자수'
]
print("\n📌 결측치 개수:")
print(df[required_columns].isna().sum())

df = df.dropna(subset=required_columns)
print("🧹 결측치 제거 후 행 수:", len(df))

# ✅ 집계
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
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5  # ✔ squared=False 사용 안 함 (버전 호환성)

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
