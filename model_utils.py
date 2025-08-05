import pandas as pd
import joblib
import math

# ✅ 모델 및 피처 로드
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ✅ 한국기상청 체감온도 계산 함수
def compute_kma_heat_index(t, rh, v=0.0):
    """
    t: 기온 (°C)
    rh: 상대습도 (%)
    v: 풍속 (m/s)
    """
    e = 6.105 * math.exp((17.27 * t) / (237.7 + t)) * rh / 100
    hi = t + 0.33 * e - 0.70 * v - 4.00
    return round(hi, 1)

# ✅ 예측 함수 (풍속 포함)
def predict_from_weather(tmx, tmn, reh, wind):
    """
    tmx: 최고기온 (°C)
    tmn: 최저기온 (°C)
    reh: 평균상대습도 (%)
    wind: 평균 풍속 (m/s)
    """
    avg_temp = round((tmx + tmn) / 2, 1)
    heat_index = compute_kma_heat_index(tmx, reh, wind)
    input_df = pd.DataFrame([{
        "최고체감온도(°C)": heat_index,
        "최고기온(°C)": tmx,
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": tmn,
        "평균상대습도(%)": reh,
        "풍속(m/s)": wind, 
    }])
    X = input_df[feature_names]
    pred = model.predict(X)[0]
    return pred, avg_temp, input_df
