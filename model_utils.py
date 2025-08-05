import pandas as pd
import joblib
import math

# 모델 및 피처 로드
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")

def compute_kma_heat_index(t, rh, v=0.0):
    e = 6.105 * math.exp((17.27 * t) / (237.7 + t)) * rh / 100
    hi = t + 0.33 * e - 0.70 * v - 4.00
    return round(hi, 1)

def predict_from_weather(tmx, tmn, reh):
    avg_temp = round((tmx + tmn) / 2, 1)
    heat_index = compute_kma_heat_index(tmx, reh)
    input_df = pd.DataFrame([{ 
        "최고체감온도(°C)": heat_index,
        "최고기온(°C)": tmx,
        "평균기온(°C)": avg_temp,
        "최저기온(°C)": tmn,
        "평균상대습도(%)": reh
    }])
    X = input_df[feature_names] 
    pred = model.predict(X)[0]
    return pred, avg_temp, input_df
