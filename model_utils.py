import pandas as pd
import joblib

# 모델 및 피처 로드
model = joblib.load("trained_model.pkl")
feature_names = joblib.load("feature_names.pkl")

def predict_from_weather(tmx, tmn, reh, heat_index):
    avg_temp = round((tmx + tmn) / 2, 1)
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
